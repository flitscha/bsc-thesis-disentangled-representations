"""
Pipeline step §5.6: loop and connected-component detection via topological data
analysis (persistent homology, gudhi).

Operates on a precomputed distance matrix (built in §5.4, see graph.py) and
returns the detected structure as index orders.
"""

import numpy as np
import gudhi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from core.ordering import order_along_path


def compute_persistence(distance_matrix, max_dimension=1, max_edge_length=None):
    if max_edge_length is None:
        max_edge_length = float(np.max(distance_matrix)) * 1.01

    rips = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension + 1)
    diagram = simplex_tree.persistence()
    generators = simplex_tree.flag_persistence_generators()
    return simplex_tree, diagram, generators


def significant_bars(diagram, dim, min_persistence):
    """
    Bars of homology dimension 'dim' with persistence (death - birth) at least 'min_persistence'.
    Infinite bars always count as significant.
    """
    bars = [(b, d) for (dd, (b, d)) in diagram if dd == dim]
    infinite = [(b, d) for (b, d) in bars if not np.isfinite(d)]
    kept = [(b, d) for (b, d) in bars if np.isfinite(d) and (d - b) >= min_persistence]
    kept.sort(key=lambda bd: -(bd[1] - bd[0]))
    return infinite + kept


def _h0_num_components(diagram, gap_factor):
    """
    Number of connected components read off the H0 barcode by its largest gap.

    H0 deaths are the scales at which components merge (their birth is 0).
    Within one structure they cluster around the typical spacing between
    neighbouring components. A merge that joins two genuinely separate
    structures sits well above that cluster, leaving a visible gap in the
    barcode.
    We cut at the largest gap between sorted deaths and count every death
    above it as an extra component.
    The cut only fires if that gap is at least 'gap_factor' times the median
    death, so a single well-connected structure stays one component.
    """
    deaths = np.array(sorted(d for (dim, (b, d)) in diagram if dim == 0 and np.isfinite(d)))
    if deaths.size < 2:
        return 1
    gaps = np.diff(deaths)
    i = int(np.argmax(gaps))
    if gaps[i] >= gap_factor * np.median(deaths):
        return 1 + (deaths.size - (i + 1))
    return 1


def extract_components(distance_matrix, n_components):
    """
    Assign each node a connected-component label (0 .. n_components-1).

    Single-linkage clustering is exactly the H0 filtration of the Rips complex
    (its merge scales are the H0 deaths), so we cut its dendrogram into the
    n_components clusters that persistent homology already told us to expect.
    """
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="single")
    labels = fcluster(Z, t=n_components, criterion="maxclust")
    return labels - 1


def extract_loop(distance_matrix, birth_edge, birth_scale):
    """
    Reconstruct a representative cycle for an H1 loop as an ordered node list.

    Persistent homology reports a loop only abstractly (a birth/death pair); this
    turns it back into a concrete ordering of the components around the hole.

    Parameters
    ----------
    distance_matrix : (N, N) array
        Pairwise distances between components (§5.4).
    birth_edge : (u, v)
        The H1 birth edge that closes the loop.
    birth_scale : float
        Filtration scale at which the loop is born, i.e. distance_matrix[u, v].

    Returns
    -------
    path : list of int
        Component indices in order around the cycle (u ... v, u).
    """
    u, v = int(birth_edge[0]), int(birth_edge[1])

    adj = distance_matrix.copy()
    adj[adj > birth_scale] = 0
    np.fill_diagonal(adj, 0)
    adj[u, v] = adj[v, u] = 0

    dist, predecessors = dijkstra(csr_matrix(adj), directed=False, indices=u, return_predecessors=True)
    if not np.isfinite(dist[v]):
        raise RuntimeError(f"Could not reconstruct loop: {u} and {v} not connected below birth scale.")

    path = [v]
    cur = v
    while cur != u:
        cur = predecessors[cur]
        path.append(cur)
    path.reverse()
    path.append(u)
    return path


def detect_tda(
    distance_matrix, min_persistence_h0=None, min_persistence_h1=None,
    component_gap_factor_h0=0.5, prominence_ratio_h1=2.0,
):
    """
    Detect connected components (H0) and loops (H1) from a distance matrix
    via topological data analysis (persistent homology).

    Parameters
    ----------
    distance_matrix : (N, N) array
        Pairwise distances between the MFA components.
    min_persistence_h0, min_persistence_h1 : float or None
        Fixed persistence thresholds. None (default) uses the automatic,
        scale-free criteria below.
    component_gap_factor_h0 : float
        Auto H0 threshold: split components at the largest gap in the H0
        barcode, provided that gap is at least this multiple of the median
        merge scale.
    prominence_ratio_h1 : float
        Auto H1 threshold: keep a loop only if death >= ratio * birth, i.e. the
        hole is much larger than the local scale at which the cycle closes.

    Returns
    -------
    result : dict
        components : (N,) int array
            Connected-component label for each node.
        curves : list of dict
            One entry per detected structure, each with a "type" ("loop" or
            "path"), its "component" label, and the "order" of node indices
            along it. Loops additionally carry "birth", "death" and
            "persistence".
        diagram : list
            The full persistence diagram (H0 and H1).
        diagram_h0 : list
            Its H0 part only (the connected-component barcode).
    """
    D = distance_matrix
    _, diagram, generators = compute_persistence(D, max_dimension=1)

    # H0: connected components
    if min_persistence_h0 is not None:
        n_components = len(significant_bars(diagram, dim=0, min_persistence=min_persistence_h0))
    else:
        n_components = _h0_num_components(diagram, gap_factor=component_gap_factor_h0)
    components = extract_components(D, n_components=n_components)

    # H1: loop detection
    # Each generator row is one distinct H1 class.
    # It is given by the birth edge (row[0], row[1]) and the death edge (row[2], row[3])
    gen_rows = generators[1][0] if len(generators[1]) > 0 else np.empty((0, 4), dtype=int)
    loops = []
    for row in gen_rows:
        u, v = int(row[0]), int(row[1])
        birth = D[u, v]
        death = D[int(row[2]), int(row[3])]
        if min_persistence_h1 is not None:
            significant = (death - birth) >= min_persistence_h1
        else:
            significant = death >= prominence_ratio_h1 * birth
        if significant:
            loops.append((death - birth, birth, death, u, v))
    loops.sort(reverse=True) # most persistent first

    # extract the loops
    curves = []
    covered_components = set()

    for (_, birth, death, u, v) in loops:
        try:
            order = extract_loop(D, (u, v), birth_scale=D[u, v])
        except RuntimeError:
            # No representative cycle below the birth scale
            continue
        comp_id = int(components[u])
        covered_components.add(comp_id)
        curves.append({
            "type": "loop", "component": comp_id,
            "birth": birth, "death": death, "persistence": death - birth,
            "order": order,
        })

    # components without a detected loop: trace as an open path (MST diameter)
    for c in np.unique(components):
        c = int(c)
        if c in covered_components:
            continue
        node_idx = np.where(components == c)[0]
        if len(node_idx) < 2:
            continue
        order = order_along_path(D, node_idx)
        curves.append({
            "type": "path", "component": c, "order": order,
        })

    return {
        "components": components,
        "curves": curves,
        "diagram": diagram,
        "diagram_h0": [pair for pair in diagram if pair[0] == 0],
    }

