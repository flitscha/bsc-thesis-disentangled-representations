"""
Detect loops and connected components with persistent homology (gudhi).

Works on the precomputed distance matrix from graph.py and returns the detected structure as
index orders, without any interpolation yet.
"""

import numpy as np
import gudhi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from core.ordering import order_along_path


def compute_persistence(distance_matrix, max_dimension=1, max_edge_length=None):
    """
    Persistent homology of the Vietoris-Rips filtration of a distance matrix.

    The complex is expanded one dimension beyond 'max_dimension', because features of dimension h
    are only correct once the (h+1)-simplices that fill them in are there. Without an explicit
    'max_edge_length' the filtration runs slightly past the largest distance, so that every pair of
    components is eventually joined.

    Returns the simplex tree, the diagram as (dim, (birth, death)) pairs, and the flag persistence
    generators, which hold the birth and death edges.
    """
    if max_edge_length is None:
        max_edge_length = float(np.max(distance_matrix)) * 1.01

    rips = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension + 1)
    diagram = simplex_tree.persistence()
    generators = simplex_tree.flag_persistence_generators()
    return simplex_tree, diagram, generators


def significant_bars(diagram, dim, min_persistence):
    """
    Bars of dimension 'dim' whose persistence (death - birth) reaches 'min_persistence'.

    Infinite bars always count as significant.
    """
    bars = [(b, d) for (dd, (b, d)) in diagram if dd == dim]
    infinite = [(b, d) for (b, d) in bars if not np.isfinite(d)]
    kept = [(b, d) for (b, d) in bars if np.isfinite(d) and (d - b) >= min_persistence]
    kept.sort(key=lambda bd: -(bd[1] - bd[0]))
    return infinite + kept


def _h0_num_components(diagram, gap_factor):
    """
    Number of connected components, read off the H0 barcode at its largest gap.

    The H0 deaths are the scales at which components merge. Within one structure they cluster
    around the typical spacing of neighbouring components, so a merge that joins two separate
    structures leaves a visible gap. We cut at the largest gap and count every death above it. The
    cut only fires if that gap is at least 'gap_factor' times the median death.
    """
    deaths = np.array(sorted(d for (dim, (_, d)) in diagram if dim == 0 and np.isfinite(d)))
    if deaths.size < 2:
        return 1
    gaps = np.diff(deaths)
    i = int(np.argmax(gaps))
    if gaps[i] >= gap_factor * np.median(deaths):
        return 1 + (deaths.size - (i + 1))
    return 1


def median_merge_scale(diagram):
    """
    Median finite H0 death, the typical scale at which two components merge.

    Persistence thresholds are lengths in the units of the distance matrix, so a bare number means
    nothing across datasets. Expressed as a multiple of this scale they stay comparable. Returns
    None if the barcode has no finite death at all.
    """
    deaths = [d for (dim, (_, d)) in diagram if dim == 0 and np.isfinite(d)]
    return float(np.median(deaths)) if deaths else None


def persistence_thresholds(diagram, h0_factor=0.0, h1_factor=0.0):
    """
    Turn the scale-free H0/H1 threshold factors into absolute persistence thresholds.

    Each factor is a multiple of the median H0 merge scale of 'diagram'. A factor of 0 or less
    keeps the automatic rule of 'detect_tda' and is left out of the result, which can therefore go
    straight into 'ManifoldPipeline.set_params'.
    """
    scale = median_merge_scale(diagram)
    if not scale or scale <= 0:
        return {}

    thresholds = {}
    if h0_factor and h0_factor > 0:
        thresholds["min_persistence_h0"] = float(h0_factor) * scale
    if h1_factor and h1_factor > 0:
        thresholds["min_persistence_h1"] = float(h1_factor) * scale
    return thresholds


def extract_components(distance_matrix, n_components):
    """
    Assign each node a connected-component label, from 0 to n_components-1.

    Single-linkage clustering is exactly the H0 filtration of the Rips complex, since its merge
    scales are the H0 deaths. So we cut its dendrogram into the number of clusters that persistent
    homology already told us to expect.
    """
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="single")
    labels = fcluster(Z, t=n_components, criterion="maxclust")
    return labels - 1


def extract_loop(distance_matrix, birth_edge, birth_scale):
    """
    Reconstruct a representative cycle for an H1 loop, as an ordered node list.

    Persistent homology only reports a loop as a birth/death pair. To get a concrete ordering of
    components around the hole, remove the edge (u, v) that closes the loop at 'birth_scale' and
    take the shortest path from u to v instead. Returns the indices around the cycle, u ... v, u.
    """
    u, v = int(birth_edge[0]), int(birth_edge[1])

    adj = distance_matrix.copy()
    adj[adj > birth_scale] = 0
    np.fill_diagonal(adj, 0)
    adj[u, v] = adj[v, u] = 0

    dist, predecessors = dijkstra(csr_matrix(adj), directed=False, indices=u, return_predecessors=True)
    if not np.isfinite(dist[v]):
        raise RuntimeError(f"cannot reconstruct loop: {u} and {v} are unconnected below birth.")

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
    component_gap_factor_h0=0.5, prominence_ratio_h1=1.8,
):
    """
    Detect connected components (H0) and loops (H1) from an (N, N) distance matrix.

    Parameters
    ----------
    min_persistence_h0, min_persistence_h1 : fixed persistence thresholds. None, the default, uses
        the scale-free criteria below instead.
    component_gap_factor_h0 : split at the largest gap in the H0 barcode, but only if that gap is
        at least this multiple of the median merge scale.
    prominence_ratio_h1 : keep a loop only if death >= ratio * birth.

    Returns
    -------
    A dict with "components" ((N,) labels), "curves", "diagram" and its H0 part "diagram_h0". Each
    curve carries its "type" ("loop" or "path"), the "component" it lives in and the "order" of
    node indices along it. Loops additionally carry birth, death and persistence.
    """
    D = distance_matrix
    _, diagram, generators = compute_persistence(D, max_dimension=1)

    # H0: connected components
    if min_persistence_h0 is not None:
        n_components = len(significant_bars(diagram, dim=0, min_persistence=min_persistence_h0))
    else:
        n_components = _h0_num_components(diagram, gap_factor=component_gap_factor_h0)
    components = extract_components(D, n_components=n_components)

    # H1: one generator row per class, holding its birth edge (row[0], row[1]) and its death edge
    # (row[2], row[3])
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
    loops.sort(reverse=True) # the most persistent loop first

    curves = []
    covered_components = set()

    for (_, birth, death, u, v) in loops:
        try:
            order = extract_loop(D, (u, v), birth_scale=D[u, v])
        except RuntimeError:
            # no representative cycle exists below the birth scale
            continue
        comp_id = int(components[u])
        covered_components.add(comp_id)
        curves.append({
            "type": "loop", "component": comp_id,
            "birth": birth, "death": death, "persistence": death - birth,
            "order": order,
        })

    # a component without a loop is traced as an open path along the MST diameter
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

