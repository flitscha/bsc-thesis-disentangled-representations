"""
Pipeline step §5.6: loop and connected-component detection via topological data
analysis (persistent homology, gudhi).

Operates on a precomputed distance matrix (built in §5.4, see graph.py) and
returns the detected structure as index orders -- interpolation into splines
(§5.7) is a separate step (see interpolation.interpolate_curves).
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


def significant_bars(diagram, dim, min_persistence=None, top_k=None, scale_ref=None, auto_ratio=0.1):
    """
    Filter persistence bars of a given homology dimension by persistence
    (death - birth). Infinite bars always count as significant.

    Automatic threshold (min_persistence=None):
      - if >= 2 finite bars exist: cut at the largest gap between sorted
        persistence values (separates "real" features from noise).
      - if exactly 1 finite bar exists, there is no gap to measure. In
        that case we require its persistence to be at least `auto_ratio`
        of `scale_ref` (e.g. the point-cloud diameter) -- otherwise a
        single small noise artifact (e.g. from local density fluctuation
        on an open arc with no real loop) would always be accepted.
    """
    bars = [(b, d) for (dd, (b, d)) in diagram if dd == dim]
    finite = [(b, d) for (b, d) in bars if np.isfinite(d)]
    infinite = [(b, d) for (b, d) in bars if not np.isfinite(d)]

    if not finite:
        return infinite

    persistences = np.array(sorted([d - b for b, d in finite], reverse=True))

    if min_persistence is None:
        if len(persistences) > 1:
            gaps = persistences[:-1] - persistences[1:]
            cut = int(np.argmax(gaps))
            min_persistence = (persistences[cut] + persistences[cut + 1]) / 2
        elif scale_ref is not None:
            min_persistence = auto_ratio * scale_ref
        else:
            min_persistence = persistences[0] / 2  # last-resort fallback

    kept = [(b, d) for (b, d) in finite if (d - b) >= min_persistence]
    kept.sort(key=lambda bd: -(bd[1] - bd[0]))
    if top_k is not None:
        kept = kept[:top_k]

    return infinite + kept


def extract_components(distance_matrix, n_components):
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="single")
    labels = fcluster(Z, t=n_components, criterion="maxclust")
    return labels - 1


def extract_loop(distance_matrix, birth_edge, birth_scale):
    u, v = int(birth_edge[0]), int(birth_edge[1])

    adj = distance_matrix.copy()
    adj[adj >= birth_scale] = 0
    np.fill_diagonal(adj, 0)

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
    auto_ratio_h1=0.1,
):
    """
    Detect connected components (H0) and loops (H1) from a distance matrix
    via topological data analysis (persistent homology).

    Parameters
    ----------
    distance_matrix : (N, N) array
        Pairwise distances between the MFA components (built in §5.4).
    min_persistence_h0, min_persistence_h1 : float or None
        Persistence thresholds; None uses an automatic heuristic.
    auto_ratio_h1 : float
        Fallback ratio for the single-bar H1 case (see significant_bars).

    Returns
    -------
    result : dict with keys
        "components" : (N,) component label per node,
        "curves"     : list of {type, component, order[, birth, death,
                       persistence]} -- index orders only, no splines,
        "diagram", "diagram_h0" : the persistence diagrams.
    """
    D = distance_matrix

    if min_persistence_h0 is None:
        min_persistence_h0 = 0.05 * float(np.max(D))

    # --- H0: connected components
    _, diagram_h0, _ = compute_persistence(D, max_dimension=0)
    h0_bars = significant_bars(diagram_h0, dim=0, min_persistence=min_persistence_h0)
    components = extract_components(D, n_components=len(h0_bars))

    # --- H1: loop detection
    _, diagram, generators = compute_persistence(D, max_dimension=1)
    h1_bars = significant_bars(
        diagram, dim=1, min_persistence=min_persistence_h1,
        scale_ref=float(np.max(D)), auto_ratio=auto_ratio_h1,
    )
    gen_rows = generators[1][0] if len(generators[1]) > 0 else np.empty((0, 4), dtype=int)

    curves = []
    covered_components = set()

    for (birth, death) in h1_bars:
        row = min(gen_rows, key=lambda r: abs(D[int(r[0]), int(r[1])] - birth))
        u, v = int(row[0]), int(row[1])
        order = extract_loop(D, (u, v), birth_scale=D[u, v])
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
        "diagram_h0": diagram_h0,
    }
