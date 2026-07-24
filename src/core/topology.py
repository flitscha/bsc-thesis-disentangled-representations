import numpy as np
import gudhi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from core.atlas import extract_tangent_frame
from core.graph import compute_score_matrix
from core.interpolation import build_closed_spline, build_open_spline


def compute_distance_matrix(model, n_tangents=1, w_overlap=0.4, w_direction=0.4):
    """
    Manifold-aware distance matrix (means + covariances), used for H1 loop
    routing where penalizing cross-manifold shortcuts is desirable.
    """
    tangents, variances, _ = extract_tangent_frame(model.covariances, n_tangents)
    return compute_score_matrix(model.means, tangents, variances, w_overlap, w_direction)


def compute_euclidean_distance_matrix(model):
    """
    Plain pairwise Euclidean distance between MFA means. Used for H0
    (connected-component) detection.

    NOTE: the geometry-aware score matrix above is intentionally NOT used
    here. It penalizes disagreeing tangent frames (chart_overlap /
    direction_alignment), which can artificially inflate the "distance"
    between spatially close components -- e.g. near-isotropic components
    in high-curvature regions, where the dominant tangent direction is
    poorly defined. That fragments a single, topologically connected
    region into several spurious components.
    """
    means = model.means
    diff = means[:, None, :] - means[None, :, :]
    return np.linalg.norm(diff, axis=-1)


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


def extract_path(distance_matrix, node_indices):
    """
    Order the points of a loop-free (path-shaped) component along its
    intrinsic 1D structure via the diameter of its minimum spanning tree
    (standard two-sweep algorithm: farthest node from an arbitrary start,
    then farthest node from that -- the path between those two endpoints
    is the tree's diameter and gives a natural traversal order).

    Parameters
    ----------
    distance_matrix : (N, N) full distance matrix
    node_indices     : indices (into distance_matrix) belonging to this component

    Returns
    -------
    order : list of global vertex indices, ordered along the path
    """
    sub_D = distance_matrix[np.ix_(node_indices, node_indices)]
    mst = minimum_spanning_tree(csr_matrix(sub_D))
    mst = mst.maximum(mst.T)  # symmetrize

    dist0, _ = dijkstra(mst, directed=False, indices=0, return_predecessors=True)
    far0 = int(np.argmax(dist0))

    dist1, predecessors = dijkstra(mst, directed=False, indices=far0, return_predecessors=True)
    far1 = int(np.argmax(dist1))

    path_local = [far1]
    cur = far1
    while cur != far0:
        cur = predecessors[cur]
        path_local.append(cur)
    path_local.reverse()

    return [int(node_indices[i]) for i in path_local]


def analyze_topology(
    model, n_tangents=1, min_persistence_h0=None, min_persistence_h1=None,
    w_overlap=0.4, w_direction=0.4, auto_ratio_h1=0.1,
):

    # --- H0: component count, on plain Euclidean distance (see note above)
    D_euclid = compute_euclidean_distance_matrix(model)
    D_score = compute_distance_matrix(model, n_tangents, w_overlap, w_direction)

    if min_persistence_h0 is None:
        min_persistence_h0 = 0.05 * float(np.max(D_score))

    _, diagram_h0, _ = compute_persistence(D_score, max_dimension=0)
    h0_bars = significant_bars(diagram_h0, dim=0, min_persistence=min_persistence_h0)
    components = extract_components(D_score, n_components=len(h0_bars))

    # --- H1: loop detection, on manifold-aware score matrix
    _, diagram, generators = compute_persistence(D_score, max_dimension=1)
    h1_bars = significant_bars(
        diagram, dim=1, min_persistence=min_persistence_h1,
        scale_ref=float(np.max(D_score)), auto_ratio=auto_ratio_h1,
    )
    gen_rows = generators[1][0] if len(generators[1]) > 0 else np.empty((0, 4), dtype=int)

    curves = []
    covered_components = set()

    for (birth, death) in h1_bars:
        row = min(gen_rows, key=lambda r: abs(D_score[int(r[0]), int(r[1])] - birth))
        u, v = int(row[0]), int(row[1])
        order = extract_loop(D_score, (u, v), birth_scale=D_score[u, v])
        points = model.means[order]
        comp_id = int(components[u])
        covered_components.add(comp_id)
        curves.append({
            "type": "loop", "component": comp_id,
            "birth": birth, "death": death, "persistence": death - birth,
            "order": order, "spline": build_closed_spline(points),
        })

    # components without a detected loop: trace as an open path (MST diameter)
    for c in np.unique(components):
        c = int(c)
        if c in covered_components:
            continue
        node_idx = np.where(components == c)[0]
        if len(node_idx) < 2:
            continue
        order = extract_path(D_score, node_idx)
        points = model.means[order]
        curves.append({
            "type": "path", "component": c,
            "order": order, "spline": build_open_spline(points),
        })

    return {
        "distance_matrix": D_score,
        "component_distance_matrix": D_euclid,
        "diagram": diagram,
        "diagram_h0": diagram_h0,
        "components": components,
        "curves": curves,
        "loops": [c for c in curves if c["type"] == "loop"],  # backward-compat
    }
