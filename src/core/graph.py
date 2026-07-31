"""
Builds a weighted graph on the MFA Gaussian components,
where edge weights reflect geodesic proximity along the manifold.
"""

import numpy as np
from core.mfa import chart_overlap, direction_alignment


# Score matrix
def compute_score_matrix(
    means: np.ndarray,
    tangents: np.ndarray,
    variances: np.ndarray = None,
    w_overlap: float = 0.4,
    w_direction: float = 0.4,
) -> np.ndarray:
    """
    Compute an (N, N) pairwise score matrix.

    Lower score = better neighbor candidate.

    Score formula:
        score(i,j) = dist(i,j) * (1 - w_overlap * overlap(i,j)
                                     - w_direction * dir_align(i,j))

    The two weights control the trade-off:
      w_overlap   : reward components whose tangent frames agree
      w_direction : reward components where the connecting vector
                    lies along the manifold (not across it)

    Parameters
    ----------
    means       : (N, D)
    tangents    : (N, D, n_tangents)
    variances   : (N, n_tangents) or None
        If given, components with high variance get a slight bonus
    w_overlap   : float in [0, 0.5]
    w_direction : float in [0, 0.5]
        Note: w_overlap + w_direction should be <= 1.0

    Returns
    -------
    score_matrix : (N, N), non-negative floats, diagonal = 0
    """
    N = means.shape[0]

    # Euclidean distance matrix (vectorized)
    diff = means[:, None, :] - means[None, :, :]  # (N, N, D)
    dist = np.linalg.norm(diff, axis=-1)  # (N, N)

    # Chart overlap and direction alignment
    overlap = np.zeros((N, N))
    dir_align = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            ov = chart_overlap(tangents[i], tangents[j])
            da = direction_alignment(means[i], means[j], tangents[i], tangents[j])
            overlap[i, j] = overlap[j, i] = ov
            dir_align[i, j] = dir_align[j, i] = da

    # variance-based confidence weighting
    if variances is not None:
        # confidence[i] = mean tangent variance, normalized to [0.5, 1.0]
        conf = variances.mean(axis=1)
        conf = 0.5 + 0.5 * (conf - conf.min()) / (conf.max() - conf.min() + 1e-12)
        # geometric mean of confidence for each pair
        conf_matrix = np.sqrt(conf[:, None] * conf[None, :])
    else:
        conf_matrix = np.ones((N, N))

    score = dist * (1.0 - w_overlap * overlap - w_direction * dir_align)
    score = score / (conf_matrix + 1e-12)

    np.fill_diagonal(score, 0.0)
    return score


def euclidean_distance_matrix(means: np.ndarray) -> np.ndarray:
    """
    Plain pairwise Euclidean distance between component means.

    An alternative to the geometry-aware score matrix. The score matrix
    penalizes disagreeing tangent frames, which can artificially inflate the
    distance between spatially close components (e.g. near-isotropic components
    in high-curvature regions, where the dominant tangent direction is poorly
    defined) and thereby fragment a connected region into spurious components.
    """
    diff = means[:, None, :] - means[None, :, :]
    return np.linalg.norm(diff, axis=-1)


# Graph construction
def build_knn_graph(score_matrix: np.ndarray, k: int = 2) -> dict:
    """
    Build an undirected k-NN graph from the score matrix.

    Parameters
    ----------
    score_matrix : (N, N)
    k            : int, number of neighbors per node

    Returns
    -------
    graph : dict with keys:
        'adjacency' : (N, N) float array (score value or 0)
        'edges'     : list of (i, j) tuples, i < j
        'degrees'   : (N,) int array
    """
    N = score_matrix.shape[0]
    adjacency = np.zeros((N, N), dtype=float)

    for i in range(N):
        # exclude self (score=0 on diagonal)
        row = score_matrix[i].copy()
        row[i] = np.inf
        neighbors = np.argsort(row)[:k]
        for j in neighbors:
            adjacency[i, j] = score_matrix[i, j]
            adjacency[j, i] = score_matrix[i, j]

    edges = [
        (i, j)
        for i in range(N)
        for j in range(i + 1, N)
        if adjacency[i, j] > 0
    ]
    degrees = (adjacency > 0).sum(axis=1).astype(int)

    return {
        'adjacency': adjacency,
        'edges': edges,
        'degrees': degrees,
    }


def graph_diagnostics(graph: dict) -> dict:
    """
    Returns a dict with:
      n_components  : number of connected components (want: 1)
      is_cycle_like : True if all degrees are 2 (perfect 1D topology)
      degree_hist   : degree histogram as dict
      isolated      : list of node indices with degree 0
    """
    adjacency = graph['adjacency']
    degrees   = graph['degrees']
    N         = adjacency.shape[0]
 
    # BFS to count connected components
    visited    = np.zeros(N, dtype=bool)
    n_comp     = 0
    for start in range(N):
        if not visited[start]:
            n_comp += 1
            queue = [start]
            while queue:
                node = queue.pop()
                if visited[node]:
                    continue
                visited[node] = True
                neighbors = np.where(adjacency[node] > 0)[0]
                queue.extend(neighbors.tolist())
 
    unique, counts = np.unique(degrees, return_counts=True)
    degree_hist = dict(zip(unique.tolist(), counts.tolist()))
 
    return {
        'n_components' : n_comp,
        'is_cycle_like': bool(np.all(degrees == 2)),
        'degree_hist'  : degree_hist,
        'isolated'     : np.where(degrees == 0)[0].tolist(),
    }

