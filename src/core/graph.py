"""
Build a geodesic distance matrix over the MFA components.

The distance approximates travel along the data manifold, in two steps. First a tangent-aware
local metric gives every pair of nearby components an edge length that penalizes moving off the
estimated tangent spaces. Then the distance between any two components is the shortest path in
their k-nearest-neighbor graph, so a path follows the manifold instead of cutting across it.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def local_metric_matrix(
    means: np.ndarray,
    tangents: np.ndarray,
    lambda_aniso: float,
) -> np.ndarray:
    """
    Symmetric (N, N) matrix of local edge lengths under the tangent-stretch metric.

    At component k with tangent projector P_k, a step d = mu_j - mu_k costs

        l_k(d)^2 = ||d||^2 + lambda_aniso * ||d - P_k d||^2,

    so only the part of d that leaves the tangent space is penalized. Since the two endpoints
    disagree on the cost, the edge length is the average of both.

    'lambda_aniso' controls the penalty: 0 gives back the Euclidean distance, larger values stretch
    the normal directions further.
    """
    means = np.asarray(means)
    tangents = np.asarray(tangents)
    N = means.shape[0]

    diff = means[:, None, :] - means[None, :, :] # (N, N, D), with diff[k, j] = mu_k - mu_j
    sq_dist = np.sum(diff ** 2, axis=-1) # ||d||^2

    # the on-manifold part ||P_k d||^2 = ||T_k^T d||^2, one source component k per row
    on_manifold = np.empty((N, N))
    for k in range(N):
        coords = diff[k] @ tangents[k] # (N, H)
        on_manifold[k] = np.sum(coords ** 2, axis=1)

    off_manifold = sq_dist - on_manifold
    length = np.sqrt(np.maximum(sq_dist + lambda_aniso * off_manifold, 0.0))

    return 0.5 * (length + length.T) # average over both endpoints


# A chart whose mixing weight falls below this fraction of the uniform weight 1/N holds almost no
# data, so it is dropped before the graph is built.
_MIN_WEIGHT_FRACTION = 1e-2

def prune_low_weight_components(weights: np.ndarray) -> np.ndarray:
    """
    Ascending indices of the components to keep, dropping the near-empty charts.

    Everything is kept if the threshold would leave fewer than two components.
    """
    weights = np.asarray(weights, dtype=float)
    N = weights.shape[0]
    threshold = _MIN_WEIGHT_FRACTION / N
    keep = np.flatnonzero(weights > threshold)
    if keep.size < 2:
        return np.arange(N)
    return keep


def _neighbor_graph(length: np.ndarray, n_neighbors: int) -> csr_matrix:
    """
    Sparse k-nearest-neighbor graph from a pairwise length matrix.

    An edge survives if either of its endpoints is among the other's n_neighbors nearest.
    """
    N = length.shape[0]
    n_neighbors = min(n_neighbors, N - 1)

    adjacency = np.zeros((N, N))
    for i in range(N):
        row = length[i].copy()
        row[i] = np.inf
        nn = np.argsort(row)[:n_neighbors]
        adjacency[i, nn] = length[i, nn]

    return csr_matrix(np.maximum(adjacency, adjacency.T))


def riemannian_distance_matrix(
    means: np.ndarray,
    tangents: np.ndarray,
    lambda_aniso: float = 30.0,
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    Geodesic (N, N) distance matrix over the MFA components.

    The entries are shortest paths in the k-nearest-neighbor graph of the tangent-stretch lengths.
    """
    length = local_metric_matrix(means, tangents, lambda_aniso)
    graph = _neighbor_graph(length, n_neighbors)
    distances = shortest_path(graph, method="D", directed=False)

    # If the graph is disconnected, some pairs have no path at all. Keep their distance finite but
    # far beyond any real one, so they stay separate in the analysis.
    if not np.isfinite(distances).all():
        finite_max = distances[np.isfinite(distances)].max()
        distances[~np.isfinite(distances)] = 2.0 * finite_max

    np.fill_diagonal(distances, 0.0)
    return distances

