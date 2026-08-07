"""
Pipeline step §5.4: build a geodesic distance matrix over the MFA components.

The distance approximates travel along the data manifold in two steps. First, a
tangent-aware local metric assigns each pair of nearby components an edge length
that penalizes moving off the estimated tangent spaces. Second, the geodesic
distance between any two components is the shortest path in their k-nearest-
neighbor graph, so paths follow the manifold instead of cutting across it.
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
    Symmetric (N, N) matrix of local edge lengths from the tangent-stretch metric.

    At component k with tangent projector P_k, a step d = mu_j - mu_k costs
        l_k(d)^2 = ||d||^2 + lambda_aniso * ||d - P_k d||^2,
    so the part of d that leaves the tangent space is penalized. The edge length
    averages this over both endpoints.

    Parameters
    ----------
    lambda_aniso : off-manifold penalty; 0 recovers Euclidean, larger stretches
                   normal directions more.
    """
    means = np.asarray(means)
    tangents = np.asarray(tangents)
    N = means.shape[0]

    diff = means[:, None, :] - means[None, :, :]   # (N, N, D), diff[k, j] = mu_k - mu_j
    sq_dist = np.sum(diff ** 2, axis=-1)           # ||d||^2

    # on-manifold part ||P_k d||^2 = ||T_k^T d||^2, one source component k per row
    on_manifold = np.empty((N, N))
    for k in range(N):
        coords = diff[k] @ tangents[k]             # (N, H)
        on_manifold[k] = np.sum(coords ** 2, axis=1)

    off_manifold = sq_dist - on_manifold
    length = np.sqrt(np.maximum(sq_dist + lambda_aniso * off_manifold, 0.0))

    return 0.5 * (length + length.T)               # symmetrize over both endpoints


def _neighbor_graph(length: np.ndarray, n_neighbors: int) -> csr_matrix:
    """
    Sparse k-nearest-neighbor graph from a pairwise length matrix. An edge is
    kept if either endpoint is among the other's `n_neighbors` nearest.
    """
    N = length.shape[0]
    n_neighbors = min(n_neighbors, N - 1)

    adjacency = np.zeros((N, N))
    for i in range(N):
        row = length[i].copy()
        row[i] = np.inf
        nn = np.argsort(row)[:n_neighbors]
        adjacency[i, nn] = length[i, nn]

    return csr_matrix(np.maximum(adjacency, adjacency.T))  # union -> symmetric


def riemannian_distance_matrix(
    means: np.ndarray,
    tangents: np.ndarray,
    lambda_aniso: float = 30.0,
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    Geodesic distance matrix over the MFA components (thesis §5.4).

    Local edge lengths come from the tangent-stretch metric, and distances are
    shortest paths in the resulting k-nearest-neighbor graph. The result is a
    proper metric (the triangle inequality holds by construction) and does not
    short-circuit loops, which the downstream topological analysis relies on.

    Returns
    -------
    distances : (N, N), non-negative, symmetric, zero diagonal.
    """
    length = local_metric_matrix(means, tangents, lambda_aniso)
    graph = _neighbor_graph(length, n_neighbors)
    distances = shortest_path(graph, method="D", directed=False)

    # A disconnected graph leaves some pairs unreachable (inf). Keep them finite
    # but far beyond any real distance so they stay separate in the analysis.
    if not np.isfinite(distances).all():
        finite_max = distances[np.isfinite(distances)].max()
        distances[~np.isfinite(distances)] = 2.0 * finite_max

    np.fill_diagonal(distances, 0.0)
    return distances


def euclidean_distance_matrix(means: np.ndarray) -> np.ndarray:
    """
    Plain pairwise Euclidean distance between component means. A baseline for
    the geodesic distance above, ignoring the tangent geometry entirely.
    """
    diff = means[:, None, :] - means[None, :, :]
    return np.linalg.norm(diff, axis=-1)


# Graph construction
def build_knn_graph(distance_matrix: np.ndarray, k: int = 2) -> dict:
    """
    Build an undirected k-NN graph from a distance matrix.

    Returns
    -------
    graph : dict with keys:
        'adjacency' : (N, N) float array (distance value or 0)
        'edges'     : list of (i, j) tuples, i < j
        'degrees'   : (N,) int array
    """
    N = distance_matrix.shape[0]
    adjacency = np.zeros((N, N), dtype=float)

    for i in range(N):
        # exclude self (distance = 0 on diagonal)
        row = distance_matrix[i].copy()
        row[i] = np.inf
        neighbors = np.argsort(row)[:k]
        for j in neighbors:
            adjacency[i, j] = distance_matrix[i, j]
            adjacency[j, i] = distance_matrix[i, j]

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

