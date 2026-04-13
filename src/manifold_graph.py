import numpy as np


# extract (1-dimensional) tangent directions from covariances
def extract_tangent_directions(covariances):
    """
    Extract principal direction (largest eigenvector)
    from each covariance matrix.

    Parameters:
    - covariances : (N, D, D)

    Returns:
    - tangents : (N, D)
    """
    N = len(covariances)
    D = covariances[0].shape[0]

    tangents = np.zeros((N, D))

    for i, cov in enumerate(covariances):
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argmax(vals)
        t = vecs[:, idx]

        # normalize
        t = t / (np.linalg.norm(t) + 1e-8)

        tangents[i] = t

    return tangents


def compute_score_matrix(means, tangents):
    """
    Compute pairwise score:
    score = distance * (1 - alignment)

    alignment = |dot(t_i, t_j)|

    Inputs:
    - means : (N, D)
    - tangents : (N, D)

    Returns:
    - score_matrix : (N, N)
    """
    N = means.shape[0]

    score_matrix = np.zeros((N, N))
    eps = 1e-8

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            d_ij = means[j] - means[i]
            dist = np.linalg.norm(d_ij) + eps

            # TODO: play with parameters (maybe conclude priors?)

            # Tangent alignment
            alignment = np.abs(np.dot(tangents[i], tangents[j]))

            # Direction alignment
            dir_i = np.abs(np.dot(d_ij, tangents[i])) / dist
            dir_j = np.abs(np.dot(d_ij, tangents[j])) / dist
            dir_align = 0.5 * (dir_i + dir_j)

            # Combined score
            score = dist * (1.0 - 0.3 * alignment - 0.3 * dir_align)

            score_matrix[i, j] = score

    return score_matrix


def build_knn_graph(score_matrix, k=2):
    """
    Build undirected k-NN graph based on score.

    Inputs:
    - score_matrix : (N, N)

    Returns:
    - adjacency : (N, N) matrix that uses score-values for vertices, and 0 for non-vertices.
    """
    N = score_matrix.shape[0]
    adjacency = np.zeros((N, N), dtype=float)

    for i in range(N):
        neighbors = np.argsort(score_matrix[i])[1:k+1]

        for j in neighbors:
            if i == j:
                continue
            adjacency[i, j] = score_matrix[i][j]
            adjacency[j, i] = score_matrix[i][j]

    return adjacency


def adjacency_to_edges(adjacency):
    edges = []
    N = adjacency.shape[0]

    for i in range(N):
        for j in range(i + 1, N):  # wichtig: doppelte vermeiden!
            if adjacency[i, j] != 0:
                edges.append((i, j))

    return edges


def compute_degrees(adjacency):
    return adjacency.sum(axis=1)


def is_cycle_like(adjacency):
    degrees = compute_degrees(adjacency)
    return np.all((degrees >= 1) & (degrees <= 3))
