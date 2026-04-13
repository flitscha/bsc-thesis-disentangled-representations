"""
Extracts local geometric structure from MFA covariance matrices.

A "chart" in the atlas sense: each Gaussian component defines a local
linear approximation (tangent frame) to the data manifold at that point.
"""

import numpy as np


def extract_tangent_frame(covariances: np.ndarray, n_tangents: int = 1):
    """
    Extract the n_tangents principal directions from each covariance matrix.
    These span the local tangent space of the manifold at each component.

    Parameters
    ----------
    covariances : array of shape (N, D, D)
    n_tangents  : int
        How many tangent directions to keep per component.

    Returns
    -------
    tangents : ndarray, shape (N, D, n_tangents)
        Orthonormal tangent vectors. tangents[i, :, k] is the k-th
        tangent direction at component i, sorted by descending variance.
    variances : ndarray, shape (N, n_tangents)
        Corresponding eigenvalues (variance along each tangent direction).
        Useful for weighting: a direction with tiny variance is unreliable.
    noise_var : ndarray, shape (N,)
        Mean variance in the normal directions (ambient - tangent space).
        Approximates the isotropic noise level of the MFA model.
    """
    covariances = np.asarray(covariances)
    N, D, _ = covariances.shape

    tangents = np.zeros((N, D, n_tangents))
    variances = np.zeros((N, n_tangents))
    noise_var = np.zeros(N)

    for i, cov in enumerate(covariances):
        # eigh: assumes symmetric, returns eigenvalues ascending
        vals, vecs = np.linalg.eigh(cov)

        # largest n_tangents eigenvalues/vectors
        idx = np.argsort(vals)[::-1]  # descending
        top_idx = idx[:n_tangents]
        rest_idx = idx[n_tangents:]

        tangents[i] = vecs[:, top_idx]  # (D, n_tangents)
        variances[i] = vals[top_idx]

        # noise = mean variance in normal directions
        if len(rest_idx) > 0:
            noise_var[i] = vals[rest_idx].mean()

    return tangents, variances, noise_var


def chart_overlap(tangents_i: np.ndarray, tangents_j: np.ndarray) -> float:
    """
    Measure how well two local tangent frames agree (chart compatibility).

    For 1D (tangents are vectors): this is |cos θ|, i.e. |dot product|.
    For nD (tangents are frames):  this is the sum of squared singular values
    of the cross-Gram matrix, normalized to [0, 1].

    A value of 1.0 means the two frames span exactly the same subspace.
    A value of 0.0 means the tangent spaces are orthogonal (very different).

    Parameters
    ----------
    tangents_i : (D, n_tangents)
    tangents_j : (D, n_tangents)

    Returns
    -------
    overlap : float in [0, 1]
    """
    # Cross-Gram matrix: how much does frame i project onto frame j?
    G = tangents_i.T @ tangents_j  # (n_tangents, n_tangents)
    # Sum of squared singular values = squared Frobenius norm of projection
    overlap = np.linalg.norm(G, 'fro') ** 2
    n = tangents_i.shape[1]
    return float(overlap / n)  # normalized: max = 1


def direction_alignment(mean_i, mean_j, tangents_i, tangents_j):
    """
    How well does the connecting vector (mean_i -> mean_j) align
    with the tangent frames of both components?

    Used in graph_builder to prefer neighbors that lie along the manifold,
    not across it.

    Parameters
    ----------
    mean_i, mean_j : (D,)
    tangents_i, tangents_j : (D, n_tangents)

    Returns
    -------
    align : float in [0, 1]
        1 = connecting vector lies entirely in both tangent spaces
        0 = connecting vector is perpendicular to both tangent spaces
    """
    d = mean_j - mean_i
    norm = np.linalg.norm(d)
    if norm < 1e-10:
        return 1.0
    d = d / norm

    # project d onto each tangent frame, measure how much is captured
    proj_i = np.linalg.norm(tangents_i.T @ d)  # in [0, 1] since tangents orthonormal
    proj_j = np.linalg.norm(tangents_j.T @ d)

    return float(0.5 * (proj_i + proj_j))


def atlas_summary(means, tangents, variances, noise_var):
    """
    Print a readable summary of the fitted atlas.
    Useful for quick sanity checks after training.

    Parameters
    ----------
    means     : (N, D)
    tangents  : (N, D, n_tangents)
    variances : (N, n_tangents)
    noise_var : (N,)
    """
    N, D = means.shape
    n_t = tangents.shape[2]
    snr = variances.mean(axis=1) / (noise_var + 1e-12)

    print(f"Atlas: {N} components, ambient dim D={D}, n_tangents={n_t}")
    print(f"  Mean tangent variance : {variances.mean():.4f}")
    print(f"  Mean noise variance   : {noise_var.mean():.4f}")
    print(f"  Mean signal/noise     : {snr.mean():.2f}")
    print(f"  Min SNR (worst chart) : {snr.min():.2f}  (component {snr.argmin()})")
    print(f"  Max SNR (best chart)  : {snr.max():.2f}  (component {snr.argmax()})")

