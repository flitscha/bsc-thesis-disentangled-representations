"""
Extracts local geometric structure from MFA covariance matrices.

A "chart" in the atlas sense: each Gaussian component defines a local
linear approximation (tangent frame) to the data manifold at that point.
"""

import numpy as np


def extract_tangent_frame(loadings: np.ndarray, n_tangents: int = 1, noise: np.ndarray = None):
    """
    Extract an orthonormal basis of the local tangent space from each MFA
    factor loading matrix W_k via the thin QR decomposition W_k = T_k R_k.

    The column space of W_k is interpreted as the local tangent space of the
    data manifold at component k. QR yields an orthonormal basis T_k of exactly
    that subspace (span(T_k) = span(W_k)). For n_tangents = 1 this reduces to
    normalizing the single tangent vector.

    Parameters
    ----------
    loadings : array of shape (N, D, H)
        Factor loading matrices W_k, e.g. 'model.A' of a vamm MFA model.
    n_tangents : int
        Local manifold dimension to keep per component (n_tangents <= H).
    noise : array of shape (N, D) or None
        Optional diagonal MFA noise variances Psi_k (e.g. 'model.variance').
        If given, noise_var[i] is their mean; otherwise noise_var is all zeros.

    Returns
    -------
    tangents : ndarray, shape (N, D, n_tangents)
        Orthonormal tangent frames. tangents[i, :, k] is the k-th tangent
        direction at component i, ordered by descending captured variance.
    variances : ndarray, shape (N, n_tangents)
        Variance captured along each tangent direction, computed as the squared
        row norms of R_k. Their sum equals ||W_k||_F^2, i.e. the total variance
        the loading matrix contributes to the tangent space. Useful for
        weighting: a direction with tiny variance is unreliable.
    noise_var : ndarray, shape (N,)
        Mean diagonal noise variance per component (0 if 'noise' is None).
        Approximates the off-manifold noise level of the MFA model.
    """
    loadings = np.asarray(loadings)
    N, D, H = loadings.shape

    if n_tangents > H:
        raise ValueError(
            f"n_tangents={n_tangents} exceeds the loading rank H={H}."
        )

    tangents = np.zeros((N, D, n_tangents))
    variances = np.zeros((N, n_tangents))
    noise_var = np.zeros(N)

    for i, W in enumerate(loadings):
        # thin QR: W (D, H) = T (D, H) @ R (H, H), T has orthonormal columns
        T, R = np.linalg.qr(W)

        # variance captured along tangent direction j equals ||W^T t_j||^2,
        # which (since W = T R) is the squared norm of the j-th row of R.
        row_var = np.sum(R ** 2, axis=1)  # (H,)

        # keep the n_tangents directions capturing the most variance
        order = np.argsort(row_var)[::-1][:n_tangents]
        tangents[i] = T[:, order]
        variances[i] = row_var[order]

    if noise is not None:
        noise_var = np.asarray(noise).mean(axis=1)

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

