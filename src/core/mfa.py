"""
Pipeline step 2 (thesis §3.3): fit a Mixture of Factor Analyzers (MFA) and
interpret it as a local geometry learner.

The MFA is fitted as a density model, but its factor loading matrices W_k are
reinterpreted geometrically: span(W_k) approximates the local tangent space of
the data manifold at component k. `extract_tangent_frame` turns each W_k into an
orthonormal tangent frame via thin QR. Together, the pairs (mean_k, frame_k)
form a discrete sample of the tangent bundle of the manifold (an "atlas" of
local charts), which the downstream steps use to build a distance matrix.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from vamm import Gaussian


def fit_mfa(data, C, H, cov_type="mfa", shared=False, seed=None):
    """
    Fit a (mixture of) Gaussian model to the data.

    Parameters
    ----------
    data : (N, D) array
    C : int
        Number of mixture components.
    H : int
        Latent / local manifold dimension (only used for cov_type "mfa").
    cov_type : {"isotropic", "diagonal", "mfa", "full"}
    shared : bool
        Whether the covariance is shared across components.
    seed : int or None
        RNG seed for the fit.

    Returns
    -------
    model : vamm.Gaussian
        The fitted model.
    obj : float
        Final training objective.
    """
    _, D = np.asarray(data).shape
    model = Gaussian(C=C, D=D, covariance_type=cov_type, shared=shared, H=H)
    obj, _ = model.fit(data, verbose=False, rng=seed)
    return model, obj


def mfa_log_likelihood(model, X):
    """
    Exact per-sample log-density log p_Theta(x_n) of a fitted (mixture of)
    Gaussian model, evaluated on (held-out) data.

    The model defines the Gaussian-mixture density
        p(x) = sum_k pi_k * N(x | mu_k, Sigma_k),
    with mixing weights pi_k (`model.prior`), means mu_k (`model.means`) and
    covariances Sigma_k = W_k W_k^T + Psi_k (`model.covariances`). This is
    computed directly instead of using the value returned by `model.fit`, which
    is only a truncated variational objective on the *training* set.

    Parameters
    ----------
    model : vamm.Gaussian
        A fitted model (any covariance_type).
    X : (N, D) array
        Points to score, in the same space the model was fitted in.

    Returns
    -------
    log_p : (N,) ndarray
        Log-density of each point under the mixture.
    """
    X = np.asarray(X)
    means = np.asarray(model.means)
    covariances = np.asarray(model.covariances)
    log_pi = np.log(np.asarray(model.prior) + 1e-300)

    C = means.shape[0]
    log_comp = np.empty((X.shape[0], C))
    for k in range(C):
        log_comp[:, k] = log_pi[k] + multivariate_normal.logpdf(
            X, mean=means[k], cov=covariances[k], allow_singular=True
        )
    return logsumexp(log_comp, axis=1)


def average_nll(model, X, per_dim=True):
    """
    Average negative log-likelihood of the model on `X`, optionally normalized
    by the ambient dimension (thesis §3.2, NLL_norm).

    NLL_norm = -1 / (N * D) * sum_n log p_Theta(x_n)

    Dividing by D removes the trivial growth of the log-density with the number
    of dimensions, making the score comparable across different PCA dimensions.
    (As noted in the thesis, scores at different D still describe different
    reduced representations, so they mainly reflect training stability + fit.)

    Parameters
    ----------
    model : vamm.Gaussian
        A fitted model.
    X : (N, D) array
        Evaluation data (use a held-out validation set).
    per_dim : bool
        If True, also divide by the ambient dimension D.

    Returns
    -------
    nll : float
    """
    log_p = mfa_log_likelihood(model, X)
    nll = -log_p.mean()
    if per_dim:
        nll /= X.shape[1]
    return float(nll)


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

    Used in graph construction to prefer neighbors that lie along the manifold,
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
    Print a readable summary of the fitted tangent frames.
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
