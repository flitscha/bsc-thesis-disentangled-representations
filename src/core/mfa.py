"""
Pipeline step §5.3: fit a Mixture of Factor Analyzers and read it as a local
geometry learner.

The MFA is fitted as a density model, but span(W_k) of its factor loading
matrices approximates the local tangent space at component k.
'extract_tangent_frame' turns each W_k into an orthonormal frame, so the pairs
(mean_k, frame_k) form an atlas of local charts for the later pipeline steps.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from vamm import Gaussian


def fit_mfa(data, C, H, cov_type="mfa", shared=False, seed=None):
    """
    Fit a (mixture of) Gaussian model to (N, D) data.

    C is the number of components, H the local manifold dimension (cov_type
    "mfa" only), cov_type one of "isotropic"/"diagonal"/"mfa"/"full".
    Returns the fitted model and its final training objective.
    """
    _, D = np.asarray(data).shape
    model = Gaussian(C=C, D=D, covariance_type=cov_type, shared=shared, H=H)
    obj, _ = model.fit(data, verbose=False, rng=seed)
    return model, obj


def mfa_log_likelihood(model, X):
    """
    Exact per-sample log-density of the mixture p(x) = sum_k pi_k N(x|mu_k, Sigma_k).

    Computed directly rather than taken from 'model.fit', whose return value is
    a truncated variational objective on the training set only. X must live in
    the space the model was fitted in.
    """
    return logsumexp(_component_log_probs(model, X), axis=1)


def _component_log_probs(model, X):
    """
    Per-component log joint log[pi_k * N(x | mu_k, Sigma_k)], shape (N, C).

    Row-wise logsumexp gives the mixture density, row-wise argmax the responsible
    component. For cov_type "mfa" the Woodbury identity keeps this at O(N C D H)
    instead of O(N C D^3) for the dense (D, D) covariance.
    """
    X = np.asarray(X)
    means = np.asarray(model.means)
    log_pi = np.log(np.asarray(model.prior) + 1e-300)
    C = means.shape[0]
    log_comp = np.empty((X.shape[0], C))

    if model.covariance_type == "mfa":
        # Low-rank + diagonal covariance Sigma_k = W_k W_k^T + Psi_k.
        W = np.asarray(model.A)          # (C, D, H)
        psi = np.asarray(model.variance)  # (C, D) diagonal noise variances
        D, Hdim = W.shape[1], W.shape[2]
        log2pi = np.log(2.0 * np.pi)
        for k in range(C):
            Wk, psi_k = W[k], psi[k]      # (D, H), (D,)
            pinv = 1.0 / psi_k
            diff = X - means[k]           # (N, D)
            # M = I_H + W^T Psi^-1 W  (H, H), then Cholesky
            M = np.eye(Hdim) + (Wk.T * pinv) @ Wk
            Lm = np.linalg.cholesky(M)
            diff_pinv = diff * pinv       # (N, D)
            proj = np.linalg.solve(Lm, (diff_pinv @ Wk).T).T  # (N, H)
            # (x-mu)^T Sigma^-1 (x-mu) via Woodbury
            quad = np.sum(diff_pinv * diff, axis=1) - np.sum(proj * proj, axis=1)
            log_det = np.sum(np.log(psi_k)) + 2.0 * np.sum(np.log(np.diag(Lm)))
            log_comp[:, k] = log_pi[k] - 0.5 * (D * log2pi + log_det + quad)
    else:
        covariances = np.asarray(model.covariances)
        for k in range(C):
            log_comp[:, k] = log_pi[k] + multivariate_normal.logpdf(
                X, mean=means[k], cov=covariances[k], allow_singular=True
            )
    return log_comp


def average_nll(model, X, per_dim=True):
    """
    Average negative log-likelihood on X, by default divided by D (§5.2, NLL_norm).

    Dividing by D removes the trivial growth of the log-density with dimension.
    Scores at different D still describe different reduced representations, so
    they mainly reflect fit quality and training stability. Use held-out data.
    """
    log_p = mfa_log_likelihood(model, X)
    nll = -log_p.mean()
    if per_dim:
        nll /= X.shape[1]
    return float(nll)


def mfa_reconstruct(model, X):
    """
    Reconstruct each point as the posterior mean of its most responsible component:

        x_hat = mu_k + W_k (I + W_k^T Psi_k^-1 W_k)^-1 W_k^T Psi_k^-1 (x - mu_k).

    This is the best the fitted model can represent x with, independent of any
    recovered ordering, and serves as the reconstruction floor in the evaluation.
    Operates in the space the model was fitted in; lift back separately.
    """
    if model.covariance_type != "mfa":
        raise NotImplementedError(
            "mfa_reconstruct is only defined for covariance_type='mfa'."
        )
    X = np.asarray(X, dtype=float)
    means = np.asarray(model.means)      # (C, D)
    W = np.asarray(model.A)              # (C, D, H)
    psi = np.asarray(model.variance)     # (C, D)
    H = W.shape[2]

    # responsible component per point (argmax posterior = argmax joint log-prob)
    resp = np.argmax(_component_log_probs(model, X), axis=1)  # (N,)

    X_hat = np.empty_like(X)
    for k in range(means.shape[0]):
        idx = np.nonzero(resp == k)[0]
        if idx.size == 0:
            continue
        Wk, mu_k = W[k], means[k]
        pinv = 1.0 / psi[k]
        M = np.eye(H) + (Wk.T * pinv) @ Wk               # (H, H)
        diff = X[idx] - mu_k                              # (n_k, D)
        latent = np.linalg.solve(M, ((diff * pinv) @ Wk).T).T  # (n_k, H) posterior mean
        X_hat[idx] = mu_k + latent @ Wk.T
    return X_hat


def extract_tangent_frame(loadings: np.ndarray, n_tangents: int = 1, noise: np.ndarray = None):
    """
    Orthonormal tangent frames from the loading matrices via thin QR, W_k = T_k R_k.

    Parameters
    ----------
    loadings : (N, D, H) factor loading matrices W_k ('model.A').
    n_tangents : directions to keep per component, at most H.
    noise : (N, D) diagonal MFA noise variances Psi_k ('model.variance'), optional.

    Returns
    -------
    tangents : (N, D, n_tangents) frames, ordered by descending captured variance.
    variances : (N, n_tangents) variance along each direction; a tiny value marks
        an unreliable direction.
    noise_var : (N,) mean noise variance per component, 0 if 'noise' is None.
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
        T, R = np.linalg.qr(W)  # T (D, H) has orthonormal columns

        # variance along direction j is ||W^T t_j||^2 = squared norm of row j of R
        row_var = np.sum(R ** 2, axis=1)

        order = np.argsort(row_var)[::-1][:n_tangents]
        tangents[i] = T[:, order]
        variances[i] = row_var[order]

    if noise is not None:
        noise_var = np.asarray(noise).mean(axis=1)

    return tangents, variances, noise_var


def atlas_summary(means, tangents, variances, noise_var):
    """Print a sanity-check summary of the fitted tangent frames."""
    N, D = means.shape
    n_t = tangents.shape[2]
    snr = variances.mean(axis=1) / (noise_var + 1e-12)

    print(f"Atlas: {N} components, ambient dim D={D}, n_tangents={n_t}")
    print(f"  Mean tangent variance : {variances.mean():.4f}")
    print(f"  Mean noise variance   : {noise_var.mean():.4f}")
    print(f"  Mean signal/noise     : {snr.mean():.2f}")
    print(f"  Min SNR (worst chart) : {snr.min():.2f}  (component {snr.argmin()})")
    print(f"  Max SNR (best chart)  : {snr.max():.2f}  (component {snr.argmax()})")

