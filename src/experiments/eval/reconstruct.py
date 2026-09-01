"""
Reconstruction-error decomposition, the metric M3.

Three reconstructions of the input are compared against a target, as a per-observation RMSE, in
order of decreasing freedom:

    pca_floor : the PCA and rotation round trip, so the reduction loss on its own.
    mfa_floor : the MFA posterior-mean reconstruction.
    full      : the nearest point of the learned curve, lifted back to the ambient space. This one
                depends on the ordering, the spline and the model together.

The caller picks the mode through the input/target pair: capacity uses clean images for both,
denoising a noisy input against the clean target.
"""

import numpy as np

from core.mfa import mfa_reconstruct


def _rmse(A, B):
    """RMSE per observation, averaged over the features."""
    return np.sqrt(np.mean((np.asarray(A) - np.asarray(B)) ** 2, axis=1))


def reconstruction_errors(pipe, X_input, X_target=None):
    """
    Reconstruction-error decomposition of X_input against X_target.

    'pipe' has to be fitted and detected. 'X_target' is row-aligned with X_input, and None makes
    the input its own target, which is the capacity mode.

    Returns the (N,) arrays "full", "mfa_floor", "pca_floor" and "input_error". The last one is
    the noise level in denoising mode and 0 in capacity mode.
    """
    X_input = np.asarray(X_input, dtype=float)
    X_target = X_input if X_target is None else np.asarray(X_target, dtype=float)

    Z = pipe.pre.transform(X_input) if pipe.pre is not None else X_input
    t, cid = pipe.transform(X_input)

    # the curve point nearest to each input, lifted back to the ambient space
    curve_pts = np.stack([pipe.curves_[cid[i]]["spline"](float(t[i])) for i in range(len(t))])
    full = _rmse(X_target, pipe.reconstruct(curve_pts))

    # what the model itself can represent, its posterior-mean projection
    mfa = _rmse(X_target, pipe.reconstruct(mfa_reconstruct(pipe.model, Z)))

    # what the dimensionality reduction alone costs, a plain round trip
    pca = _rmse(X_target, pipe.reconstruct(Z))

    return {
        "full": full,
        "mfa_floor": mfa,
        "pca_floor": pca,
        "input_error": _rmse(X_target, X_input),
    }

