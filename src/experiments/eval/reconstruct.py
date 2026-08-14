"""
Reconstruction-error decomposition (M3).

Three reconstructions of the INPUT are compared against a TARGET, as
per-observation RMSE, in order of decreasing model freedom:

    pca_floor : PCA + rotation round-trip, i.e. the reduction loss alone.
    mfa_floor : MFA posterior-mean reconstruction.
    full      : nearest point of the learned curve, lifted back to ambient space;
                depends on ordering, spline and model together.

The caller picks the mode via the input/target pair: capacity uses clean images
for both, denoising a noisy input against the clean target.
"""

import numpy as np

from core.mfa import mfa_reconstruct


def _rmse(A, B):
    """Per-observation RMSE over the feature axis."""
    return np.sqrt(np.mean((np.asarray(A) - np.asarray(B)) ** 2, axis=1))


def reconstruction_errors(pipe, X_input, X_target=None):
    """
    Reconstruction-error decomposition of X_input against X_target.

    'pipe' must be fitted and detected. 'X_target' is row-aligned with X_input;
    None makes the input its own target (capacity mode).

    Returns (N,) arrays "full", "mfa_floor", "pca_floor" and "input_error", the
    latter being the noise level in denoising mode and 0 in capacity mode.
    """
    X_input = np.asarray(X_input, dtype=float)
    X_target = X_input if X_target is None else np.asarray(X_target, dtype=float)

    Z = pipe.pre.transform(X_input) if pipe.pre is not None else X_input
    t, cid = pipe.transform(X_input)

    # curve point nearest to each input, lifted back to ambient space
    curve_pts = np.stack([pipe.curves_[cid[i]]["spline"](float(t[i])) for i in range(len(t))])
    full = _rmse(X_target, pipe.reconstruct(curve_pts))

    # model representational floor (posterior-mean projection), lifted to ambient
    mfa = _rmse(X_target, pipe.reconstruct(mfa_reconstruct(pipe.model, Z)))

    # dimensionality-reduction floor: PCA + rotation round-trip of the input
    pca = _rmse(X_target, pipe.reconstruct(Z))

    return {
        "full": full,
        "mfa_floor": mfa,
        "pca_floor": pca,
        "input_error": _rmse(X_target, X_input),
    }

