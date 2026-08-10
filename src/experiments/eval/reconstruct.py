"""
Reconstruction-error decomposition.

For each observation, three reconstructions of the INPUT are compared against a
TARGET image, as per-observation pixel RMSE (root mean square error):

- full      : the learned curve's point nearest to the input, reconstructed to
              ambient space. Depends on the whole pipeline (ordering + spline + model).

- mfa_floor : the MFA posterior-mean reconstruction of the input.

- pca_floor : the PCA + rotation round-trip. This measures how much information
              is lost through PCA.


Two modes, chosen by the caller via the input/target pair:
- capacity  : input = target = clean images.
- denoising : input = noisy images, target = clean images. Measures how well
              the method recovers the clean signal from a noisy input.
"""

import numpy as np

from core.mfa import mfa_reconstruct


def _rmse(A, B):
    """Per-observation RMSE over the feature axis."""
    return np.sqrt(np.mean((np.asarray(A) - np.asarray(B)) ** 2, axis=1))


def reconstruction_errors(pipe, X_input, X_target=None):
    """
    Reconstruction-error decomposition of X_input against X_target.

    Parameters
    ----------
    pipe : ManifoldPipeline
        Fitted and detected (curves with splines available).
    X_input : (N, D0)
        Images fed through the representation (encode -> reconstruct).
    X_target : (N, D0) or None
        Ground-truth images to compare against, row-aligned with X_input. If
        None, the input is used as its own target (standard self-reconstruction,
        capacity mode).

    Returns
    -------
    dict of (N,) arrays: "full", "mfa_floor", "pca_floor", "input_error"
    (input_error = input-vs-target RMSE: the noise level in denoising mode, 0 in
    capacity mode).
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

