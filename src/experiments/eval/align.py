"""
Post-hoc alignment of a learned 1D coordinate to a ground-truth factor.

The model is unsupervised and recovers the factor only up to a few freedoms,
which are the only ones fitted out here:

    loop: direction s in {+1, -1} and offset delta.
    arc:  direction and scale, since t is always in [0, 1] while the true factor
          may span 0..180 or 0..90.

Non-linear speed distortion is deliberately NOT fitted out: a non-uniform
traversal stays visible as a residual.
"""

import numpy as np


def align_loop(t, theta_deg):
    """
    Align a loop coordinate t in [0, 1) to a periodic ground-truth angle (deg).

    Returns the recovered direction "s", the offset "delta_deg", the (N,) unsigned
    residual "error_deg" and the two conversions "theta_of_t" / "t_of_theta".
    """
    t = np.asarray(t, dtype=float)
    theta = np.asarray(theta_deg, dtype=float)
    phi = 360.0 * t

    best = None
    for s in (1.0, -1.0):
        d = np.deg2rad(theta - s * phi)
        delta = np.rad2deg(np.arctan2(np.sin(d).mean(), np.cos(d).mean()))
        err = np.abs((theta - (s * phi + delta) + 180.0) % 360.0 - 180.0)
        if best is None or err.mean() < best["error_deg"].mean():
            best = {"s": s, "delta_deg": float(delta), "error_deg": err}

    s, delta = best["s"], best["delta_deg"]
    best["theta_of_t"] = lambda tt, s=s, delta=delta: (s * 360.0 * np.asarray(tt, float) + delta) % 360.0
    best["t_of_theta"] = lambda th, s=s, delta=delta: (s * (np.asarray(th, float) - delta) % 360.0) / 360.0
    return best


def align_arc(t, factor):
    """
    Align an arc coordinate t in [0, 1] to an interval-valued ground-truth factor
    by the best affine map factor ~ a * t + b.

    Returns the coefficients "a"/"b", the (N,) unsigned residual "error" in the
    factor's units and the two conversions "factor_of_t" / "t_of_factor".
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(factor, dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    err = np.abs(y - (a * t + b))
    return {
        "a": float(a), "b": float(b), "error": err,
        "factor_of_t": lambda tt, a=a, b=b: a * np.asarray(tt, float) + b,
        "t_of_factor": lambda yy, a=a, b=b: (np.asarray(yy, float) - b) / a,
    }

