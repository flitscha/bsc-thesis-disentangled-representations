"""
Post-hoc alignment of a learned 1D coordinate to a ground-truth factor.

The model is unsupervised and can recover the factor only up to a small set of freedoms:

- loop: direction s in {+1, -1} and offset delta.

- arc: direction and scale. The model does not know, if an arc apans 0..180
  or 0..90. The variable t is always in [0, 1]

In both cases we DO NOT fit out non-linear speed distortion: a non-uniform
traversal shows up as residual.
"""

import numpy as np


def align_loop(t, theta_deg):
    """
    Align a loop coordinate t in [0, 1) to a periodic ground-truth angle (deg).

    Returns
    -------
    dict with
        s          : +1 or -1 (recovered direction)
        delta_deg  : offset in degrees
        error_deg  : (N,) unsigned angular residual in degrees
        theta_of_t : callable t -> predicted angle (deg, in [0, 360))
        t_of_theta : callable angle(deg) -> t in [0, 1)
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
    Align an arc coordinate t in [0, 1] to an interval-valued ground-truth
    factor by the best affine map factor ~ a * t + b.

    Returns
    -------
    dict with
        a, b        : affine coefficients
        error       : (N,) unsigned residual in the factor's units
        factor_of_t : callable t -> predicted factor
        t_of_factor : callable factor -> t
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

