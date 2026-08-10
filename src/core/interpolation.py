"""
Pipeline step §5.7: interpolate the detected 1D structures.

'interpolate_curves' turns index orders plus the component means into a smooth
curve that interpolates the means exactly while softly aligning its tangents
with the MFA chart tangents.

The curve is a cubic Hermite spline: on each segment it is the unique cubic with
prescribed endpoint positions and endpoint derivatives.

Open paths use natural (free-end) boundary tangents; loops are closed with a
periodic wrap segment so position and tangent match across the seam.
"""

import numpy as np
from scipy.interpolate import CubicHermiteSpline


def interpolate_curves(curves, points, weights=None, tangents=None, tangent_weight=3.0):
    """
    Attach a smooth curve to each detected structure.

    Parameters
    ----------
    curves : list of dict
        Each has at least "type" ("loop" or "path") and "order" (indices).
    points : (N, D) array
        The point set the orders index into.
    weights : (N,) array or None
        Per-point pi_k (MFA mixing weights). Drives how strongly each chart
        tangent is honored (small pi_k -> weak tangent).
    tangents : (N, D) or (N, D, L) array or None
        Local chart tangent frames. For (N, D, L) the primary (top-variance)
        direction is used. None disables tangent alignment (pure minimal-
        curvature interpolation).
    tangent_weight : float
        Overall strength of the tangent alignment relative to curvature.

    Returns
    -------
    curves : list of dict
        Copies of the input curves with added "points" and "spline" entries.
        The means are interpolated exactly regardless of the weights.
    """
    points = np.asarray(points)
    W = None if weights is None else np.asarray(weights)
    T = None if tangents is None else np.asarray(tangents)

    out = []
    for curve in curves:
        order = curve["order"]
        pts = points[order]
        spline = build_spline(
            pts, closed=(curve["type"] == "loop"),
            weights=None if W is None else W[order],
            tangents=None if T is None else T[order],
            tangent_weight=tangent_weight,
        )
        out.append({**curve, "points": pts, "spline": spline})
    return out


def build_spline(points_ordered, closed, weights=None, tangents=None, tangent_weight=3.0):
    """
    Fit a cubic Hermite spline that interpolates the ordered points exactly and
    softly aligns its knot tangents with the chart tangents.

    Returns a callable f(t), t in [0, 1]: f(scalar) -> (D,), f(array) -> (M, D).
    For loops the parameter is periodic (f(1) == f(0)).
    """
    pts = np.asarray(points_ordered, dtype=float)
    n, D = pts.shape
    if n == 1:
        return _constant_spline(pts[0])

    tang, wt = _prep_tangents(tangents, weights, n)

    # Distance between consecutive means. A loop also gets the closing seam
    # (last -> first) so it is treated like any other segment.
    periodic = bool(closed and n >= 3)
    diffs = np.diff(pts, axis=0)
    if periodic:
        diffs = np.vstack([diffs, pts[0] - pts[-1]])
    steps = np.linalg.norm(diffs, axis=1)

    # floor near-zero segments so the bending energy's 1/h terms stay finite.
    med = np.median(steps[steps > 0]) if np.any(steps > 0) else 1.0
    steps = np.maximum(steps, 1e-6 * med)
    total = steps.sum()

    # Chord-length parametrization
    if periodic:
        u = np.concatenate([[0.0], np.cumsum(steps[:-1])]) / total
        segments = [(i, (i + 1) % n, steps[i] / total) for i in range(n)]
    else:
        u = np.concatenate([[0.0], np.cumsum(steps)]) / total
        segments = [(i, i + 1, steps[i] / total) for i in range(n - 1)]

    m = _solve_tangents(pts, segments, tang, wt, tangent_weight)
    return _hermite_spline(u, pts, m, periodic)


def _solve_tangents(pts, segments, tang, wt, tangent_weight):
    """
    Choose the derivative (tangent) m_i at each knot.

    We pick the tangents that make the curve as smooth as possible (least bending energy)
    while nudging each one toward its chart tangent (strongly where pi_k is large, barely
    where it is small).
    This is a single quadratic in the tangents, so it comes down to one linear system.

    Returns m (n, D): the tangent vector at every knot.
    """
    n, D = pts.shape

    # Write the bending energy (integral ||s''||^2 du) as a quadratic in the tangents m
    K = np.zeros((n, n))
    g = np.zeros((n, D))
    for a, b, h in segments:
        K[a, a] += 4.0 / h
        K[b, b] += 4.0 / h
        K[a, b] += 2.0 / h
        K[b, a] += 2.0 / h
        g[a] += (pts[a] - pts[b]) * (12.0 / h ** 2)
        g[b] += (pts[a] - pts[b]) * (12.0 / h ** 2)

    # Setting the gradient of E to zero gives the linear system A vec(m) = rhs.
    A = np.kron(K, np.eye(D))
    rhs = -0.5 * g.reshape(-1)

    # for each knot add a penalty on the part of m_i that is perpendicular to the
    # chart tangent (projector I - t t^T).
    if tang is not None and tangent_weight > 0:
        kscale = np.median(np.diag(K))
        for i in range(n):
            lam = tangent_weight * kscale * wt[i]
            proj = np.eye(D) - np.outer(tang[i], tang[i])
            A[i * D:(i + 1) * D, i * D:(i + 1) * D] += lam * proj

    A += 1e-9 * (np.trace(A) / (n * D)) * np.eye(n * D) # numerical safeguard
    return np.linalg.solve(A, rhs).reshape(n, D)


def _hermite_spline(u, pts, m, periodic):
    """
    Build the callable curve f(t) from the knots: positions 'pts' at parameters
    'u' with tangents 'm'.

    For a loop we repeat the first knot at u = 1 (same position and tangent)
    """
    if periodic:
        x = np.concatenate([u, [1.0]])
        y = np.vstack([pts, pts[0]])
        dydx = np.vstack([m, m[0]])
    else:
        x, y, dydx = u, pts, m
    herm = CubicHermiteSpline(x, y, dydx, axis=0)

    def spline(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        t_arr = (t_arr % 1.0) if periodic else np.clip(t_arr, 0.0, 1.0)
        coords = np.atleast_2d(herm(t_arr))
        return coords[0] if np.ndim(t) == 0 else coords

    return spline



# small helpers

def _norm_mean1(x):
    """Normalize non-negative weights to mean 1 (fall back to uniform)."""
    mean = np.mean(x)
    return x / mean if mean > 0 else np.ones_like(x)


def _prep_tangents(tangents, weights, n):
    """
    Return (unit primary tangents, per-knot tangent weights) or (None, None).

    The weight is pi_k normalized to mean 1 (so 'tangent_weight' stays scale-
    free); uniform weights if none are given.
    """
    if tangents is None:
        return None, None
    tang = np.asarray(tangents, dtype=float)
    tang = tang[:, :, 0] if tang.ndim == 3 else tang # primary direction
    tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
    wt = np.ones(n) if weights is None else _norm_mean1(np.asarray(weights, float))
    return tang, wt


def _constant_spline(point):
    """Curve collapsed to a single point (degenerate one-component structure)."""
    def spline(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.repeat(point[None, :], t_arr.size, axis=0)
        return out[0] if np.ndim(t) == 0 else out
    return spline

