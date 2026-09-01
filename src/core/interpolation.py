"""
Interpolate the detected 1D structures with cubic splines.

'interpolate_curves' turns index orders plus the component means into a cubic Hermite spline. The
spline passes through the means exactly, while its knot tangents are only softly aligned with the
MFA chart tangents. Loops get an extra wrap segment, so position and tangent match across the seam.
"""

import numpy as np
from scipy.interpolate import CubicHermiteSpline


def interpolate_curves(curves, points, weights=None, tangents=None, tangent_weight=3.0):
    """
    Attach a smooth curve to each detected structure.

    Parameters
    ----------
    curves : dicts with at least "type" ("loop" or "path") and "order" (indices).
    points : (N, D) point set the orders index into.
    weights : (N,) mixing weights. The smaller the weight, the less that chart's tangent counts.
    tangents : (N, D) or (N, D, L) chart frames, of which the top-variance direction is used.
        None turns the tangent alignment off.
    tangent_weight : how strongly the tangents are enforced, relative to the curvature.

    Returns copies of the curves with "points" and "spline" added. The means are hit exactly no
    matter what the weights are.
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
    Fit a cubic Hermite spline through the ordered points, aligned softly with the chart tangents.

    Returns a callable f(t) for t in [0, 1], where f(scalar) gives a (D,) point and f(array) an
    (M, D) block. On a loop the parameter is periodic, so f(1) equals f(0).
    """
    pts = np.asarray(points_ordered, dtype=float)
    n = pts.shape[0]
    if n == 1:
        return _constant_spline(pts[0])

    tang, wt = _prep_tangents(tangents, weights, n)

    # Distances between consecutive means. A loop also gets the closing seam from last to first,
    # so that it is treated like any other segment.
    periodic = bool(closed and n >= 3)
    diffs = np.diff(pts, axis=0)
    if periodic:
        diffs = np.vstack([diffs, pts[0] - pts[-1]])
    steps = np.linalg.norm(diffs, axis=1)

    # put a floor under near-zero segments, so the 1/h terms of the bending energy stay finite
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
    Choose the derivative m_i at each knot. Returns m of shape (n, D).

    The derivatives minimize the bending energy while being nudged towards their chart tangent,
    strongly where the chart carries a lot of data and barely where it carries little. Both terms
    are quadratic in m, so the whole thing is a single linear system.
    """
    n, D = pts.shape

    # write the bending energy, the integral of ||s''||^2, as a quadratic form in the tangents m
    K = np.zeros((n, n))
    g = np.zeros((n, D))
    for a, b, h in segments:
        K[a, a] += 4.0 / h
        K[b, b] += 4.0 / h
        K[a, b] += 2.0 / h
        K[b, a] += 2.0 / h
        g[a] += (pts[a] - pts[b]) * (12.0 / h ** 2)
        g[b] += (pts[a] - pts[b]) * (12.0 / h ** 2)

    # setting the gradient to zero gives the linear system A vec(m) = rhs
    A = np.kron(K, np.eye(D))
    rhs = -0.5 * g.reshape(-1)

    # At each knot, penalize the part of m_i perpendicular to the chart tangent, using the
    # projector I - t t^T. This constrains the direction but leaves the speed free.
    if tang is not None and tangent_weight > 0:
        kscale = np.median(np.diag(K))
        for i in range(n):
            lam = tangent_weight * kscale * wt[i]
            proj = np.eye(D) - np.outer(tang[i], tang[i])
            A[i * D:(i + 1) * D, i * D:(i + 1) * D] += lam * proj

    A += 1e-9 * (np.trace(A) / (n * D)) * np.eye(n * D) # guard against a singular system
    return np.linalg.solve(A, rhs).reshape(n, D)


def _hermite_spline(u, pts, m, periodic):
    """
    Build the callable curve f(t) from the knots: positions 'pts' at parameters 'u', tangents 'm'.

    A loop repeats its first knot at u = 1, with the same position and the same tangent.
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


def _norm_mean1(x):
    mean = np.mean(x)
    return x / mean if mean > 0 else np.ones_like(x)


def _prep_tangents(tangents, weights, n):
    """
    Return the unit primary tangents and the per-knot weights, or (None, None).

    The weights are the mixing weights normalized to mean 1, which keeps 'tangent_weight' free of
    the data scale. Without given weights they are uniform.
    """
    if tangents is None:
        return None, None
    tang = np.asarray(tangents, dtype=float)
    tang = tang[:, :, 0] if tang.ndim == 3 else tang # the primary direction of the frame
    tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
    wt = np.ones(n) if weights is None else _norm_mean1(np.asarray(weights, float))
    return tang, wt


def _constant_spline(point):
    """Curve collapsed to a single point, for a structure with only one component."""
    def spline(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.repeat(point[None, :], t_arr.size, axis=0)
        return out[0] if np.ndim(t) == 0 else out
    return spline

