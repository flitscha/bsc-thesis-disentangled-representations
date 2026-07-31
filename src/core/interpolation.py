"""
Pipeline step §3.7: interpolate the detected 1D structures with cubic splines.

`interpolate_curves` turns index orders (from §3.5/§3.6) plus the component
means into smooth spline parametrizations: closed splines for loops, open
splines for paths.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def interpolate_curves(curves, points):
    """
    Attach a cubic spline to each detected curve.

    Parameters
    ----------
    curves : list of dict
        Each has at least "type" ("loop" or "path") and "order" (indices).
    points : (N, D) array
        The point set the orders index into (e.g. model.means).

    Returns
    -------
    curves : list of dict
        Copies of the input curves with an added "spline" (and "points") entry.
    """
    out = []
    for curve in curves:
        pts = points[curve["order"]]
        if curve["type"] == "loop":
            spline = build_closed_spline(pts)
        else:
            spline = build_open_spline(pts)
        out.append({**curve, "points": pts, "spline": spline})
    return out


def build_closed_spline(points_ordered):
    """
    Inputs:
    - points_ordered : (N,D) ndarray

    Returns:
    - sline : CubicSpline object, parametrized from 0 to 1
    """

    # first and last point must be identical for periodic spline
    if not np.allclose(points_ordered[0], points_ordered[-1]):
        points_ordered = np.vstack([points_ordered, points_ordered[0]])

    N = len(points_ordered)
    t_vals = np.linspace(0, 1, N)
    spline = CubicSpline(t_vals, points_ordered, axis=0, bc_type='periodic')
    return spline


def build_open_spline(points_ordered):
    """
    Inputs:
    - points_ordered : (N,D) ndarray, an open path (not a closed loop)
    Returns:
    - spline : CubicSpline object, parametrized from 0 to 1, natural boundary conditions
    """
    N = len(points_ordered)
    t_vals = np.linspace(0, 1, N)
    spline = CubicSpline(t_vals, points_ordered, axis=0, bc_type='natural')
    return spline
