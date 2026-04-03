import numpy as np
from scipy.interpolate import CubicSpline


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
