"""
Geometric building blocks:
Drawing of individual Gaussian components (ellipse/ellipsoid/line/plane)
"""

import numpy as np
from matplotlib.patches import Ellipse


# --------------------------- Axis Limits ---------------------------
def set_axis_limits(ax, points, padding=0.1, adjustable=None):
    """
    Set axis limits and an equal aspect ratio from the (N, D) points, D in {2, 3}.

    'padding' is a scalar or one value per dimension; 'adjustable' is passed
    through to ax.set_aspect().
    """
    D = points.shape[1]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    padding = np.broadcast_to(padding, (D,))

    ax.set_xlim(mins[0] - padding[0], maxs[0] + padding[0])
    ax.set_ylim(mins[1] - padding[1], maxs[1] + padding[1])

    if D == 3:
        ax.set_zlim(mins[2] - padding[2], maxs[2] + padding[2])
        ax.set_box_aspect([1, 1, 1])
    else:
        if adjustable is not None:
            ax.set_aspect("equal", adjustable=adjustable)
        else:
            ax.set_aspect("equal")


# --------------------------- Component-Plots ---------------------------
def plot_ellipse(ax, mean, cov, prior=None, color="orange"):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    vals = np.clip(vals, 1e-8, None) # avoid negative eigenvalues
    width, height = 4 * np.sqrt(vals)

    alpha = min(0.6, 0.2 + prior * 3) if prior is not None else 0.4

    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  color=color, alpha=alpha)
    ax.add_patch(ell)


def plot_line(ax, mean, cov, prior=None, length=None, color="orange", linewidth=3.0, alpha=None):
    """Line along the major axis (largest eigenvector) of a Gaussian component."""
    D = mean.shape[0]
    vals, vecs = np.linalg.eigh(cov)
    direction = vecs[:, np.argmax(vals)]

    if length is None:
        length = 0.5 if D == 2 else 3.0
    direction = direction * (length / 2.0)

    p1 = mean - direction
    p2 = mean + direction

    if alpha is None:
        alpha = min(0.8, 0.3 + prior * 2) if prior is not None else 0.8

    coords = [[p1[i], p2[i]] for i in range(D)]
    ax.plot(*coords, color=color, linewidth=linewidth, alpha=alpha)


def plot_ellipsoid(ax, mean, cov, prior=None, resolution=20, color="orange", alpha=0.2):
    """3D ellipsoid of a Gaussian component (`prior` is currently not used for visibility)"""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-10, None)
    radii = 2.0 * np.sqrt(vals)

    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([x, y, z], axis=0).reshape(3, -1)

    ellipsoid = radii[:, np.newaxis] * sphere
    ellipsoid = vecs @ ellipsoid
    ellipsoid = ellipsoid + mean[:, np.newaxis]

    x_e = ellipsoid[0].reshape(resolution, resolution)
    y_e = ellipsoid[1].reshape(resolution, resolution)
    z_e = ellipsoid[2].reshape(resolution, resolution)

    ax.plot_surface(x_e, y_e, z_e, color=color, alpha=alpha, linewidth=0)


def plot_plane(ax, mean, cov, prior=None, size=3.0, color="orange", alpha=0.2):
    """3D plane spanned by the two principal axes of a Gaussian component."""
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vec1 = vecs[:, idx[0]] * size
    vec2 = vecs[:, idx[1]] * size

    p1 = mean - vec1 - vec2
    p2 = mean - vec1 + vec2
    p3 = mean + vec1 - vec2
    p4 = mean + vec1 + vec2

    X = np.array([[p1[0], p2[0]], [p3[0], p4[0]]])
    Y = np.array([[p1[1], p2[1]], [p3[1], p4[1]]])
    Z = np.array([[p1[2], p2[2]], [p3[2], p4[2]]])

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)


def _none_plotter(ax, mean, cov, prior=None):
    return None


# --------------------------- Dispatch ---------------------------
_COMPONENT_PLOTTERS = {
    2: {
        "ellipsoid": plot_ellipse,
        "line": plot_line,
        "plane": plot_line, # in 2D-Plots: "plane" is handled the same as "line"
        "none": _none_plotter,
    },
    3: {
        "ellipsoid": plot_ellipsoid,
        "line": plot_line,
        "plane": plot_plane,
        "none": _none_plotter,
    },
}


def get_component_plotter(mode, dim):
    try:
        return _COMPONENT_PLOTTERS[dim][mode]
    except KeyError:
        raise ValueError(f"Unknown visualisation_mode={mode!r} for dim={dim}")

