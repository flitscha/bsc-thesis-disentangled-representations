"""
Drawing of single Gaussian components as an ellipse, ellipsoid, line or plane.
"""

import numpy as np
from matplotlib.patches import Ellipse


# --- plot space ---
def plot_transform(pre, projection, ambient_dim):
    """
    The affine map (M, b) from the space the fitted model lives in to the plotted one.

    A model point can then be drawn next to the observations as model_point @ M + b. The model is
    fitted on the preprocessed data, so with PCA switched on its means and covariances live in the
    reduced space while the data and the embedding map are ambient. Two steps have to be undone:

        reduced -> ambient, via pre.inverse_transform, then ambient -> plotted, via the projection

    'pre' is the fitted PCARotation or None, 'projection' is the (D, d) map back to the intrinsic
    coordinates of an embedded dataset, and 'ambient_dim' is the observation dimension. A
    covariance maps as M.T @ cov @ M.
    """
    if pre is None:
        M = np.eye(ambient_dim) if projection is None else np.asarray(projection)
        return M, np.zeros(M.shape[1])

    M = pre.components_.T # reduced -> ambient
    b = pre.mean_
    if projection is not None:
        M, b = M @ projection, b @ projection
    return M, b


# --- axis limits ---
def set_axis_limits(ax, points, padding=0.1, adjustable=None):
    """
    Set the axis limits and an equal aspect ratio from the (N, D) points, for D of 2 or 3.

    'padding' is either a scalar or one value per dimension. 'adjustable' goes straight through to
    ax.set_aspect().
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
    """A line along the major axis of a Gaussian component, its largest eigenvector."""
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
    """The 3D ellipsoid of a Gaussian component. `prior` does not affect the drawing yet."""
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
    """The plane spanned by the two principal axes of a Gaussian component."""
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


# --- dispatch ---
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

