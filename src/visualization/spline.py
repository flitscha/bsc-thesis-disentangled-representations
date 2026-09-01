"""Draws a spline path over the data points, in 2D or 3D."""

import numpy as np

from visualization.geometry import set_axis_limits


def _draw_data_points(ax, plot_data, is_3d, colors, cmap, colorbar,
                      color_label="Rotation angle", color_scale=(0, 360)):
    coords = [plot_data[:, i] for i in range(3 if is_3d else 2)]
    if colors is None:
        kwargs = dict(c="grey", alpha=(0.15 if is_3d else 0.2), s=(4 if is_3d else 5))
        ax.scatter(*coords, **kwargs)
        return
    vmin, vmax = color_scale
    scatter = ax.scatter(*coords, c=colors, cmap=cmap, s=15, alpha=0.5,
                         vmin=vmin, vmax=vmax)
    if colorbar and not hasattr(ax, "_has_colorbar"):
        cbar = ax.figure.colorbar(scatter, ax=ax, pad=0.04, shrink=0.75)
        cbar.set_label(color_label, fontsize=15)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks(np.linspace(vmin, vmax, 9))
        ax._has_colorbar = True


def _draw_spline_curve(ax, spline, t, D, is_3d):
    t_vals = np.linspace(0, t, 200)
    curve = spline(t_vals)[:, :D]

    current_p = spline(t)
    if current_p.ndim > 1:
        current_p = current_p[0]
    current_p = current_p[:D]

    curve_coords = [curve[:, i] for i in range(D)]
    point_coords = [[current_p[i]] for i in range(D)]

    ax.plot(*curve_coords, c="blue", linewidth=2.5, zorder=10)
    ax.scatter(*point_coords, s=180, c="crimson", marker="*", edgecolors="black", zorder=20)


def visualize_spline(
    ax, data, spline, t, draw_points=True, colors=None, cmap="hsv", colorbar=False,
    color_label="Rotation angle", color_scale=(0, 360)
):
    """
    Draw the spline path up to the coordinate t, in 2D or 3D.

    `colors` holds the per-point values behind the colorbar, and `color_label` and `color_scale`
    describe them, since not every dataset colours by an angle.
    """
    is_3d = hasattr(ax, "get_zlim")
    D = 3 if is_3d else 2

    # ignore any columns beyond the ones being drawn
    plot_data = data[:, :D]
    ranges = plot_data.max(axis=0) - plot_data.min(axis=0)
    padding = 0.1 * np.where(ranges > 0, ranges, 1.0)

    if draw_points:
        _draw_data_points(ax, plot_data, is_3d, colors, cmap, colorbar,
                          color_label=color_label, color_scale=color_scale)

    if spline is not None and t > 0:
        _draw_spline_curve(ax, spline, t, D, is_3d)

    set_axis_limits(ax, plot_data, padding=padding, adjustable="box")

