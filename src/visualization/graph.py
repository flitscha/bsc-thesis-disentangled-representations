"""Visualization of a traversal order over the MFA component means."""

from visualization.geometry import set_axis_limits


def visualize_traversal(ax, means, order):
    """Draw the traversal path and number each node with its position in it."""
    D = means.shape[1]
    set_axis_limits(ax, means)

    if order is None or len(order) == 0:
        return

    ordered_means = means[order]
    coords = [ordered_means[:, i] for i in range(D)]

    ax.scatter(*coords, c="black", s=30, zorder=5)
    ax.plot(*coords, color="gray", linewidth=1.5, alpha=0.7, zorder=4)

    for idx, point in enumerate(ordered_means):
        text_pos = list(point)
        text_pos[0] += 0.04 # offset so the label does not sit on the node
        ax.text(*text_pos, str(idx), fontsize=9, color="blue", zorder=6)

