"""Visualization of MFA components (means and covariances) as a graph, as well as traversal paths."""

from visualization.geometry import set_axis_limits, get_component_plotter


def visualize_graph_on_mfa(
    ax, means, covariances, priors, edges=None, draw_nodes=True,
    visualisation_mode="line", node_color="red", edge_color="blue", edge_alpha=0.6
):
    """
    Visualize MFA components (means + covariances) and optionally a graph connecting them.

    Parameters
    ----------
    ax : matplotlib axis
    means : ndarray (N, D)
    covariances : list of ndarray (D, D)
    priors : ndarray (N,)
    edges : list of tuples (i, j), optional
        Each tuple connects mean i with mean j.
    """
    N, D = means.shape
    set_axis_limits(ax, means)

    plot_component = get_component_plotter(visualisation_mode, D)
    for mean, cov, prior in zip(means, covariances, priors):
        plot_component(ax, mean, cov, prior)

    if draw_nodes:
        node_coords = [means[:, i] for i in range(D)]
        ax.scatter(*node_coords, s=20, c=node_color, zorder=3)

    if edges is not None:
        for i, j in edges:
            edge_coords = [[means[i, k], means[j, k]] for k in range(D)]
            ax.plot(*edge_coords, color=edge_color, alpha=edge_alpha, linewidth=2)


def visualize_traversal(ax, means, order):
    """Visualizes a traversal order using indices at each node."""
    N, D = means.shape
    set_axis_limits(ax, means)

    if order is None or len(order) == 0:
        return

    ordered_means = means[order]
    coords = [ordered_means[:, i] for i in range(D)]

    ax.scatter(*coords, c="black", s=30, zorder=5)
    ax.plot(*coords, color="gray", linewidth=1.5, alpha=0.7, zorder=4)

    for idx, point in enumerate(ordered_means):
        text_pos = list(point)
        text_pos[0] += 0.04 # small offset to avoid text directly over the node
        ax.text(*text_pos, str(idx), fontsize=9, color="blue", zorder=6)

