"""Visualization of Gaussian mixture models (data + components) in 2D/3D."""

from visualization.geometry import set_axis_limits, get_component_plotter


def _visualize_gmm_nd(
    ax, data, means, covariances, priors, draw_points=True,
    visualisation_mode="ellipsoid", draw_means=True
):
    """
    Unified implementation for 2D and 3D. The dimension is derived from data.shape[1]
    """
    D = data.shape[1]
    is_3d = D == 3

    if draw_points:
        if is_3d:
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5, alpha=0.8, label="Data")
        else:
            ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.8, label="Data")

    if draw_means:
        if is_3d:
            ax.scatter(means[:, 0], means[:, 1], means[:, 2], s=10, c="red", label="Means")
        else:
            ax.scatter(means[:, 0], means[:, 1], s=20, c="red", label="Means")

    set_axis_limits(ax, data)

    plot_component = get_component_plotter(visualisation_mode, D)
    for mean, cov, prior in zip(means, covariances, priors):
        plot_component(ax, mean, cov, prior)


def _visualize_gmm_higher_dimension(
    ax, data, means, covariances, priors, projection_matrix,
    draw_points=True, visualisation_mode="ellipsoid", draw_means=True
):
    """Visualizes high-dimensional data, projected onto 2D/3D via projection_matrix."""
    projected_data = data @ projection_matrix
    projected_means = means @ projection_matrix
    projected_covariances = [
        projection_matrix.T @ sigma @ projection_matrix for sigma in covariances
    ]

    visualize_gmm(ax, projected_data, projected_means, projected_covariances,
                  priors, projection_matrix=None, draw_points=draw_points,
                  visualisation_mode=visualisation_mode, draw_means=draw_means)


def visualize_gmm(
    ax, data, means, covariances, priors, projection_matrix=None,
    draw_points=True, visualisation_mode="ellipsoid", draw_means=True
):
    """
    Main entry point: automatically selects 2D/3D or projects first
    if `data` has more than 3 dimensions and a `projection_matrix` has been provided.
    """
    if projection_matrix is not None:
        _visualize_gmm_higher_dimension(
            ax, data, means, covariances, priors, projection_matrix,
            draw_points, visualisation_mode, draw_means
        )
        return

    N, D = data.shape
    if D == 2:
        _visualize_gmm_nd(ax, data, means, covariances, priors, draw_points, visualisation_mode, draw_means)
    elif D == 3:
        _visualize_gmm_nd(ax, data, means, covariances, priors, draw_points, visualisation_mode, draw_means)
    else:
        print("visualisation of dimension higher than 3 is not supported yet.")

