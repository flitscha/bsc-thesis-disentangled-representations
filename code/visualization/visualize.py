import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def visualize_data(data):
    N, D = data.shape
    if D == 2:
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], s=5)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()
    elif D == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5)
        plt.tight_layout()
        plt.show()
    else:
        print("visualisation of dimension higher than 3 is not supported yet.")


def visualize_embedded_data(data, projection_matrix):
    projected_data = data @ projection_matrix
    N, D = projected_data.shape
    visualize_data(projected_data)


def plot_ellipse(ax, mean, cov, prior):
    # get tha angle of the ellipse, by calculating the eigenvalues of cov
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    vals = np.clip(vals, 1e-8, None)  # avoid negative eigenvalues
    width, height = 4 * np.sqrt(vals)

    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        color="orange",
        alpha=min(0.6, 0.2 + prior * 3),  # scale visibility by prior
    )

    ax.add_patch(ell)


def plot_line(ax, mean, cov, prior, length=0.5):
    vals, vecs = np.linalg.eigh(cov)

    # use the eigenvector with the biggest eigenvalue as direction
    idx = np.argmax(vals)
    direction = vecs[:, idx]

    # fixed length
    direction = direction * (length / 2.0)

    p1 = mean - direction
    p2 = mean + direction

    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color="orange",
        linewidth=3.0,
        alpha=min(0.8, 0.3 + prior * 2)
    )


def visualize_gmm_2d(
    ax, data, means, covariances, priors, draw_points=True,
    visualisation_mode="ellipsoid", draw_means=True
):
    """
    Plot 2D data and Gaussian components as ellipses.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if draw_points:
        ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.8, label="Data")

    if draw_means:
        ax.scatter(means[:, 0], means[:, 1], s=20, c="red", label="Means")

    # set limits
    x_min, y_min = data.min(axis=0)
    x_max, y_max = data.max(axis=0)
    padding = 0.5
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect("equal")

    # draw the lines to visualize the covariances
    plot_functions = {
        "ellipsoid": plot_ellipse,
        "line": plot_line,
        "plane": plot_line,  # just draw the line, since we are in a 2d-plot
        "none": lambda ax, mean, cov, prior: None,
    }

    plot_function = plot_functions[visualisation_mode]

    for mean, cov, prior in zip(means, covariances, priors):
        plot_function(ax, mean, cov, prior)


def plot_ellipsoid(ax, mean, cov, resolution=20, color="orange", alpha=0.2):
    """
    Plot a 3D Gaussian ellipsoid based on mean and covariance

    Parameters
    ----------
    ax : matplotlib 3D axis
    mean : shape (3,)
        Center of the ellipsoid
    cov : shape (3,3)
        Covariance matrix
    resolution : int
        Number of grid points for sphere parameterization
    """

    # Eigen decomposition
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-10, None)
    radii = 2.0 * np.sqrt(vals)

    # Parametric sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([x, y, z], axis=0).reshape(3, -1)

    # make the sphere to an ellipsoid by scaling and rotating it
    ellipsoid = radii[:, np.newaxis] * sphere
    ellipsoid = vecs @ ellipsoid
    ellipsoid = ellipsoid + mean[:, np.newaxis]

    x_e = ellipsoid[0].reshape(resolution, resolution)
    y_e = ellipsoid[1].reshape(resolution, resolution)
    z_e = ellipsoid[2].reshape(resolution, resolution)

    ax.plot_surface(x_e, y_e, z_e, color=color, alpha=alpha, linewidth=0)


def plot_plane(ax, mean, cov, size=3.0, color="orange", alpha=0.2):
    vals, vecs = np.linalg.eigh(cov)

    # use the two directions with the largest eigenvalues
    idx = np.argsort(vals)[::-1]
    vec1 = vecs[:, idx[0]]
    vec2 = vecs[:, idx[1]]

    # scale with fixed size
    vec1 = vec1 * size
    vec2 = vec2 * size

    # four vertices of the plane
    p1 = mean - vec1 - vec2
    p2 = mean - vec1 + vec2
    p3 = mean + vec1 - vec2
    p4 = mean + vec1 + vec2

    # plot the surface
    X = np.array([[p1[0], p2[0]],
                  [p3[0], p4[0]]])

    Y = np.array([[p1[1], p2[1]],
                  [p3[1], p4[1]]])

    Z = np.array([[p1[2], p2[2]],
                  [p3[2], p4[2]]])

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)


def plot_line_in_3d(ax, mean, cov, length=3.0, color="orange", alpha=0.8):
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argmax(vals)
    direction = vecs[:, idx]
    direction = direction * (length / 2.0)

    p1 = mean - direction
    p2 = mean + direction

    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        color="orange",
        linewidth=3.0,
        alpha=alpha
    )


def visualize_gmm_3d(
    ax, data, means, covariances, priors, draw_points=True,
    visualisation_mode="ellipsoid", draw_means=True
):
    if draw_points:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                   s=5, alpha=0.8, label="Data")

    if draw_means:
        ax.scatter(means[:, 0], means[:, 1], means[:, 2],
                   s=10, label="Means", c="red")

    # set limits
    x_min, y_min, z_min = data.min(axis=0)
    x_max, y_max, z_max = data.max(axis=0)
    padding = 0.5
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_zlim(z_min - padding, z_max + padding)
    ax.set_aspect("equal")

    plot_functions = {
        "ellipsoid": plot_ellipsoid,
        "line": plot_line_in_3d,
        "plane": plot_plane,
        "none": lambda ax, mean, cov: None,
    }

    plot_function = plot_functions[visualisation_mode]

    for mean, cov, prior in zip(means, covariances, priors):
        plot_function(ax, mean, cov)


def _visualize_gmm_higher_dimension(
    ax, data, means, covariances, priors, projection_matrix,
    draw_points=True, visualisation_mode="ellipsoid", draw_means=True
):
    """
    visualize data in higher dimensions by projecting it down using the projection_matrix
    """
    projected_data = data @ projection_matrix
    projected_means = means @ projection_matrix
    projected_covariances = [
        projection_matrix.T @ sigma @ projection_matrix
        for sigma in covariances
    ]

    N, D = projected_data.shape
    if D == 2:
        visualize_gmm_2d(
            ax,
            projected_data,
            projected_means,
            projected_covariances,
            priors,
            draw_points=draw_points,
            visualisation_mode=visualisation_mode,
            draw_means=draw_means
        )
    if D == 3:
        visualize_gmm_3d(
            ax,
            projected_data,
            projected_means,
            projected_covariances,
            priors,
            draw_points=draw_points,
            visualisation_mode=visualisation_mode,
            draw_means=draw_means
        )


def visualize_gmm(
    ax,
    data,
    means,
    covariances,
    priors,
    projection_matrix=None,
    draw_points=True,
    visualisation_mode="ellipsoid",
    draw_means=True
):
    if projection_matrix is not None:
        _visualize_gmm_higher_dimension(
            ax, data, means, covariances, priors, projection_matrix,
            draw_points, visualisation_mode, draw_means
        )
        return

    N, D = data.shape
    if D == 2:
        visualize_gmm_2d(
            ax, data, means, covariances, priors, draw_points=draw_points,
            visualisation_mode=visualisation_mode, draw_means=draw_means
        )
    if D == 3:
        visualize_gmm_3d(
            ax, data, means, covariances, priors, draw_points=draw_points,
            visualisation_mode=visualisation_mode, draw_means=draw_means
        )


def visualize_graph_on_mfa(
    ax,
    means,
    covariances,
    priors,
    edges=None,  # list of tuples [(i,j), ...]
    draw_nodes=True,
    visualisation_mode="line",
    node_color="red",
    edge_color="blue",
    edge_alpha=0.6
):
    """
    Visualize MFA components (means + covariances) and optionally a graph connecting them.

    Parameters
    ----------
    ax : matplotlib axis
    means : ndarray (N, D)
    covariances : list of ndarray (N, D, D)
    priors : ndarray (N,)
    edges : list of tuples (i,j) optional
        Each tuple is a connection between mean i and mean j
    """
    N, D = means.shape
    # Draw MFA components
    if D == 2:
        for mean, cov, prior in zip(means, covariances, priors):
            if visualisation_mode == "line":
                plot_line(ax, mean, cov, prior)
            elif visualisation_mode == "ellipsoid":
                plot_ellipse(ax, mean, cov, prior)
        if draw_nodes:
            ax.scatter(means[:, 0], means[:, 1], s=20, c=node_color, zorder=3)

        # Draw edges
        if edges is not None:
            for i, j in edges:
                ax.plot(
                    [means[i, 0], means[j, 0]],
                    [means[i, 1], means[j, 1]],
                    color=edge_color,
                    alpha=edge_alpha,
                    linewidth=2
                )
    elif D == 3:
        for mean, cov, prior in zip(means, covariances, priors):
            if visualisation_mode == "line":
                plot_line_in_3d(ax, mean, cov)
            elif visualisation_mode == "ellipsoid":
                plot_ellipsoid(ax, mean, cov)
        if draw_nodes:
            ax.scatter(means[:, 0], means[:, 1], means[:, 2],
                       s=20, c=node_color, zorder=3)

        # Draw edges
        if edges is not None:
            for i, j in edges:
                ax.plot(
                    [means[i, 0], means[j, 0]],
                    [means[i, 1], means[j, 1]],
                    [means[i, 2], means[j, 2]],
                    color=edge_color,
                    alpha=edge_alpha,
                    linewidth=2
                )
