import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from data_generation.basic_manifolds import line_in_2d, circle, swiss_roll


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


def visualize_gmm_2d(data, means, covariances, priors):
    """
    Plot 2D data and Gaussian components as ellipses.
    """
    fig, ax = plt.subplots()

    ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.8, label="Data")

    for mean, cov, prior in zip(means, covariances, priors):
        # get tha angle of the ellipse, by calculating the eigenvalues of cov
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        vals = np.clip(vals, 1e-8, None) # avoid negative eigenvalues
        width, height = 4 * np.sqrt(vals)

        ell = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            color="orange",
            alpha=min(0.6, 0.2 + prior * 3), # scale visibility by prior
        )

        ax.add_patch(ell)
        ax.scatter(*mean, c="red", s=30)

    ax.set_title("GMM Components")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    plt.show()



def plot_ellipsoid(ax, mean, cov, n_std=2.0, resolution=20, color="orange", alpha=0.2):
    """
    Plot a 3D Gaussian ellipsoid based on mean and covariance

    Parameters
    ----------
    ax : matplotlib 3D axis
    mean : shape (3,)
        Center of the ellipsoid
    cov : shape (3,3)
        Covariance matrix
    n_std : float
        Radius of ellipsoid in terms of standard deviations
    resolution : int
        Number of grid points for sphere parameterization
    """

    # Eigen decomposition
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-10, None)
    radii = n_std * np.sqrt(vals)

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




def visualize_gmm_3d(data, means, covariances, priors):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5, alpha=0.8, label="Data")

    for mean, cov, prior in zip(means, covariances, priors):
        plot_ellipsoid(ax, mean, cov)

    plt.tight_layout()
    plt.show()



def visualize_gmm_higher_dimension(data, means, covariances, priors, projection_matrix):
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
        visualize_gmm_2d(projected_data, projected_means, projected_covariances, priors)
    if D == 3:
        visualize_gmm_3d(projected_data, projected_means, projected_covariances, priors)
    
