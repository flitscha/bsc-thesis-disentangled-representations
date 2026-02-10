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


def visualize_gmm_2d(data, means, covariances, priors):
    """
    Plot 2D data and Gaussian components as ellipses.
    """
    fig, ax = plt.subplots()

    ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.4, label="Data")

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
            alpha=min(0.6, 0.2 + prior * 5), # scale visibility by prior
        )

        ax.add_patch(ell)
        ax.scatter(*mean, c="red", s=30)

    ax.set_title("GMM Components")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    plt.show()

