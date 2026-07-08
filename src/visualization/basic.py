"""Simple visualization of potentially projected point clouds (without GMM context)"""

import matplotlib.pyplot as plt


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
    visualize_data(projected_data)

