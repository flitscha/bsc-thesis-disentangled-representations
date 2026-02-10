import matplotlib.pyplot as plt
from data_generation.basic_manifolds import line_in_2d, circle, swiss_roll


def visualize_data(data):
    N, D = data.shape
    if D == 2:
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], s=5)
        plt.axis("equal")
        plt.show()
    elif D == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5)
        plt.show()
    else:
        print("visualisation of dimension higher than 3 is not supported yet.")