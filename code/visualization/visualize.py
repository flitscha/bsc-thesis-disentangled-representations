import matplotlib.pyplot as plt
from data_generation.basic_manifolds import line_in_2d, circle, swiss_roll


def plot_line_in_2d(n=500):
    data = line_in_2d(n)
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=5)
    plt.title("1D curve in 2D")
    plt.axis("equal")
    plt.show()


def plot_circle(n=500):
    data = circle(n)
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=5)
    plt.title("Circle")
    plt.axis("equal")
    plt.show()


def plot_swiss_roll(n=1000):
    data = swiss_roll(n)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5)
    ax.set_title("Swiss Roll")
    plt.show()