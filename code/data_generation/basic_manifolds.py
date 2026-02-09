import numpy as np


def line_in_2d(n=500):
    t = np.linspace(-3, 3, n)
    x = t
    y = np.sin(t)
    return np.stack([x, y], axis=1)


def circle(n=500):
    theta = np.linspace(0, 2*np.pi, n)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack([x, y], axis=1)


def swiss_roll(n=1000):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    x = t * np.cos(t)
    y = 21 * np.random.rand(n)
    z = t * np.sin(t)
    return np.stack([x, y, z], axis=1)
