import numpy as np

# TODO: make everything reproducable (random seed as input)

def line_in_2d(n=500):
    t = np.linspace(-3, 3, n)
    x = t
    y = np.sin(t)
    return np.stack([x, y], axis=1)


def circle(n=500):
    theta = np.linspace(0, 2 * np.pi, n)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack([x, y], axis=1)


def swiss_roll(n=1000):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    x = t * np.cos(t)
    y = 21 * np.random.rand(n)
    z = t * np.sin(t)
    return np.stack([x, y, z], axis=1)


def torus(n=1000, R=15.0, r=5.0):
    theta = 2 * np.pi * np.random.random(n)
    phi = 2 * np.pi * np.random.random(n)

    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    return np.stack([x, y, z], axis=1)


def curve_in_3d(n=1000):
    t = np.linspace(start=0.0, stop=6*np.pi, num=n)

    x = 15 * np.cos(t)
    y = 15 * np.sin(2 * t)
    z = 15 * np.sin(3 * t)

    return np.stack([x, y, z], axis=1)


def embed_data_to_dimension(data, new_dimension, noise=0.005, random_rotation=True, random_state=None):
    """
    Embed low-dimensional data into a higher-dimensional space.

    Parameters
    ----------
    data : array-like, shape (N, D)
        Input data.
    new_dimension : int
        Target dimension (must be >= D).
    noise_std : float
        Standard deviation of isotropic Gaussian noise.
    random_rotation : bool
        If True, use a random orthonormal projection.
        If False, embed by zero-padding.
    random_state : int or None
        For reproducibility.

    Returns
    -------
    data_high : array, shape (N, new_dimension)
    projection_matrix : array, shape (new_dimension, D)
    """

    rng = np.random.default_rng(random_state)

    N, D = data.shape

    if new_dimension < D:
        raise ValueError(
            "new_dimension must be >= original dimension"
        )

    # calculate the projection matrix
    if random_rotation:
        # Random matrix (since it is random, it has full rank with probability 1)
        A = rng.normal(size=(new_dimension, D))

        # QR decomposition to get orthonormal columns
        Q, _ = np.linalg.qr(A)

        W = Q[:, :D]
    else:
        # W = diag(1,...,1, 0,...,0)
        W = np.zeros((new_dimension, D))
        W[:D, :D] = np.eye(D)

    data_high = data @ W.T

    if noise > 0:
        data_high += rng.normal(
            scale=noise, size=data_high.shape
        )

    return data_high, W
