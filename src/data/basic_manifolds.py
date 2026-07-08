import numpy as np


# --------- helpers -----------------
def _make_rng(random_state):
    return np.random.default_rng(random_state)


# --------- 1d manifolds ---------------
def line_in_2d(n=500, random_state=None):
    t = np.linspace(-3, 3, n)
    x = t
    y = np.sin(t)
    data = np.stack([x, y], axis=1)
    return data


def circle(n=500, random_state=None):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    data = np.stack([x, y], axis=1)
    return data

def curve_in_3d(n=1000, random_state=None):
    t = np.linspace(0.0, 6 * np.pi, n)
    x = 15 * np.cos(t)
    y = 15 * np.sin(2 * t)
    z = 15 * np.sin(3 * t)
    data = np.stack([x, y, z], axis=1)
    return data


# ------------ 2d Manifolds ------------------
def swiss_roll(n=1000, random_state=None):
    rng = _make_rng(random_state)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x = t * np.cos(t)
    y = 21 * rng.random(n)
    z = t * np.sin(t)
    data = np.stack([x, y, z], axis=1)
    return data


def torus(n=1000, R=15.0, r=5.0, random_state=None):
    """
    Torus parametrized by two angles (theta, phi).
    The manifold is S1 x S1.
    """
    rng = _make_rng(random_state)
    theta = 2 * np.pi * rng.random(n)
    phi   = 2 * np.pi * rng.random(n)

    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    data = np.stack([x, y, z], axis=1)
    return data



# -------------------- Embedding ---------------------------
def embed_data_to_dimension(
    data, new_dimension, noise=0.005, random_rotation=True, random_state=None
):
    """
    Embed low-dimensional data into a higher-dimensional space.

    Parameters
    ----------
    data            : (N, D)
    new_dimension   : int, must be >= D
    noise           : std of isotropic Gaussian noise added after embedding
    random_rotation : if True, random orthonormal projection
    random_state    : for reproducibility

    Returns
    -------
    data_high          : (N, new_dimension)
    projection_matrix  : (new_dimension, D) -- Matrix W such that data_high = data @ W.T
    """
    rng = _make_rng(random_state)
    N, D = data.shape

    if new_dimension < D:
        raise ValueError("new_dimension must be >= original dimension")

    if random_rotation:
        A = rng.normal(size=(new_dimension, D))
        Q, _ = np.linalg.qr(A)
        W = Q[:, :D]
    else:
        W = np.zeros((new_dimension, D))
        W[:D, :D] = np.eye(D)

    data_high = data @ W.T
    if noise > 0:
        data_high += rng.normal(scale=noise, size=data_high.shape)

    return data_high, W

