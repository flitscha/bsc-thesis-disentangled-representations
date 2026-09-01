"""
Toy manifolds with known topology, the ground truth of the demo tabs.

Every generator returns an (n, D) point cloud and takes the same '(n, random_state)' arguments, so
that 'data.synthetic.make_dataset' can dispatch on a name alone. The deterministic shapes ignore
'random_state' and only accept it to keep that interface uniform.

'embed_data_to_dimension' lifts any of the shapes into a higher-dimensional space, which is what
makes the toy problem resemble the real datasets.
"""

import numpy as np


# --- helpers ---
def _make_rng(random_state):
    return np.random.default_rng(random_state)


# --- 1d manifolds ---
def line_in_2d(n=500, random_state=None):
    """An open arc: the graph of sin(t) over t in [-3, 3]. One component, no loop."""
    t = np.linspace(-3, 3, n)
    x = t
    y = np.sin(t)
    data = np.stack([x, y], axis=1)
    return data


def circle(n=500, random_state=None):
    """The unit circle, sampled without the duplicate endpoint. One loop."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    data = np.stack([x, y], axis=1)
    return data


def curve_in_3d(n=1000, random_state=None):
    """
    A closed Lissajous curve in 3D, wound three times around itself.

    The curve comes close to itself in several places, so the Euclidean distance is misleading
    here and only the tangent-aware one follows it.
    """
    t = np.linspace(0.0, 6 * np.pi, n)
    x = 15 * np.cos(t)
    y = 15 * np.sin(2 * t)
    z = 15 * np.sin(3 * t)
    data = np.stack([x, y, z], axis=1)
    return data


# --- 2d manifolds ---
def swiss_roll(n=1000, random_state=None):
    """The classic rolled-up 2D sheet in 3D: one component, no loop."""
    rng = _make_rng(random_state)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x = t * np.cos(t)
    y = 21 * rng.random(n)
    z = t * np.sin(t)
    data = np.stack([x, y, z], axis=1)
    return data


def torus(n=1000, R=15.0, r=5.0, random_state=None):
    """
    Torus of tube radius r around a circle of radius R, sampled uniformly in both angles.

    The manifold is S1 x S1, so it carries two independent loops.
    """
    rng = _make_rng(random_state)
    theta = 2 * np.pi * rng.random(n)
    phi   = 2 * np.pi * rng.random(n)

    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    data = np.stack([x, y, z], axis=1)
    return data



# --- shapes for testing the TDA detector ---
# Each shape below has a topology the traversal baseline cannot express, since that one always
# produces a single curve. The multi-component ones return ground-truth labels on request, which
# is what the TDA tab scores its detection against.

def half_circle(n=500, random_state=None):
    """Half of the unit circle: one component that does not close. One arc."""
    theta = np.linspace(0, np.pi, n)
    x = np.cos(theta)
    y = np.sin(theta)
    data = np.stack([x, y], axis=1)
    return data


def two_circles(n=500, radius=1.0, separation=3.0, return_labels=False, random_state=None):
    """
    Two circles side by side, 'separation' apart: two components, two loops.

    The test case for H0. 'separation' controls how far apart the structures are, and with it how
    visible the gap in the H0 barcode becomes.
    """
    n1 = n // 2
    n2 = n - n1
    theta1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    theta2 = np.linspace(0, 2 * np.pi, n2, endpoint=False)

    c1 = np.stack([radius * np.cos(theta1) - separation / 2, radius * np.sin(theta1)], axis=1)
    c2 = np.stack([radius * np.cos(theta2) + separation / 2, radius * np.sin(theta2)], axis=1)

    data = np.concatenate([c1, c2], axis=0)
    if return_labels:
        labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
        return data, labels
    return data


def circle_and_half_circle(n=500, radius=1.0, separation=3.0, return_labels=False, random_state=None):
    """
    A full circle next to a half one: two components, one loop and one arc.

    The mixed case, where the detector has to keep H0 and H1 apart instead of assuming that every
    component closes.
    """
    n1 = n // 2
    n2 = n - n1
    theta_full = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    theta_half = np.linspace(0, np.pi, n2)

    full = np.stack([radius * np.cos(theta_full) - separation / 2, radius * np.sin(theta_full)], axis=1)
    half = np.stack([radius * np.cos(theta_half) + separation / 2, radius * np.sin(theta_half)], axis=1)

    data = np.concatenate([full, half], axis=0)
    if return_labels:
        labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
        return data, labels
    return data


def linked_circles_3d(n=1000, radius=1.0, offset=None, return_labels=False, random_state=None):
    """
    Two interlocking circles (Hopf link): two components, two loops.

    C1 lies in the xy-plane around the origin, C2 in the xz-plane shifted by 'offset' along x, so
    each circle passes through the disc spanned by the other. This is the hard case, because the
    two loops come very close without ever touching, and that is exactly the situation the
    detector struggles with on real data.
    """
    if offset is None:
        offset = radius

    n1 = n // 2
    n2 = n - n1
    theta1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    theta2 = np.linspace(0, 2 * np.pi, n2, endpoint=False)

    c1 = np.stack([radius * np.cos(theta1), radius * np.sin(theta1), np.zeros(n1)], axis=1)
    c2 = np.stack([offset + radius * np.cos(theta2), np.zeros(n2), radius * np.sin(theta2)], axis=1)

    data = np.concatenate([c1, c2], axis=0)
    if return_labels:
        labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
        return data, labels
    return data


# --- embedding ---
def embed_data_to_dimension(
    data, new_dimension, noise=0.005, random_rotation=True, random_state=None
):
    """
    Embed (N, D) data into 'new_dimension' >= D dimensions.

    Uses a random orthonormal map unless 'random_rotation' is False, then adds isotropic Gaussian
    noise of standard deviation 'noise'. Returns the embedded points and the map W, where
    data_high = data @ W.T.
    """
    rng = _make_rng(random_state)
    D = data.shape[1]

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

