"""
Multi-frequency Fourier curve, the shared synthetic test dataset for the two
PCA-preprocessing experiments (thesis §5.2):

- `experiments/pca_preprocessing.py` measures density fit (NLL_norm),
- `experiments/tangent_alignment.py` measures geometry (tangent error).

The curve is a closed 1D loop whose image spans exactly 2*M linear dimensions.
It is randomly embedded into `AMBIENT` dimensions by an orthonormal map W
and perturbed by small isotropic noise.
"""

import numpy as np

# ---------- dataset configuration (shared by both experiments) --------
M = 10 # Fourier modes -> signal spans 2*M = 20 linear dimensions
AMBIENT = 300 # ambient dimension the curve is embedded in
N = 4000 # noisy sample points used to fit the MFA
NOISE = 0.15 # std of the isotropic ambient noise on the samples
CURVE_SEED = 42 # shared seed for the curve RNG so both experiments use the same curves


def fourier_curve(t, m=M):
    """Signal-space coordinates of the curve at parameters t: shape (len(t), 2m)."""
    modes = np.arange(1, m + 1)
    ang = np.outer(t, modes) # (T, m)
    coords = np.empty((len(t), 2 * m))
    coords[:, 0::2] = np.cos(ang)
    coords[:, 1::2] = np.sin(ang)
    return coords


def random_embedding(rng, ambient=AMBIENT, m=M):
    """A random orthonormal embedding W: (ambient, 2m) of the signal space."""
    W, _ = np.linalg.qr(rng.standard_normal((ambient, 2 * m)))
    return W


def sample_curve(rng, W, n=N, noise=NOISE, m=M, grid=True):
    """
    'n' noisy samples of the curve embedded through 'W': shape (n, ambient).

    grid=True  : evenly spaced parameters
    grid=False : independent uniform-random parameters (used for the NLL validation set)
    """
    if grid:
        t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    else:
        t = rng.uniform(0.0, 2 * np.pi, size=n)
    return fourier_curve(t, m) @ W.T + rng.normal(scale=noise, size=(n, W.shape[0]))


def dense_curve(W, dense, m=M):
    """Clean (noise-free) ambient curve on a dense grid: shape (dense, ambient)."""
    t = np.linspace(0.0, 2 * np.pi, dense, endpoint=False)
    return fourier_curve(t, m) @ W.T

