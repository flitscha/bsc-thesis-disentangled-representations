"""
Preprocessing: PCA followed by a random orthogonal transformation.

PCA aligns the axes with the principal directions, which hurts an MFA with a diagonal noise
covariance: variation along a single axis can be absorbed by the noise instead of by the loading
matrices. The random orthogonal transformation keeps the subspace, and with it all distances and
angles, but removes that alignment.
"""

import numpy as np


def random_orthogonal(dim: int, rng=None) -> np.ndarray:
    """
    Draw a Haar-uniform random orthogonal (dim, dim) matrix.

    Take the QR decomposition of a matrix of i.i.d. standard normals. The sign correction
    Q * diag(R)/|diag(R)| is what makes the distribution exactly Haar-uniform. The result may
    contain a reflection, so it is orthogonal but not always a rotation.
    """
    rng = np.random.default_rng(rng)
    H = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(H)
    d = np.diag(R)
    return Q * (d / np.abs(d))


class PCARotation:
    """
    PCA onto 'n_components' dimensions, then a random orthogonal transformation.

    Follows the sklearn fit / transform / inverse_transform convention, so the fitted transform can
    be reused on a validation set. After 'fit', with d = min(n_components, rank(X)): 'mean_' (D0,)
    is the training mean, 'rotation_' (d, d) the random matrix, 'components_' (D0, d) their product.
    """

    def __init__(self, n_components: int, rng=None):
        self.n_components = n_components
        self.rng = rng
        self.mean_ = None
        self.rotation_ = None
        self.components_ = None

    def fit(self, X: np.ndarray) -> "PCARotation":
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # PCA via SVD: the right singular vectors are the principal axes
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)

        dim = min(self.n_components, Vt.shape[0]) # cannot exceed the rank
        basis = Vt[:dim].T

        self.rotation_ = random_orthogonal(dim, self.rng)
        self.components_ = basis @ self.rotation_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X) - self.mean_) @ self.components_

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Back-project from the rotated reduced space to the ambient space."""
        return np.asarray(Z) @ self.components_.T + self.mean_

