"""
Pipeline step 1 (thesis §5.2): dimensionality reduction by PCA followed by a
random orthogonal transformation.

Standard PCA aligns the coordinate axes with the principal directions of the
data. This alignment is unfavorable for an MFA model with diagonal noise
covariance, because variation along individual PCA coordinates may be absorbed
by the noise covariance instead of the factor loading matrices. Applying a
random orthogonal transformation preserves the reduced subspace (distances and
angles are unchanged) but removes the preferred alignment with the PCA axes.
"""

import numpy as np


def random_orthogonal(dim: int, rng=None) -> np.ndarray:
    """
    Draw a Haar-uniform random orthogonal matrix Q in O(dim) (Q^T Q = I).

    Constructed from the QR decomposition of a matrix with i.i.d. standard
    normal entries. The sign correction Q * diag(R)/|diag(R)| makes the
    distribution exactly Haar-uniform. The result may contain a reflection
    (det = -1), i.e. it is an orthogonal transformation, not merely a rotation.

    Parameters
    ----------
    dim : int
    rng : np.random.Generator, int or None

    Returns
    -------
    Q : (dim, dim) orthogonal matrix
    """
    rng = np.random.default_rng(rng)
    H = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(H)
    d = np.diag(R)
    return Q * (d / np.abs(d))


class PCARotation:
    """
    PCA projection onto ``n_components`` dimensions followed by a random
    orthogonal transformation (thesis §5.2).

    Follows the sklearn-style fit / transform / inverse_transform convention so
    the same fitted transform can be applied to a separate validation set.

    Attributes (set after ``fit``)
    ------------------------------
    mean_ : (D0,)
        Empirical mean of the training data, removed before projection.
    rotation_ : (d, d)
        The random orthogonal matrix, where d = min(n_components, rank(X)).
    components_ : (D0, d)
        Combined transform (PCA basis @ rotation); columns map ambient space
        to the rotated reduced coordinates.
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

        # Standard PCA via SVD; right singular vectors are the principal axes.
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Guard against requesting more components than available directions.
        dim = min(self.n_components, Vt.shape[0])
        basis = Vt[:dim].T  # (D0, dim)

        self.rotation_ = random_orthogonal(dim, self.rng)  # (dim, dim)
        self.components_ = basis @ self.rotation_  # (D0, dim)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X) - self.mean_) @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Back-project from the rotated reduced space to the ambient space."""
        return np.asarray(Z) @ self.components_.T + self.mean_
