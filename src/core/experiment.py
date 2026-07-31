import numpy as np
from data.basic_manifolds import (
    line_in_2d,
    circle,
    swiss_roll,
    embed_data_to_dimension,
    torus,
    curve_in_3d
)
from core.preprocessing import PCARotation
from vamm import Gaussian


class Experiment:
    def __init__(
        self,
        data_type,
        N,
        C,
        H,
        cov_type,
        shared=False,
        embed_dim=None,
        noise=0.005,
        pca_dim=None,
        seed=None,
    ):
        # type of toy-data ("line", "circle", "swiss_roll")
        self.data_type = data_type
        self.N = N  # number of data points
        self.C = C  # number of gaussian components
        # assumption that the data lives on H-dimensional manifold (needed for mfa)
        self.H = H
        self.cov_type = cov_type  # ("isotropic", "diagonal", "mfa", "full")
        self.shared = shared  # use shared covariances?
        self.embed_dim = embed_dim
        if self.embed_dim == 0:
            self.embed_dim = None
        self.noise = noise
        self.seed = seed

        # --- PCA pre-projection (thesis §3.2, see core/preprocessing.py) ---
        self.pca_dim = pca_dim if (pca_dim is not None and pca_dim > 0) else None
        self.pre = None  # fitted PCARotation, set in _apply_pca

        self.data = None
        self.projection_matrix = None
        self.model = None  # model for training
        self.obj = None  # objective (how good is the training-result?)

    def generate_data(self):
        data = None
        if self.data_type == "line":
            data = line_in_2d(n=self.N)
        elif self.data_type == "circle":
            data = circle(n=self.N)
        elif self.data_type == "swiss_roll":
            data = swiss_roll(n=self.N)
        elif self.data_type == "torus":
            data = torus(n=self.N)
        elif self.data_type == "curve_in_3d":
            data = curve_in_3d(n=self.N)
        else:
            print('warning: data type "' + self.data_type + '" does not exist')

        if self.embed_dim is None:
            self.data = data
        else:
            self.data, self.projection_matrix = embed_data_to_dimension(
                data, self.embed_dim, noise=self.noise, random_state=self.seed
            )

    # ------------------------------------------------------------------
    # PCA + random orthogonal transformation (delegated to PCARotation)
    # ------------------------------------------------------------------

    def _apply_pca(self):
        """Fit the PCA + random orthogonal transform and project self.data."""
        X = self.data
        self.pre = PCARotation(self.pca_dim, rng=self.seed).fit(X)
        self.data = self.pre.transform(X)
        print(f"[pca + rot] projected {X.shape} -> {self.data.shape}")

    def reconstruct(self, points: np.ndarray) -> np.ndarray:
        """Back-project points from rotated PCA space to the original data space."""
        if self.pre is None:
            return points
        return self.pre.inverse_transform(points)

    # Backward-compatible accessors for the fitted transform.
    @property
    def pca_components(self):
        return None if self.pre is None else self.pre.components_

    @property
    def pca_mean(self):
        return None if self.pre is None else self.pre.mean_

    @property
    def pca_rotation(self):
        return None if self.pre is None else self.pre.rotation_


    def train(self):
        if self.pca_dim is not None:
            self._apply_pca()

        _, D = self.data.shape

        self.model = Gaussian(
            C=self.C, D=D, covariance_type=self.cov_type, shared=self.shared, H=self.H
        )

        obj, _ = self.model.fit(self.data, verbose=False, rng=self.seed)
        self.obj = obj

