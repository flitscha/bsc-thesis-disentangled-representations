import numpy as np
from data.basic_manifolds import (
    line_in_2d,
    circle,
    swiss_roll,
    embed_data_to_dimension,
    torus,
    curve_in_3d
)
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
        self.noise = 0.005
        self.seed = seed


        # --- PCA pre-projection -------------------------------------------
        # If pca_dim is set (int > 0), data is projected to that many PCA
        # dimensions before training and back-projected for reconstruction.
        self.pca_dim = pca_dim if (pca_dim is not None and pca_dim > 0) else None
        self.pca_components = None # (D_original, pca_dim) set after _apply_pca
        self.pca_mean = None # (D_original,) for zero-mean PCA

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
    # PCA helpers  (called automatically by train() when pca_dim is set)
    # ------------------------------------------------------------------
 
    def _apply_pca(self):
        """
        Project self.data from D dimensions down to pca_dim dimensions.
 
        Sets
        ----
        self.pca_mean        : (D,)           column mean of the original data
        self.pca_components  : (D, pca_dim)   top-k right singular vectors
        self.data            : (N, pca_dim)   projected data used for training
        """
        X = self.data
        self.pca_mean = X.mean(axis=0)          # (D,)
        X_centered = X - self.pca_mean
 
        # economy SVD — only compute the first pca_dim components
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.pca_components = Vt[:self.pca_dim].T   # (D, pca_dim)
 
        self.data = X_centered @ self.pca_components  # (N, pca_dim)
        print(f"[pca] projected {X.shape} -> {self.data.shape}")
 
    def reconstruct(self, points: np.ndarray) -> np.ndarray:
        """
        Back-project points from PCA space to the original data space.
 
        Parameters
        ----------
        points : (..., pca_dim)  — e.g. a single vector or a batch
 
        Returns
        -------
        (..., D_original)
 
        If no PCA was applied, returns points unchanged.
        """
        if self.pca_components is None:
            return points
        # points @ P^T  maps (pca_dim,) -> (D,), then add back the mean
        return points @ self.pca_components.T + self.pca_mean

    def train(self):

        if self.pca_dim is not None:
            self._apply_pca()

        _, D = self.data.shape

        self.model = Gaussian(
            C=self.C, D=D, covariance_type=self.cov_type, shared=self.shared, H=self.H
        )

        obj, _ = self.model.fit(self.data, verbose=False, rng=self.seed)
        self.obj = obj
