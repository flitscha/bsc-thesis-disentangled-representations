"""
High-level manifold-learning pipeline (thesis §3.2-§3.7).

'ManifoldPipeline' wraps the individual pipeline steps behind one object that
operates on a plain numpy array X, so it is usable independently of the demos.

Two levels of granularity are available:

  Convenience (coarse):
      pipe.fit(X)          # §3.2 preprocessing + §3.3 MFA / tangent frames
      pipe.detect()        # §3.4 distances + §3.5/6 detection + §3.7 splines
      pipe.fit_detect(X)

  Fine-grained (run / inspect one step at a time; each enriches the object):
      X_red = pipe.preprocess(X)     # §3.2  -> .pre
      pipe.fit_model(X_red)          # §3.3  -> .model, .tangents, .variances
      D = pipe.distance_matrix()     # §3.4  -> .distances_
      pipe.detect_structure()        # §3.5/6 -> .structure_, .curves_ (orders)
      pipe.interpolate()             # §3.7  -> .curves_ (with splines)

Hyperparameters are set in the constructor but stay mutable (set_params /
get_params). They are grouped by the step they feed, so the "what do I re-run
after changing this?" rule lines up with the fine-grained methods:

    PCA (§3.2)         : pca_dim                                    -> preprocess()
    MFA (§3.3)         : n_components, latent_dim, cov_type, shared -> fit_model()
    distance (§3.4)    : w_overlap, w_direction                    -> distance_matrix()
    detection (§3.5/6) : detection, n_neighbors, + TDA thresholds  -> detect_structure()
    seed               : feeds preprocess() and fit_model()        -> full re-fit

Changing a parameter means re-running from its step onward. The convenience
fit() / detect() always re-run their whole stage, so a fitted model can be
re-detected with new distance / detection parameters without an expensive re-fit.
"""

import numpy as np

from core.preprocessing import PCARotation
from core.mfa import fit_mfa, extract_tangent_frame
from core.graph import compute_score_matrix
from core.tda import detect_tda
from core.traversal import detect_traversal
from core.interpolation import interpolate_curves


class ManifoldPipeline:

    # Hyperparameters grouped by the step they feed. Changing one means
    # re-running from that step onward (see module docstring).
    _PCA_PARAMS = ("pca_dim",)                                          # -> preprocess()  §3.2
    _MFA_PARAMS = ("n_components", "latent_dim", "cov_type", "shared")  # -> fit_model()  §3.3
    _DISTANCE_PARAMS = ("w_overlap", "w_direction")                    # -> distance_matrix()  §3.4
    _DETECT_PARAMS = ("detection", "n_neighbors")                      # -> detect_structure()  §3.5/6
    # `seed` feeds preprocess() and fit_model() -> full re-fit when changed
    _PARAMS = _PCA_PARAMS + _MFA_PARAMS + ("seed",) + _DISTANCE_PARAMS + _DETECT_PARAMS

    def __init__(
        self,
        n_components,
        latent_dim=1,
        pca_dim=None,
        cov_type="mfa",
        shared=False,
        detection="tda",
        n_neighbors=2,
        w_overlap=0.4,
        w_direction=0.4,
        seed=None,
        **detect_kwargs,
    ):
        # --- PCA parameter (§3.2) ---
        self.pca_dim = pca_dim if (pca_dim and pca_dim > 0) else None

        # --- MFA parameters (§3.3) ---
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.cov_type = cov_type
        self.shared = shared

        # --- distance parameters (§3.4) ---
        self.w_overlap = w_overlap
        self.w_direction = w_direction

        # --- detection parameters (§3.5/6) ---
        self.detection = detection
        self.n_neighbors = n_neighbors
        # extra kwargs forwarded to the TDA detector (min_persistence_h0/h1,
        # auto_ratio_h1)
        self.detect_kwargs = dict(detect_kwargs)

        # --- reproducibility (feeds preprocess + fit_model) ---
        self.seed = seed

        # --- fitted artifacts ---
        self.pre = None         # PCARotation, or None if pca_dim is None
        self.model = None        # fitted MFA model
        self.tangents = None     # (C, D, latent_dim) orthonormal tangent frames
        self.variances = None    # (C, latent_dim) captured variance per direction
        self.noise_ = None       # (C,) mean off-manifold noise variance
        self.obj = None          # final training objective
        self.distances_ = None   # (C, C) distance matrix (§3.4)
        self.structure_ = None   # raw detection result (index orders, §3.5/6)
        self.curves_ = None      # detected curves (with splines after interpolate)

    # ------------------------------------------------------------------
    # parameter access
    # ------------------------------------------------------------------
    def get_params(self):
        """Return all hyperparameters as a flat dict (for logging / saving)."""
        params = {k: getattr(self, k) for k in self._PARAMS}
        params.update(self.detect_kwargs)
        return params

    def set_params(self, **kwargs):
        """Update hyperparameters in place. Unknown keys go to detect_kwargs."""
        for key, value in kwargs.items():
            if key in self._PARAMS:
                setattr(self, key, value)
            else:
                self.detect_kwargs[key] = value
        return self

    # ------------------------------------------------------------------
    # §3.2 preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, X):
        """Apply PCA + random orthogonal transform; return the reduced data."""
        X = np.asarray(X)
        if self.pca_dim is not None:
            self.pre = PCARotation(self.pca_dim, rng=self.seed).fit(X)
            return self.pre.transform(X)
        self.pre = None
        return X

    # ------------------------------------------------------------------
    # §3.3 MFA as local geometry learner
    # ------------------------------------------------------------------
    def fit_model(self, X):
        """Fit the MFA on (already preprocessed) data and extract tangent frames."""
        self.model, self.obj = fit_mfa(
            X, self.n_components, self.latent_dim,
            cov_type=self.cov_type, shared=self.shared, seed=self.seed,
        )
        if self.cov_type == "mfa":
            self.tangents, self.variances, self.noise_ = extract_tangent_frame(
                self.model.A, n_tangents=self.latent_dim, noise=self.model.variance
            )
        else:
            self.tangents = self.variances = self.noise_ = None
        return self

    def fit(self, X):
        """Convenience: preprocess(X) followed by fit_model()."""
        return self.fit_model(self.preprocess(X))

    # ------------------------------------------------------------------
    # §3.4 distance matrix
    # ------------------------------------------------------------------
    def distance_matrix(self):
        """Build and cache the geometry-aware distance matrix between components."""
        if self.model is None:
            raise RuntimeError("call fit() before distance_matrix().")
        self.distances_ = compute_score_matrix(
            self.model.means, self.tangents, self.variances,
            w_overlap=self.w_overlap, w_direction=self.w_direction,
        )
        return self.distances_

    # ------------------------------------------------------------------
    # §3.5 / §3.6 structure detection (index orders, no splines)
    # ------------------------------------------------------------------
    def detect_structure(self, **overrides):
        """
        Detect components / loops from the distance matrix (builds it if needed).

        Dispatches on `self.detection`: "tda" (persistent homology, the standard)
        or "traversal" (naive k-NN graph baseline).
        """
        if self.model is None:
            raise RuntimeError("call fit() before detect_structure().")
        if self.distances_ is None:
            self.distance_matrix()

        if self.detection == "tda":
            kwargs = {**self.detect_kwargs, **overrides}
            self.structure_ = detect_tda(self.distances_, **kwargs)
        elif self.detection == "traversal":
            self.structure_ = detect_traversal(self.distances_, k=self.n_neighbors)
        else:
            raise ValueError(f"unknown detection method '{self.detection}'")

        self.structure_["method"] = self.detection
        self.curves_ = self.structure_["curves"]
        return self.structure_

    # ------------------------------------------------------------------
    # §3.7 interpolation
    # ------------------------------------------------------------------
    def interpolate(self):
        """Attach cubic splines to the detected curves."""
        if self.curves_ is None:
            raise RuntimeError("call detect_structure() before interpolate().")
        self.curves_ = interpolate_curves(self.curves_, self.model.means)
        return self.curves_

    # ------------------------------------------------------------------
    # convenience: full detection (§3.4 -> §3.7)
    # ------------------------------------------------------------------
    def detect(self, **overrides):
        """
        Run the full detection: distance matrix, structure detection, and spline
        interpolation. Rebuilds the distance matrix so changed detection
        parameters take effect. Returns a unified result dict.
        """
        if self.model is None:
            raise RuntimeError("call fit() before detect().")
        self.distance_matrix()
        self.detect_structure(**overrides)
        self.interpolate()
        return self._result()

    def fit_detect(self, X, **overrides):
        """Convenience: fit(X) followed by detect()."""
        return self.fit(X).detect(**overrides)

    def _result(self):
        """Assemble a unified result dict from the detection artifacts."""
        result = dict(self.structure_)
        result["curves"] = self.curves_
        result["distance_matrix"] = self.distances_
        if result.get("method") == "traversal":
            loop = self.curves_[0]
            result["order"] = loop["order"]
            result["spline"] = loop["spline"]
        else:
            result["loops"] = [c for c in self.curves_ if c["type"] == "loop"]
        return result

    # ------------------------------------------------------------------
    # reconstruct: map reduced-space points back to the ambient space
    # ------------------------------------------------------------------
    def reconstruct(self, points):
        """Back-project points from the reduced (PCA+rotation) space to ambient."""
        if self.pre is None:
            return np.asarray(points)
        return self.pre.inverse_transform(points)

