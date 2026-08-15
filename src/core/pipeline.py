"""
High-level manifold-learning pipeline (§5.2-§5.7).

'ManifoldPipeline' wraps all pipeline steps behind one object operating on a
plain numpy array, usable independently of the demos. Either run whole stages
(fit / detect / fit_detect) or one step at a time (preprocess, fit_model,
distance_matrix, detect_structure, interpolate), where each step enriches the
object with its artifacts.

Hyperparameters stay mutable via set_params and are grouped by the step they
feed; changing one means re-running from that step onward:

    pca_dim                                    -> preprocess()      §5.2
    n_components, latent_dim, cov_type, shared -> fit_model()       §5.3
    lambda_aniso, n_neighbors                  -> distance_matrix() §5.4
    detection, TDA thresholds, `closed`        -> detect_structure()§5.5/6
    interp_tangent_weight                      -> interpolate()     §5.7
    seed                                       -> full re-fit

detect() re-runs its whole stage, so a fitted model can be re-detected with new
distance or detection parameters without an expensive re-fit.
"""

import numpy as np

from core.preprocessing import PCARotation
from core.mfa import fit_mfa, extract_tangent_frame
from core.graph import riemannian_distance_matrix, prune_low_weight_components
from core.tda import detect_tda, persistence_thresholds
from core.ordering import detect_traversal
from core.interpolation import interpolate_curves


class ManifoldPipeline:

    # grouped by the step they feed (see module docstring)
    _PCA_PARAMS = ("pca_dim",)
    _MFA_PARAMS = ("n_components", "latent_dim", "cov_type", "shared")
    _DISTANCE_PARAMS = ("lambda_aniso", "n_neighbors")
    _DETECT_PARAMS = ("detection",)
    _INTERP_PARAMS = ("interp_tangent_weight",)
    _PARAMS = _PCA_PARAMS + _MFA_PARAMS + ("seed",) + _DISTANCE_PARAMS + _DETECT_PARAMS + _INTERP_PARAMS

    def __init__(
        self,
        n_components,
        latent_dim=1,
        pca_dim=None,
        cov_type="mfa",
        shared=False,
        detection="tda",
        lambda_aniso=30.0,
        n_neighbors=5,
        interp_tangent_weight=3.0,
        seed=None,
        **detect_kwargs,
    ):
        self.pca_dim = pca_dim if (pca_dim and pca_dim > 0) else None

        self.n_components = n_components
        self.latent_dim = latent_dim
        self.cov_type = cov_type
        self.shared = shared

        self.lambda_aniso = lambda_aniso   # off-manifold penalty of the local metric
        self.n_neighbors = n_neighbors     # neighbors of the geodesic graph

        self.interp_tangent_weight = interp_tangent_weight  # chart-tangent alignment strength

        self.detection = detection
        # TDA thresholds (min_persistence_h0/h1, component_gap_factor_h0,
        # prominence_ratio_h1) or the traversal baseline's `closed` override
        self.detect_kwargs = dict(detect_kwargs)

        self.seed = seed

        # --- fitted artifacts ---
        self.pre = None          # PCARotation, or None if pca_dim is None
        self.model = None        # fitted MFA model
        self.tangents = None     # (C, D, latent_dim) orthonormal tangent frames
        self.variances = None    # (C, latent_dim) captured variance per direction
        self.noise_ = None       # (C,) mean off-manifold noise variance
        self.obj = None          # final training objective
        self.kept_ = None        # original indices of the charts kept after pruning
        self.distances_ = None   # (M, M) distance matrix over the kept charts
        self.structure_ = None   # raw detection result (index orders)
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
    # §5.2 preprocessing
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
    # §5.3 MFA as local geometry learner
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
    # §5.4 distance matrix
    # ------------------------------------------------------------------
    def distance_matrix(self):
        """Build and cache the geometry-aware distance matrix between components.

        Near-empty charts (negligible mixing weight pi_k) are pruned first, so
        they cannot create shortcut edges or spurious topology.
        """
        if self.model is None:
            raise RuntimeError("call fit() before distance_matrix().")
        self.kept_ = prune_low_weight_components(self.model.prior)
        self.distances_ = riemannian_distance_matrix(
            self.model.means[self.kept_], self.tangents[self.kept_],
            lambda_aniso=self.lambda_aniso, n_neighbors=self.n_neighbors,
        )
        return self.distances_

    # ------------------------------------------------------------------
    # §5.5 / §5.6 structure detection (index orders, no splines)
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
            kwargs.pop("closed", None)  # traversal-only override, not a TDA param
            self.structure_ = detect_tda(self.distances_, **kwargs)
        elif self.detection == "traversal":
            closed = {**self.detect_kwargs, **overrides}.get("closed")
            self.structure_ = detect_traversal(self.distances_, closed=closed)
        else:
            raise ValueError(f"unknown detection method '{self.detection}'")

        self._lift_indices_to_original()
        self.structure_["method"] = self.detection
        self.curves_ = self.structure_["curves"]
        return self.structure_

    def _lift_indices_to_original(self):
        """
        Map detection indices back to the original component indices via 'self.kept_'
        """
        kept = self.kept_
        N = self.model.means.shape[0]

        for curve in self.structure_["curves"]:
            curve["order"] = [int(kept[i]) for i in curve["order"]]

        # per-node component labels: scatter back to full length, -1 = pruned
        if "components" in self.structure_:
            full = np.full(N, -1, dtype=int)
            full[kept] = self.structure_["components"]
            self.structure_["components"] = full

        # traversal baseline exposes a k-NN graph whose edges are node indices
        graph = self.structure_.get("graph")
        if graph is not None and "edges" in graph:
            graph["edges"] = [(int(kept[i]), int(kept[j])) for i, j in graph["edges"]]

    def apply_persistence_thresholds(self, h0_factor=0.0, h1_factor=0.0):
        """
        Re-detect with explicit H0/H1 persistence thresholds and re-interpolate.

        Both factors are multiples of the barcode's median H0 merge scale, which
        keeps them free of the data scale but is only known once a detection has
        run: call this after detect() / detect_structure(). A factor of 0 keeps
        the automatic rule of §5.6 for that dimension, and if both are 0 this is
        a no-op (no re-detection at all).
        """
        if self.structure_ is None:
            raise RuntimeError("call detect_structure() before applying thresholds.")

        diagram = self.structure_.get("diagram") # only the TDA branch has one
        if diagram is None:
            return self

        thresholds = persistence_thresholds(diagram, h0_factor=h0_factor, h1_factor=h1_factor)
        if not thresholds:
            return self

        self.set_params(**thresholds)
        self.detect_structure()
        self.interpolate()
        return self

    # ------------------------------------------------------------------
    # §5.7 interpolation
    # ------------------------------------------------------------------
    def interpolate(self):
        """Attach smooth curves to the detected structures.

        The curve interpolates the component means exactly (Hermite spline); its
        knot tangents are softly aligned with the MFA chart tangents, weighted by
        the mixing weight pi_k (low-pi_k charts barely pull). See core.interpolation.
        """
        if self.curves_ is None:
            raise RuntimeError("call detect_structure() before interpolate().")
        self.curves_ = interpolate_curves(
            self.curves_, self.model.means,
            weights=self.model.prior, tangents=self.tangents,
            tangent_weight=self.interp_tangent_weight,
        )
        return self.curves_

    # ------------------------------------------------------------------
    # convenience: full detection (§5.4 -> §5.7)
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
        return self.result()

    def fit_detect(self, X, **overrides):
        """Convenience: fit(X) followed by detect()."""
        return self.fit(X).detect(**overrides)

    def result(self):
        """
        Assemble a unified result dict from the current detection artifacts.

        What detect() returns, but callable again after the artifacts changed
        (e.g. after apply_persistence_thresholds) without re-running detection.
        """
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

    # ------------------------------------------------------------------
    # encode: map observations onto the learned 1D coordinate (f: X -> Z)
    # ------------------------------------------------------------------
    def transform(self, X, n_samples=512):
        """
        Encode (N, D0) observations into the learned 1D coordinate: the map f: X -> Z.

        Each observation is projected onto the nearest detected curve, found by
        sampling every spline at 'n_samples' points. Label-free, so it stays part
        of the unsupervised model; ground-truth alignment happens at evaluation.

        Returns t in [0, 1] (in [0, 1) for periodic loops) and the index into
        self.curves_ of the assigned curve.
        """
        if self.curves_ is None:
            raise RuntimeError("call detect() before transform().")
        Z = self.pre.transform(np.asarray(X)) if self.pre is not None else np.asarray(X)
        N = Z.shape[0]
        z2 = np.sum(Z * Z, axis=1)[:, None] # (N, 1)

        best_t = np.zeros(N)
        best_id = np.zeros(N, dtype=int)
        best_d2 = np.full(N, np.inf)

        for cid, curve in enumerate(self.curves_):
            spline = curve.get("spline")
            if spline is None:
                continue
            endpoint = curve.get("type") != "loop"  # loops close, so t=1 duplicates t=0
            ts = np.linspace(0.0, 1.0, n_samples, endpoint=endpoint)
            pts = np.asarray(spline(ts)) # (n_samples, D)
            # squared distances (N, n_samples): |z|^2 + |p|^2 - 2 z.p
            d2 = z2 + np.sum(pts * pts, axis=1)[None, :] - 2.0 * (Z @ pts.T)
            j = np.argmin(d2, axis=1)
            dmin = d2[np.arange(N), j]
            take = dmin < best_d2
            best_d2[take] = dmin[take]
            best_t[take] = ts[j[take]]
            best_id[take] = cid
        return best_t, best_id

