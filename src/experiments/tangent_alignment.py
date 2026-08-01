"""
Does reducing the PCA dimension improve the geometry of the MFA: how well its
tangents match the true manifold tangents. And does the random transform Q help?

We fit an MFA at 100 PCA dimensions (D = 5, 10, ..., 500), 5 times each, once on
plain PCA and once PCA+Q, and measure the mean angle between estimated and true
tangents.
Ground truth is a Fourier curve with known tangents (see data/fourier_curve.py).
Its signal spans 2M = 40 dimensions, so the error is expected to bottom out
near D = 40.

Note: The MFA means do not lie exactly on the curve, so the true tangent is taken
at the nearest point of a densely sampled clean copy.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from core.mfa import fit_mfa, extract_tangent_frame
from core.preprocessing import random_orthogonal
from data.fourier_curve import (
    M, AMBIENT, N, NOISE, CURVE_SEED, random_embedding, sample_curve, dense_curve,
)


class Progress:
    """Print a progress line every 'step' fraction of 'total' steps (with ETA)."""
    def __init__(self, total, step=0.05, label="progress"):
        self.total, self.step, self.label = total, step, label
        self.done, self.next_frac, self.t0 = 0, step, time.time()

    def tick(self, n=1):
        self.done += n
        frac = self.done / self.total
        if frac >= self.next_frac or self.done == self.total:
            el = time.time() - self.t0
            eta = el / frac - el if frac > 0 else 0.0
            print(f"  {self.label}: {frac * 100:3.0f}%  ({self.done}/{self.total} "
                  f"fits, {el:4.0f}s elapsed, ETA {eta:4.0f}s)", flush=True)
            while self.next_frac <= frac:
                self.next_frac += self.step


# ------------ configuration ------------------
C, H = 30, 1 # MFA components and latent dim
DIMS = list(range(5, AMBIENT + 1, 5)) # PCA dimensions to sweep
N_REPS = 5 # random repetitions per dim
DENSE = 4000 # resolution of the clean curve used as ground truth
MAX_FIT_RETRIES = 4

OUT_DIR = os.path.dirname(__file__)
FIG = "tangent_alignment_vs_pca_dim.pdf"
RESULTS_NPZ = os.path.join(os.path.dirname(__file__), "tangent_alignment_results.npz")


def build_curve(rng):
    """Return (X noisy samples, dense clean ambient curve) for one repetition."""
    W = random_embedding(rng) # (AMBIENT, 2M) orthonormal
    X = sample_curve(rng, W, n=N)
    dense = dense_curve(W, DENSE) # clean ambient curve
    return X, dense


def dense_tangents(dense):
    """Unit central-difference tangents of the dense clean curve (periodic)."""
    tang = np.roll(dense, -1, axis=0) - np.roll(dense, 1, axis=0)
    return tang / np.linalg.norm(tang, axis=1, keepdims=True)


def mean_tangent_error(R, dense_red, dense_tan_amb, model):
    """
    Mean angle (degrees) between MFA tangents and the true curve tangents.

    R             : (AMBIENT, d) reduced basis (PCA [+ Q]), for back-projection.
    dense_red     : (DENSE, d) clean curve in reduced coordinates.
    dense_tan_amb : (DENSE, AMBIENT) true unit tangents in ambient space.
    """
    tangents, _, _ = extract_tangent_frame(model.A, n_tangents=1, noise=model.variance)
    T_amb = np.einsum("ad,kd->ka", R, tangents[:, :, 0]) # back-project (C, AMBIENT)
    T_amb /= np.linalg.norm(T_amb, axis=1, keepdims=True)

    mu = np.asarray(model.means) # (C, d)
    # nearest dense point per component (squared distances via inner products)
    d2 = (
        np.sum(mu ** 2, axis=1)[:, None] +
        np.sum(dense_red ** 2, axis=1)[None, :] -
        2.0 * mu @ dense_red.T
    )
    nearest = np.argmin(d2, axis=1) # (C,)
    tau = dense_tan_amb[nearest] # (C, AMBIENT)

    cos = np.abs(np.sum(T_amb * tau, axis=1)).clip(0.0, 1.0)
    return np.degrees(np.arccos(cos)).mean()


def fit_error(Z, R, dense_red, dense_tan_amb):
    for _ in range(MAX_FIT_RETRIES):
        try:
            model, _ = fit_mfa(Z, C=C, H=H, seed=None)
            return mean_tangent_error(R, dense_red, dense_tan_amb, model)
        except AssertionError:
            continue
    return np.nan


def main():
    # curve_rng (fixed seed) builds the curves -> identical to the NLL experiment;
    # rng (unseeded) drives Q and the MFA init
    curve_rng = np.random.default_rng(CURVE_SEED)
    rng = np.random.default_rng()
    dims = [d for d in DIMS if d <= AMBIENT]
    err_without = np.full((len(dims), N_REPS), np.nan)
    err_with = np.full((len(dims), N_REPS), np.nan)

    print(f"dataset: Fourier curve, AMBIENT={AMBIENT}, N={N}, noise={NOISE}, "
          f"signal dim 2M={2 * M}")

    # two fits per dim (without/with Q) per rep
    prog = Progress(N_REPS * len(dims) * 2, step=0.05, label="tangent sweep")

    for r in range(N_REPS):
        X, dense = build_curve(curve_rng)
        tan_amb = dense_tangents(dense)
        mean = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - mean, full_matrices=False) # one SVD, nested PCA
        comp_full = Vt.T # (AMBIENT, AMBIENT)
        for i, d in enumerate(dims):
            comp = comp_full[:, :d] # (AMBIENT, d)
            # without Q
            Z = (X - mean) @ comp
            dense_red = (dense - mean) @ comp
            err_without[i, r] = fit_error(Z, comp, dense_red, tan_amb)
            prog.tick()
            # with random orthogonal Q
            Q = random_orthogonal(d, rng)
            Rq = comp @ Q
            err_with[i, r] = fit_error((X - mean) @ Rq, Rq, (dense - mean) @ Rq, tan_amb)
            prog.tick()
        print(f"  rep {r+1}/{N_REPS} done", flush=True)

    os.makedirs(os.path.dirname(RESULTS_NPZ), exist_ok=True)
    np.savez(RESULTS_NPZ, dims=dims, err_without=err_without, err_with=err_with,
             M=M, ambient=AMBIENT, noise=NOISE, C=C)
    _plot(dims, err_without, err_with)
    _summary(dims, err_without, err_with)


def _plot(dims, err_without, err_with):
    dims = np.asarray(dims)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for err, color, lab in [(err_without, "C0", "PCA only"),
                            (err_with, "C1", "PCA + orthogonal transform")]:
        m, s = np.nanmean(err, axis=1), np.nanstd(err, axis=1)
        ax.plot(dims, m, color=color, label=lab)
        ax.fill_between(dims, m - s, m + s, color=color, alpha=0.2)
    ax.axvline(2 * M, color="grey", ls=":", lw=1, label=f"signal dim $2M={2*M}$")
    ax.set_xlabel("PCA dimension $D$")
    ax.set_ylabel("mean tangent error (deg)")
    ax.set_title("Tangent recovery vs. PCA dimension")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, FIG))
    plt.close(fig)
    print(f"saved {os.path.join(OUT_DIR, FIG)}")


def _summary(dims, err_without, err_with):
    dims = np.asarray(dims)
    for err, lab in [(err_without, "without Q"), (err_with, "with Q")]:
        m = np.nanmean(err, axis=1)
        print(f"{lab:10s}: min {m.min():.2f} deg at D={dims[np.nanargmin(m)]}, "
              f"no-PCA(D={dims[-1]}) {m[-1]:.2f} deg")


if __name__ == "__main__":
    main()

