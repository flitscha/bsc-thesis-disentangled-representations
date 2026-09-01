"""
Does reducing the PCA dimension improve the geometry of the MFA, and does the transform Q help?

Geometry here means how well the estimated tangents match the true manifold tangents. An MFA is
fitted across PCA dimensions from 5 to 300, densely around the optimum, ten times per dimension,
once on plain PCA and once on PCA followed by Q. What is measured is the mean angle between the
estimated and the true tangents.

The ground truth is the Fourier curve of data/fourier_curve.py, whose tangents are known. Its
signal spans 2M = 20 dimensions, so the error is expected to bottom out around D = 20. The MFA
means do not sit exactly on the curve, so the true tangent is taken at the nearest point of a
densely sampled clean copy.

Writes `tangent_alignment_vs_pca_dim.pdf` next to this file. The companion study on the density is
experiments/pca_preprocessing.py.
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
    """Prints a progress line every 'step' of the way, with an estimate of the time left."""
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


# --- configuration ---
C, H = 50, 1 # MFA components and latent dimension
# The PCA dimensions to sweep: every fifth up to AMBIENT, plus every integer in a band around the
# signal dimension 2M = 20, where the error drops steeply.
DIMS = sorted(set(range(5, AMBIENT + 1, 5)) | set(range(10, 31)))
N_REPS = 10 # random repetitions per dimension
DENSE = 4000 # resolution of the clean curve that serves as ground truth
MAX_FIT_RETRIES = 4

OUT_DIR = os.path.dirname(__file__)
FIG = "tangent_alignment_vs_pca_dim.pdf"
RESULTS_NPZ = os.path.join(os.path.dirname(__file__), "tangent_alignment_results.npz")


def build_curve(rng):
    """The noisy samples and the dense clean curve of one repetition."""
    W = random_embedding(rng) # (AMBIENT, 2M) orthonormal
    X = sample_curve(rng, W, n=N)
    dense = dense_curve(W, DENSE) # clean ambient curve
    return X, dense


def dense_tangents(dense):
    """Unit tangents of the dense clean curve, by central differences and wrapping around."""
    tang = np.roll(dense, -1, axis=0) - np.roll(dense, 1, axis=0)
    return tang / np.linalg.norm(tang, axis=1, keepdims=True)


def mean_tangent_error(R, dense_red, dense_tan_amb, model):
    """
    Mean angle in degrees between the MFA tangents and the true curve tangents.

    R             : (AMBIENT, d) reduced basis, used to project the tangents back.
    dense_red     : (DENSE, d) clean curve in the reduced coordinates.
    dense_tan_amb : (DENSE, AMBIENT) true unit tangents in the ambient space.
    """
    tangents, _, _ = extract_tangent_frame(model.A, n_tangents=1, noise=model.variance)
    T_amb = np.einsum("ad,kd->ka", R, tangents[:, :, 0]) # back into the ambient space
    T_amb /= np.linalg.norm(T_amb, axis=1, keepdims=True)

    mu = np.asarray(model.means) # (C, d)
    # the nearest point on the dense curve for every component
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
    # curve_rng has a fixed seed and builds the curves, so they are the same ones the density
    # study sees. rng is unseeded and drives Q and the MFA initialization.
    curve_rng = np.random.default_rng(CURVE_SEED)
    rng = np.random.default_rng()
    dims = [d for d in DIMS if d <= AMBIENT]
    err_without = np.full((len(dims), N_REPS), np.nan)
    err_with = np.full((len(dims), N_REPS), np.nan)
    base = np.full(N_REPS, np.nan) # no PCA at all, an MFA on the raw ambient data

    print(f"dataset: Fourier curve, AMBIENT={AMBIENT}, N={N}, noise={NOISE}, "
          f"signal dim 2M={2 * M}")

    # two fits per dimension, with and without Q, plus one baseline fit per repetition
    prog = Progress(N_REPS * (len(dims) * 2 + 1), step=0.05, label="tangent sweep")

    for r in range(N_REPS):
        X, dense = build_curve(curve_rng)
        tan_amb = dense_tangents(dense)
        mean = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - mean, full_matrices=False) # one SVD serves all dimensions
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

        # the reference without any PCA: an MFA on the raw ambient data, in the original axes
        base[r] = fit_error(X - mean, np.eye(AMBIENT), dense - mean, tan_amb)
        prog.tick()
        print(f"  rep {r+1}/{N_REPS} done", flush=True)

    os.makedirs(os.path.dirname(RESULTS_NPZ), exist_ok=True)
    np.savez(RESULTS_NPZ, dims=dims, err_without=err_without, err_with=err_with,
             baseline=base, M=M, ambient=AMBIENT, noise=NOISE, C=C)
    _plot(dims, err_without, err_with, base)
    _summary(dims, err_without, err_with, base)


def _plot(dims, err_without, err_with, base):
    dims = np.asarray(dims)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for err, color, lab in [(err_without, "C0", "PCA only"),
                            (err_with, "C1", "PCA + orthogonal transform")]:
        m, s = np.nanmean(err, axis=1), np.nanstd(err, axis=1)
        ax.plot(dims, m, color=color, label=lab)
        ax.fill_between(dims, m - s, m + s, color=color, alpha=0.2)
    bm, bs = np.nanmean(base), np.nanstd(base)
    ax.axhline(bm, color="C3", ls="--", label=f"no PCA (D={AMBIENT})")
    ax.fill_between(dims, bm - bs, bm + bs, color="C3", alpha=0.12)
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


def _summary(dims, err_without, err_with, base):
    dims = np.asarray(dims)
    for err, lab in [(err_without, "without Q"), (err_with, "with Q")]:
        m = np.nanmean(err, axis=1)
        print(f"{lab:10s}: min {m.min():.2f} deg at D={dims[np.nanargmin(m)]}, "
              f"full-D(D={dims[-1]}) {m[-1]:.2f} deg")
    print(f"no-PCA baseline (raw axes): {np.nanmean(base):.2f} "
          f"+- {np.nanstd(base):.2f} deg")


if __name__ == "__main__":
    main()

