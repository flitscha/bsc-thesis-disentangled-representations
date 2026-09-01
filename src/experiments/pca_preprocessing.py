"""
Does PCA, and an extra random orthogonal transform Q, help the MFA density fit?

An MFA is fitted across PCA dimensions from 5 to 300, densely around the optimum, five times per
dimension, once on plain PCA and once on PCA followed by Q. Quality is the held-out negative
log-likelihood per sample and dimension, where lower is better. The "no PCA" line is an MFA on the
raw 300-dimensional data.

Runs on the Fourier curve of data/fourier_curve.py and writes `pca_preprocessing_nll.pdf` next to
this file. The companion study on the geometry rather than the density is
experiments/tangent_alignment.py.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from core.mfa import fit_mfa, average_nll
from core.preprocessing import random_orthogonal
from data.fourier_curve import (
    M, AMBIENT, N, NOISE, CURVE_SEED, random_embedding, sample_curve,
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
N_REPS = 5 # random repetitions per dimension
N_VAL = N # held-out validation samples, drawn independently
MAX_FIT_RETRIES = 4

OUT_DIR = os.path.dirname(__file__)
FIG = "pca_preprocessing_nll.pdf"
RESULTS_NPZ = os.path.join(os.path.dirname(__file__), "pca_preprocessing_results.npz")


def fit_nll(Z_train, Z_val):
    """Fit an MFA from a random start and return the validation NLL, retrying if the fit fails."""
    for _ in range(MAX_FIT_RETRIES):
        try:
            model, _ = fit_mfa(Z_train, C=C, H=H, seed=None)
            return average_nll(model, Z_val)
        except AssertionError:
            continue
    return np.nan


def main():
    # curve_rng has a fixed seed and builds the curves, so they are the same ones the tangent
    # study sees. rng is unseeded and drives the validation draw, Q and the MFA initialization.
    curve_rng = np.random.default_rng(CURVE_SEED)
    rng = np.random.default_rng()
    dims = [d for d in DIMS if d <= AMBIENT]

    nll_without = np.full((len(dims), N_REPS), np.nan) # plain PCA
    nll_with = np.full((len(dims), N_REPS), np.nan) # PCA followed by the random orthogonal Q
    base = np.full(N_REPS, np.nan) # no PCA at all, on the raw ambient data

    print(f"dataset: Fourier curve, AMBIENT={AMBIENT}, N={N}, noise={NOISE}, "
          f"signal dim 2M={2 * M}")

    # two fits per dimension, with and without Q, plus one baseline fit per repetition
    prog = Progress(N_REPS * (len(dims) * 2 + 1), step=0.05, label="NLL sweep")

    for r in range(N_REPS):
        # The curve of this repetition. The validation set draws from rng, so that curve_rng
        # stays in step with the tangent study.
        W = random_embedding(curve_rng)
        Xtr = sample_curve(curve_rng, W, n=N, grid=True)
        Xva = sample_curve(rng, W, n=N_VAL, grid=False)

        # A single SVD gives the full PCA coordinates, and reducing to d is just the first d
        # columns of it.
        mean_tr = Xtr.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xtr - mean_tr, full_matrices=False)
        Ztr_full = (Xtr - mean_tr) @ Vt.T
        Zva_full = (Xva - mean_tr) @ Vt.T

        for i, d in enumerate(dims):
            Ztr, Zva = Ztr_full[:, :d], Zva_full[:, :d]
            nll_without[i, r] = fit_nll(Ztr, Zva)
            prog.tick()
            Q = random_orthogonal(d, rng)
            nll_with[i, r] = fit_nll(Ztr @ Q, Zva @ Q)
            prog.tick()

        # the reference without any PCA: an MFA on the raw ambient data
        base[r] = fit_nll(Xtr, Xva)
        prog.tick()
        print(f"  rep {r + 1}/{N_REPS} done (no-PCA NLL={base[r]:+.3f})", flush=True)

    base_mean, base_std = np.nanmean(base), np.nanstd(base)
    print(f"no-PCA baseline (D={AMBIENT}): {base_mean:+.3f} +- {base_std:.3f}")

    os.makedirs(os.path.dirname(RESULTS_NPZ), exist_ok=True)
    np.savez(RESULTS_NPZ, dims=dims, nll_without=nll_without, nll_with=nll_with,
             baseline=base, D0=AMBIENT, C=C, H=H, M=M, n_samples=N, noise=NOISE)

    _plot(dims, nll_without, nll_with, base_mean, base_std, AMBIENT)
    _summary(dims, nll_without, nll_with, base_mean)


def _plot(dims, nll_without, nll_with, base_mean, base_std, D0):
    """Both curves, with and without the random transform Q, in one figure."""
    dims = np.asarray(dims)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for nll, color, lab in [(nll_without, "C0", "PCA only"),
                            (nll_with, "C1", "PCA + orthogonal transform")]:
        mean, std = np.nanmean(nll, axis=1), np.nanstd(nll, axis=1)
        ax.plot(dims, mean, color=color, label=lab)
        ax.fill_between(dims, mean - std, mean + std, color=color, alpha=0.2)
    ax.axhline(base_mean, color="C3", ls="--", label=f"no PCA (D={D0})")
    ax.fill_between(dims, base_mean - base_std, base_mean + base_std,
                    color="C3", alpha=0.12)
    ax.axvline(2 * M, color="grey", ls=":", lw=1, label=f"signal dim $2M={2 * M}$")
    ax.set_xlabel("PCA dimension $D$")
    ax.set_ylabel(r"$\mathrm{NLL}_{\mathrm{norm}}$ (validation)")
    ax.set_title("Density fit vs. PCA dimension")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, FIG))
    plt.close(fig)
    print(f"saved {os.path.join(OUT_DIR, FIG)}")


def _summary(dims, nll_without, nll_with, base_mean):
    dims = np.asarray(dims)
    mw = np.nanmean(nll_with, axis=1)
    print("\n--- summary ---")
    print(f"best dim (min NLL, with Q):  D={dims[np.nanargmin(mw)]}")
    print(f"mean std with Q:    {np.nanmean(np.nanstd(nll_with, axis=1)):.4f}")
    print(f"mean std without Q: {np.nanmean(np.nanstd(nll_without, axis=1)):.4f}")
    print(f"NLL at full D: without={np.nanmean(nll_without[-1]):+.3f}  "
          f"with={np.nanmean(nll_with[-1]):+.3f}  no-PCA={base_mean:+.3f}")


if __name__ == "__main__":
    main()

