"""
Plot functions for the evaluation.

Each function builds and returns a matplotlib Figure; the runner saves them.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure


def plot_persistence(diagram, title="Persistence diagram"):
    """
    Persistence diagram (birth vs death) with H0 and H1 features distinguished.
    Infinite deaths are drawn on a line above the finite features.
    """
    pts = {0: [], 1: []}
    for dim, (b, d) in diagram:
        if dim in pts:
            pts[dim].append((b, d))
    finite = [d for _, (_, d) in diagram if np.isfinite(d)]
    top = (max(finite) if finite else 1.0) * 1.1
    inf_y = top * 1.05

    fig = Figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot([0, top], [0, top], "--", color="gray", lw=1)
    for dim, color, marker in ((0, "tab:blue", "o"), (1, "tab:red", "^")):
        arr = np.array(pts[dim]) if pts[dim] else np.empty((0, 2))
        if arr.size:
            fin = np.isfinite(arr[:, 1])
            ax.scatter(arr[fin, 0], arr[fin, 1], c=color, marker=marker,
                       s=40, label=f"H{dim}", alpha=0.8)
            if (~fin).any():
                ax.scatter(arr[~fin, 0], np.full((~fin).sum(), inf_y), c=color,
                           marker=marker, s=40, edgecolors="k")
    ax.axhline(inf_y, color="gray", lw=0.5, ls=":")
    ax.text(0, inf_y, " inf", va="bottom", ha="left", color="gray", fontsize=8)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_factor_residual(
    true, pred, periodic=True, factor_label="angle (deg)",
    title="Factor recovery residual"
):
    """
    Signed residual (pred - true) after post-hoc alignment, plotted against the
    true factor. For periodic factors it is wrapped to (-180, 180].
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    order = np.argsort(true)
    t, p = true[order], pred[order]
    resid = (p - t + 180.0) % 360.0 - 180.0 if periodic else p - t

    fig = Figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    ax.axhline(0, color="gray", lw=1)
    ax.scatter(t, resid, s=12, color="tab:red", alpha=0.8)
    ax.set_xlabel(f"true {factor_label}", fontsize=13)
    ax.set_ylabel("residual (deg)" if periodic else "residual", fontsize=13)
    ax.set_title(title, fontsize=15)
    fig.tight_layout()
    return fig


def _binned_mean(x, y, n_bins):
    """Mean of y per bin over the range of x; returns bin centres + means."""
    x = np.asarray(x)
    y = np.asarray(y)
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    centres, means = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(y[m].mean())
    return np.array(centres), np.array(means)


def plot_reconstruction_decomposition(
    factor, errors, factor_label="angle (deg)",
    title="Reconstruction decomposition", n_bins=40
):
    """
    full / mfa_floor / pca_floor as a function of the factor
    """
    factor = np.asarray(factor)

    fig = Figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111)
    styles = {"full": ("tab:red", "-"), "mfa_floor": ("tab:orange", "-"),
              "pca_floor": ("tab:green", "-")}
    for key, (color, ls) in styles.items():
        if key in errors:
            cx, cy = _binned_mean(factor, errors[key], n_bins)
            ax.plot(cx, cy, ls, color=color, lw=1.8, label=key)
    if "input_error" in errors:
        ie = float(np.mean(errors["input_error"]))
        if ie > 1e-9:
            ax.axhline(ie, color="gray", ls="--", lw=1, label="input error (noise)")
    ax.set_xlabel(factor_label)
    ax.set_ylabel("pixel RMSE (binned mean)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_image_strip(rows, image_shape, col_labels=None, to_image=None, title=None):
    """
    Grid of images. 'rows' is a list of (row_label, vectors (K, D0)). 'to_image'
    maps a vector to a 2D image (defaults to a plain reshape). imshow uses a
    fixed [0, 1] range so brightness is comparable across the whole strip.
    """
    to_image = to_image or (lambda v: np.asarray(v).reshape(image_shape))
    n_rows = len(rows)
    n_cols = len(rows[0][1])

    fig = Figure(figsize=(1.1 * n_cols, 1.25 * n_rows + 0.3))
    for r, (label, vecs) in enumerate(rows):
        for c in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1)
            ax.imshow(to_image(vecs[c]), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
            if col_labels is not None and r == 0:
                ax.set_title(str(col_labels[c]), fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig

