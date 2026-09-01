"""
The plots of the evaluation.

Each function builds a matplotlib Figure and returns it. Saving is left to the caller.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure


def plot_persistence(diagram, title="Persistence diagram", h1_labels=None):
    """
    Persistence diagram of birth against death, with H0 and H1 drawn apart.

    Infinite deaths go on a line above the finite features. 'h1_labels' annotates the H1 markers,
    one dict per loop with the keys "birth", "death" and "label". Those labels are the same ones
    the sweep figures use, so both name the same loops.
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

    # annotate the significant H1 triangles with the loop identity from the sweep figures.
    for item in h1_labels or []:
        b, d = float(item["birth"]), float(item["death"])
        y = d if np.isfinite(d) else inf_y
        ax.annotate(item["label"], (b, y),
                    xytext=(5, -1), textcoords="offset points",
                    fontsize=8, color="tab:red", ha="left", va="top")

    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def _is_pruned(label):
    """A chart no observation backs is labelled -1, while a class label may well be a string."""
    return np.issubdtype(np.asarray(label).dtype, np.number) and label < 0


def plot_component_scatter(
    Z2, digit_id, component_id, title="Detected structure", subtitle=None,
    label_name="ground-truth digit",
):
    """
    Two scatter panels of the same (N, 2) embedding: ground truth on the left, detection on the
    right.
    """
    Z2 = np.asarray(Z2)
    digit_id = np.asarray(digit_id)
    component_id = np.asarray(component_id)

    fig = Figure(figsize=(10, 4.8))
    for ax_i, (labels, name) in enumerate(
        ((digit_id, label_name), (component_id, "detected H0 component"))
    ):
        ax = fig.add_subplot(1, 2, ax_i + 1)
        for lab in np.unique(labels):
            m = labels == lab
            if _is_pruned(lab):
                ax.scatter(Z2[m, 0], Z2[m, 1], s=10, c="lightgray", alpha=0.6)
            else:
                ax.scatter(Z2[m, 0], Z2[m, 1], s=10, alpha=0.8, label=str(lab))
        ax.set_title(f"coloured by {name}", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=8, markerscale=1.5, loc="best")
    _suptitle(fig, title, subtitle)
    fig.tight_layout()
    return fig


def plot_component_scatter_3d(
    Z3, digit_id, component_id, title="Detected structure (3D)", subtitle=None,
    label_name="ground-truth digit",
):
    """The 3D version of plot_component_scatter, over the top three principal directions."""
    Z3 = np.asarray(Z3)
    digit_id = np.asarray(digit_id)
    component_id = np.asarray(component_id)

    fig = Figure(figsize=(11, 5.2))
    for ax_i, (labels, name) in enumerate(
        ((digit_id, label_name), (component_id, "detected H0 component"))
    ):
        ax = fig.add_subplot(1, 2, ax_i + 1, projection="3d")
        for lab in np.unique(labels):
            m = labels == lab
            if _is_pruned(lab):
                ax.scatter(Z3[m, 0], Z3[m, 1], Z3[m, 2], s=8, c="lightgray", alpha=0.6)
            else:
                ax.scatter(Z3[m, 0], Z3[m, 1], Z3[m, 2], s=8, alpha=0.8, label=str(lab))
        ax.set_title(f"coloured by {name}", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.legend(fontsize=8, markerscale=1.5, loc="upper left")
    _suptitle(fig, title, subtitle)
    fig.tight_layout()
    return fig


def _suptitle(fig, title, subtitle=None):
    """The figure title, with an optional second line for the H0 and H1 counts."""
    if subtitle:
        fig.suptitle(f"{title}\n{subtitle}", fontsize=13)
    else:
        fig.suptitle(title, fontsize=15)


def _cell_drawer(image_shape, to_image=None, draw=None):
    """
    How a single observation is drawn into one cell of a strip.

    By default as an image. Data that are not images, such as the mocap poses, pass their own
    `draw(ax, vector)` instead.
    """
    if draw is not None:
        return draw
    to_image = to_image or (lambda v: np.asarray(v).reshape(image_shape))

    def draw_image(ax, vector):
        ax.imshow(to_image(vector), cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="nearest")
    return draw_image


def plot_recovery_strips(
    components, image_shape, to_image=None, title=None,
    empty_note="Nothing to show.", draw=None, cell_size=(1.15, 1.15),
    show_captions=True, row_gap=0.15, component_gap=0.15,
):
    """
    Two stacked rows per component: the ground-truth frame on top, the model below it.

    `row_gap` is the vertical space between those two rows and `component_gap` the space between
    components. `show_captions` turns the per-column value titles on and off.
    """
    draw_cell = _cell_drawer(image_shape, to_image, draw)
    if not components:
        fig = Figure(figsize=(5, 1.6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, empty_note, ha="center", va="center",
                fontsize=13, color="gray")
        ax.axis("off")
        return fig

    def captions(c):
        return c.get("labels") or [f"{a:.0f}°" for a in c["angles"]]

    n = len(components)
    max_cols = max(len(c["angles"]) for c in components)
    # the title band is a fixed height in inches, so that a figure with many
    # components does not get a proportionally huge gap under its title
    title_band = 0.4 if title else 0.05
    height = 2 * cell_size[1] * n + title_band
    fig = Figure(figsize=(cell_size[0] * max_cols, height))

    outer = fig.add_gridspec(n, 1, hspace=component_gap, left=0.09, right=0.99,
                             top=1.0 - title_band / height, bottom=0.01)
    for i, c in enumerate(components):
        inner = outer[i].subgridspec(2, max_cols, hspace=row_gap, wspace=0.15)
        rows = ((c["true_imgs"], c["label"], True),
                (c["recon_imgs"], "model", False))
        for r_off, (imgs, row_label, is_top) in enumerate(rows):
            for col in range(len(c["angles"])):
                ax = fig.add_subplot(inner[r_off, col])
                draw_cell(ax, imgs[col])
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=9)
                if is_top and show_captions:
                    ax.set_title(captions(c)[col], fontsize=8)
    if title:
        fig.suptitle(title, fontsize=13, y=1.0 - 0.16 / height)
    return fig


def plot_reconstruction_strips(strips, image_shape, to_image=None, title=None,
                               empty_note="Nothing to show.", draw=None,
                               cell_size=(1.15, 1.35)):
    """
    One row of reconstructions per detected component, walking along its curve.

    Each `strips` entry is (label, images (K, D), angles (K,) or None). K may differ per row, for
    instance 8 frames for a full loop against 4 for a half arc, and the rows are left-aligned in a
    shared grid.
    """
    draw_cell = _cell_drawer(image_shape, to_image, draw)
    if not strips:
        fig = Figure(figsize=(5, 1.6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, empty_note, ha="center", va="center",
                fontsize=13, color="gray")
        ax.axis("off")
        return fig

    n_rows = len(strips)
    max_cols = max(len(imgs) for _, imgs, _ in strips)
    fig = Figure(figsize=(cell_size[0] * max_cols, cell_size[1] * n_rows + 0.5))
    for r, (label, imgs, angs) in enumerate(strips):
        for c in range(len(imgs)):
            ax = fig.add_subplot(n_rows, max_cols, r * max_cols + c + 1)
            draw_cell(ax, imgs[c])
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
            if angs is not None:
                ax.set_title(f"{angs[c]:.0f}°", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_factor_residual(
    true, pred, periodic=True, factor_label="angle (deg)",
    title="Factor recovery residual"
):
    """
    The signed residual after alignment, plotted against the true factor.

    A periodic factor is wrapped to (-180, 180].
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
    """The mean of y per bin over the range of x. Returns the bin centres and the means."""
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
    """The three reconstruction errors of M3, plotted against the factor."""
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
    ax.set_xlabel(factor_label, fontsize=13)
    ax.set_ylabel("pixel RMSE (binned mean)", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.legend(fontsize=11)
    fig.tight_layout()
    return fig


def plot_image_strip(rows, image_shape, col_labels=None, to_image=None, title=None):
    """
    A grid of images, given as a list of (row_label, vectors (K, D0)) rows.

    'to_image' maps one vector to a 2D image and defaults to a plain reshape. The brightness range
    is fixed to [0, 1], so that the cells stay comparable across the whole strip.
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
    # equal, small gaps between the paired rows and the columns
    fig.subplots_adjust(hspace=0.12, wspace=0.12)
    return fig

