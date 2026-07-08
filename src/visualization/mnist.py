"""
Visualizations for the MNIST rotation example.
Uses `visualize_spline` from `spline.py` as the basis for the PCA background view.
"""

import numpy as np

from .spline import visualize_spline


def render_samples_frame(fig, X, angles, digit):
    n_show = 20
    idx = np.linspace(0, len(X) - 1, n_show, dtype=int)
    cols, rows = 10, 2
    axes = fig.subplots(rows, cols)

    pixel_mean = X.mean(axis=0) if X is not None else 0

    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        img = (X[i] + pixel_mean).reshape(30, 30)
        ax.imshow(np.clip(img, 0.0, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{angles[i]:.0f}°", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"Samples - digit {digit}", fontsize=12)
    fig.tight_layout()


def draw_pca_background_layer(fig, ax, state, pca_data, angles, exp, pca_basis, spline_to_pixel_fn):
    is_3d = state["is_3d"]

    def spline_projected(tt):
        tt_arr = np.atleast_1d(tt)
        pts = np.stack([spline_to_pixel_fn(v) for v in tt_arr])
        proj = pts @ pca_basis
        return proj if np.ndim(tt) > 0 else proj[0]

    visualize_spline(
        ax=ax,
        data=pca_data,
        spline=spline_projected,
        t=state["t"] if state["mode"] == "spline" else 0,
        draw_points=True,
        colors=angles,
        colorbar=True,
    )

    if exp is not None:
        means_px = exp.reconstruct(exp.model.means)
        means_proj = means_px @ pca_basis
        cluster_color = "gold"

        if is_3d:
            ax.scatter(means_proj[:, 0], means_proj[:, 1], means_proj[:, 2],
                       c=cluster_color, s=70, zorder=15, edgecolors="black")
        else:
            ax.scatter(means_proj[:, 0], means_proj[:, 1],
                       c=cluster_color, s=70, zorder=15, edgecolors="black")

    ax.set_title("3D PCA Projection" if is_3d else "2D PCA Projection", fontsize=12)


def render_pca_frame(fig, state, pca_data, angles, exp, pca_basis, spline_to_pixel_fn):
    if state["is_3d"]:
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=state["elev"], azim=state["azim"])
    else:
        ax = fig.add_subplot(111)

    draw_pca_background_layer(fig, ax, state, pca_data, angles, exp, pca_basis, spline_to_pixel_fn)
    fig.tight_layout()


def render_spline_frame(fig, state, X, pca_data, angles, exp, pca_basis, spline, spline_to_pixel_fn):
    if spline is None:
        render_pca_frame(fig, state, pca_data, angles, exp, pca_basis, spline_to_pixel_fn)
        return

    t = state["t"]
    point = spline_to_pixel_fn(t)
    pixel_mean = X.mean(axis=0) if X is not None else 0
    img = np.clip((point + pixel_mean).reshape(30, 30), 0.0, 1.0)

    ax_img = fig.add_subplot(1, 2, 1)
    if state["is_3d"]:
        ax_pca = fig.add_subplot(1, 2, 2, projection='3d')
        ax_pca.view_init(elev=state["elev"], azim=state["azim"])
    else:
        ax_pca = fig.add_subplot(1, 2, 2)

    ax_img.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_img.axis("off")
    ax_img.set_title(f"Spline reconstruction\nt = {t:.3f}  (~{t*360:.0f}°)", fontsize=10)

    draw_pca_background_layer(fig, ax_pca, state, pca_data, angles, exp, pca_basis, spline_to_pixel_fn)
    fig.tight_layout()

