"""
Visualizations for the MNIST rotation example.
Uses `visualize_spline` from `spline.py` as the basis for the PCA background view.
"""

import numpy as np

from .spline import visualize_spline


def _to_image(v, pixel_mean=None, pixel_std=None, shape=(30, 30)):
    """
    Invert the standardization (v * std + mean) to turn a standardized vector
    back into a displayable [0, 1] image.
    """
    v = np.asarray(v, dtype=float)
    if pixel_std is not None:
        v = v * np.asarray(pixel_std)
    if pixel_mean is not None:
        v = v + np.asarray(pixel_mean)
    return np.clip(v, 0.0, 1.0).reshape(shape)


def render_samples_frame(fig, X, angles, digit, pixel_mean=None, pixel_std=None):
    n_show = 20
    idx = np.linspace(0, len(X) - 1, n_show, dtype=int)
    cols, rows = 10, 2
    axes = fig.subplots(rows, cols)

    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        ax.imshow(_to_image(X[i], pixel_mean, pixel_std), cmap="gray",
                  vmin=0, vmax=1, interpolation="nearest")
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

    ax.set_title("3D PCA Projection" if is_3d else "2D PCA Projection", fontsize=18)  # war 12
    ax.tick_params(labelsize=13)  # Achsen-Zahlen vergrößern


def render_pca_frame(fig, state, pca_data, angles, exp, pca_basis, spline_to_pixel_fn):
    if state["is_3d"]:
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=state["elev"], azim=state["azim"])
    else:
        ax = fig.add_subplot(111)

    draw_pca_background_layer(fig, ax, state, pca_data, angles, exp, pca_basis, spline_to_pixel_fn)
    fig.tight_layout()


def render_spline_frame(fig, state, pca_data, angles, exp, pca_basis, spline, spline_to_pixel_fn,
                        pixel_mean=None, pixel_std=None):
    if spline is None:
        render_pca_frame(fig, state, pca_data, angles, exp, pca_basis, spline_to_pixel_fn)
        return

    t = state["t"]
    point = spline_to_pixel_fn(t)
    img = _to_image(point, pixel_mean, pixel_std)

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

