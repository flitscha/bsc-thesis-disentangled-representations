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

    fig.suptitle(f"Samples - digit(s) {digit}", fontsize=12)
    fig.tight_layout()


def render_persistence_frame(fig, diagram, curves=None, selected=None):
    """
    Persistence diagram (birth vs death) of the detected structure: H0 gives the
    connected components, H1 the loops. Infinite deaths are drawn on a line above
    the finite features.

    Each detected loop is annotated at its own H1 point with its component index
    (the same number as in the "component to traverse" selector); the loop
    matching `selected` is ringed, so it is clear which triangle is which loop.
    """
    ax = fig.add_subplot(111)
    if not diagram:
        ax.text(0.5, 0.5, "Train + Detect first to see the\npersistence diagram",
                ha="center", va="center", fontsize=13, color="gray")
        ax.axis("off")
        return

    pts = {0: [], 1: []}
    for dim, (b, d) in diagram:
        if dim in pts:
            pts[dim].append((b, d))
    finite = [d for _, (_, d) in diagram if np.isfinite(d)]
    top = (max(finite) if finite else 1.0) * 1.1
    inf_y = top * 1.05

    ax.plot([0, top], [0, top], "--", color="gray", lw=1)
    for dim, color, marker in ((0, "tab:blue", "o"), (1, "tab:red", "^")):
        arr = np.array(pts[dim]) if pts[dim] else np.empty((0, 2))
        if arr.size:
            fin = np.isfinite(arr[:, 1])
            ax.scatter(arr[fin, 0], arr[fin, 1], c=color, marker=marker,
                       s=45, label=f"H{dim}", alpha=0.8)
            if (~fin).any():
                ax.scatter(arr[~fin, 0], np.full((~fin).sum(), inf_y), c=color,
                           marker=marker, s=45, edgecolors="k")

    # tie each detected loop to its H1 point via the component index
    if curves:
        for j, c in enumerate(curves):
            if c.get("type") != "loop":
                continue
            b, d = c.get("birth"), c.get("death")
            if b is None or d is None:
                continue
            is_sel = (j == selected)
            if is_sel:
                ax.scatter([b], [d], s=280, facecolors="none",
                           edgecolors="limegreen", linewidths=2.6, zorder=20)
            ax.annotate(
                str(j), (b, d), textcoords="offset points", xytext=(7, 4),
                fontsize=14 if is_sel else 11,
                fontweight="bold", zorder=21,
                color="green" if is_sel else "black",
            )

    ax.axhline(inf_y, color="gray", lw=0.5, ls=":")
    ax.text(0, inf_y, " inf", va="bottom", ha="left", color="gray", fontsize=9)
    ax.set_xlabel("birth", fontsize=13)
    ax.set_ylabel("death", fontsize=13)
    ax.set_title("Persistence diagram (H0 = components, H1 = loops)", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    fig.tight_layout()


def draw_pca_background_layer(fig, ax, state, pca_data, angles, exp, pca_basis, spline_to_pixel_fn):
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
        means_proj = exp.reconstruct(exp.model.means) @ pca_basis
        ax.scatter(means_proj[:, 0], means_proj[:, 1], means_proj[:, 2],
                   c="gold", s=70, zorder=15, edgecolors="black")

    # faint outline of the other detected components (the selected one is drawn
    # bold by visualize_spline above), so several loops/arcs stay visible at once
    overlay = state.get("overlay_curves")
    selected = state.get("selected_component", 0)
    if overlay:
        for i, proj in enumerate(overlay):
            if i == selected or proj is None:
                continue
            ax.plot(proj[:, 0], proj[:, 1], proj[:, 2],
                    color="gray", lw=1.0, alpha=0.45, zorder=6)

    ax.set_title("3D PCA Projection", fontsize=18)
    ax.tick_params(labelsize=13)


def render_pca_frame(fig, state, pca_data, angles, exp, pca_basis, spline_to_pixel_fn):
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=state["elev"], azim=state["azim"])
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
    ax_pca = fig.add_subplot(1, 2, 2, projection='3d')
    ax_pca.view_init(elev=state["elev"], azim=state["azim"])

    ax_img.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_img.axis("off")
    ax_img.set_title(f"Spline reconstruction\nt = {t:.3f}  (~{t*360:.0f}°)", fontsize=10)

    draw_pca_background_layer(fig, ax_pca, state, pca_data, angles, exp, pca_basis, spline_to_pixel_fn)
    fig.tight_layout()

