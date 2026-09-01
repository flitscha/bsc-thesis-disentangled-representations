"""
The plots of the faces tab.

The 3D PCA view and the persistence diagram are the same as in the MNIST tab and are reused from
visualization/mnist.py. Only the frames showing images differ, because the faces are rendered at
their own resolution and their factor is not an angle, since a smile has no degrees.
"""

import numpy as np

from .mnist import to_image, draw_pca_background_layer, render_pca_frame


def render_samples_frame(fig, X, captions, title, pixel_mean=None,
                         pixel_std=None, image_shape=(64, 64)):
    """
    A grid of frames spread evenly over the dataset, so that every component gets a share.

    `captions` holds one already formatted factor value per observation, since the units differ
    from component to component.
    """
    n_show = 20
    idx = np.linspace(0, len(X) - 1, n_show, dtype=int)
    cols, rows = 10, 2
    axes = fig.subplots(rows, cols)

    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        ax.imshow(to_image(X[i], pixel_mean, pixel_std, image_shape), cmap="gray",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(captions[i], fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()


def render_spline_frame(fig, state, pca_data, colors, exp, pca_basis, spline,
                        spline_to_pixel_fn, pixel_mean=None, pixel_std=None,
                        image_shape=(64, 64)):
    """
    The face reconstructed at the current spline coordinate, next to the PCA view.

    `state["factor_label"]` names the factor of the selected component, which is all the caption
    can say: t is the model's own coordinate and only gets a physical value in the evaluation.
    """
    if spline is None:
        render_pca_frame(fig, state, pca_data, colors, exp, pca_basis,
                         spline_to_pixel_fn)
        return

    t = state["t"]
    img = to_image(spline_to_pixel_fn(t), pixel_mean, pixel_std, image_shape)

    ax_img = fig.add_subplot(1, 2, 1)
    ax_pca = fig.add_subplot(1, 2, 2, projection="3d")
    ax_pca.view_init(elev=state["elev"], azim=state["azim"])

    ax_img.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_img.axis("off")
    factor = state.get("factor_label")
    ax_img.set_title(f"Spline reconstruction\nt = {t:.3f}" +
                     (f"  ({factor})" if factor else ""), fontsize=10)

    draw_pca_background_layer(fig, ax_pca, state, pca_data, colors, exp,
                              pca_basis, spline_to_pixel_fn)
    fig.tight_layout()

