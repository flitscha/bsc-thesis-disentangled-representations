"""
Visualizations for the motion capture demo.

The observations are poses, not images, so the frames that show data are stick
figures instead of `imshow`. Everything else (the 3D PCA view, the persistence
diagram) is identical to the other demos and reused from `visualization.mnist`.

A pose is drawn with a fixed oblique projection: the camera looks at the body
from the side, turned by `VIEW_AZIMUTH` degrees, so that the walking profile
stays readable while movements across the body (the swing of a waving arm) do
not collapse onto a single line.
"""

import numpy as np

from data.mocap import bone_edges, joint_names
from .mnist import draw_pca_background_layer, render_pca_frame

# The body-centred frame is x = right, y = up, z = forward.
VIEW_AZIMUTH = 25.0

# Left limbs, right limbs and the rest are drawn in different colours, which is
# what makes the two legs of a stick figure tellable apart at all.
LIMB_COLORS = {"l": "#1f77b4", "r": "#d62728", "": "#333333"}

# Fixed drawing box in metres, so that every stick figure of a figure is on the
# same scale. Measured on the recordings: the topmost joint (`head`, the base of
# the skull, not the crown) sits about 1.45 m above the floor, and no joint gets
# further than a metre from the root; the box leaves headroom on both counts.
VIEW_BOX = ((-0.85, 0.85), (0.0, 1.9))


def to_pose(vector, pose_mean=None, pose_std=None):
    """Turn a standardized data vector back into (J, 3) joint positions."""
    v = np.asarray(vector, dtype=float)
    if pose_std is not None:
        v = v * pose_std
    if pose_mean is not None:
        v = v + pose_mean
    return v.reshape(-1, 3)


def _project(pose):
    """(J, 3) body-centred positions -> (J, 2) screen coordinates."""
    angle = np.deg2rad(VIEW_AZIMUTH)
    horizontal = pose[:, 2] * np.cos(angle) + pose[:, 0] * np.sin(angle)
    return np.column_stack([horizontal, pose[:, 1]])


def _edge_colors():
    """One colour per bone, by the side of the body its child joint is on."""
    names = joint_names()
    colors = []
    for _, child in bone_edges():
        side = names[child][0]
        colors.append(LIMB_COLORS.get(side if side in "lr" else "", LIMB_COLORS[""]))
    return colors


def draw_pose(ax, pose, alpha=1.0, lw=1.8, box=VIEW_BOX):
    """Draw one stick figure into `ax` and set a fixed, equal-aspect box."""
    xy = _project(np.asarray(pose))
    for (parent, child), color in zip(bone_edges(), _edge_colors()):
        ax.plot(xy[[parent, child], 0], xy[[parent, child], 1],
                color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.set_xlim(*box[0])
    ax.set_ylim(*box[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_vector(ax, vector, pose_mean=None, pose_std=None, **kwargs):
    """draw_pose for a standardized data vector."""
    draw_pose(ax, to_pose(vector, pose_mean, pose_std), **kwargs)


def render_samples_frame(fig, X, captions, title, pose_mean=None, pose_std=None):
    """
    A grid of dataset poses spread evenly over the whole dataset, so every
    motion gets a share of the rows. `captions` holds one already formatted
    progress value per observation.
    """
    n_show = 20
    idx = np.linspace(0, len(X) - 1, n_show, dtype=int)
    cols, rows = 10, 2
    axes = fig.subplots(rows, cols)

    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        draw_vector(ax, X[i], pose_mean, pose_std, lw=1.2)
        ax.set_title(captions[i], fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()


def render_spline_frame(fig, state, pca_data, colors, exp, pca_basis, spline,
                        spline_to_pose_fn, pose_mean=None, pose_std=None):
    """
    The pose reconstructed at the current spline coordinate next to the PCA
    view. `state["factor_label"]` names the motion of the selected component,
    which is all the caption can say: t is the model's own coordinate and is
    only mapped to a physical value in the evaluation.
    """
    if spline is None:
        render_pca_frame(fig, state, pca_data, colors, exp, pca_basis,
                         spline_to_pose_fn)
        return

    t = state["t"]
    ax_pose = fig.add_subplot(1, 2, 1)
    ax_pca = fig.add_subplot(1, 2, 2, projection="3d")
    ax_pca.view_init(elev=state["elev"], azim=state["azim"])

    draw_vector(ax_pose, spline_to_pose_fn(t), pose_mean, pose_std, lw=2.5)
    motion = state.get("factor_label")
    ax_pose.set_title(f"Spline reconstruction\nt = {t:.3f}"
                      + (f"  ({motion})" if motion else ""), fontsize=10)

    draw_pca_background_layer(fig, ax_pca, state, pca_data, colors, exp,
                              pca_basis, spline_to_pose_fn)
    fig.tight_layout()
