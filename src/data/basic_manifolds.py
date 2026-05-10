import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_rng(random_state):
    return np.random.default_rng(random_state)


# ── 1D Manifolds ──────────────────────────────────────────────────────────────

def line_in_2d(n=500, random_state=None):
    t = np.linspace(-3, 3, n)
    x = t
    y = np.sin(t)
    data = np.stack([x, y], axis=1)
    labels = {"t": t}          # one latent factor: position along line
    return data


def circle(n=500, random_state=None):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    data = np.stack([x, y], axis=1)
    labels = {"theta": theta}  # one latent factor: angle in [0, 2pi)
    return data


def swiss_roll(n=1000, random_state=None):
    rng = _make_rng(random_state)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x = t * np.cos(t)
    y = 21 * rng.random(n)
    z = t * np.sin(t)
    data = np.stack([x, y, z], axis=1)
    labels = {
        "t": t,                # unrolled position along the roll
        "height": y,           # height (independent second factor)
    }
    return data


# ── 2D Manifolds ──────────────────────────────────────────────────────────────

def torus(n=1000, R=15.0, r=5.0, random_state=None):
    """
    Torus parametrized by two angles (theta, phi).

    This is the key example for 2-factor disentanglement:
      theta : "small" circle (e.g. local rotation)
      phi   : "large" circle (e.g. global position)

    Both factors are periodic in [0, 2*pi) — the manifold is S1 x S1.
    Labels are the ground-truth latent coordinates.
    """
    rng = _make_rng(random_state)
    theta = 2 * np.pi * rng.random(n)
    phi   = 2 * np.pi * rng.random(n)

    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    data = np.stack([x, y, z], axis=1)
    labels = {
        "theta": theta,   # latent factor 1 — small circle
        "phi":   phi,     # latent factor 2 — large circle
    }
    return data


def torus_grid(n_theta=20, n_phi=20, R=15.0, r=5.0):
    """
    Torus on a regular grid of (theta, phi) values.
    Useful for demos and visualisation — every point is fully labeled.

    Returns
    -------
    data   : (n_theta * n_phi, 3)
    labels : dict with 'theta', 'phi' — shape (n_theta * n_phi,)
    grid   : dict with 'theta_vals', 'phi_vals' for reshaping back to grid
    """
    theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi_vals   = np.linspace(0, 2 * np.pi, n_phi,   endpoint=False)

    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals, indexing="ij")
    theta_flat = theta_grid.ravel()
    phi_flat   = phi_grid.ravel()

    x = (R + r * np.cos(theta_flat)) * np.cos(phi_flat)
    y = (R + r * np.cos(theta_flat)) * np.sin(phi_flat)
    z = r * np.sin(theta_flat)

    data = np.stack([x, y, z], axis=1)
    labels = {"theta": theta_flat, "phi": phi_flat}
    grid   = {"theta_vals": theta_vals, "phi_vals": phi_vals,
              "n_theta": n_theta, "n_phi": n_phi}
    return data, labels, grid


def curve_in_3d(n=1000, random_state=None):
    t = np.linspace(0.0, 6 * np.pi, n)
    x = 15 * np.cos(t)
    y = 15 * np.sin(2 * t)
    z = 15 * np.sin(3 * t)
    data = np.stack([x, y, z], axis=1)
    labels = {"t": t}
    return data


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_data_to_dimension(data, new_dimension, noise=0.005,
                             random_rotation=True, random_state=None):
    """
    Embed low-dimensional data into a higher-dimensional space.

    Parameters
    ----------
    data          : (N, D)
    new_dimension : int, must be >= D
    noise         : std of isotropic Gaussian noise added after embedding
    random_rotation : if True, random orthonormal projection; else zero-pad
    random_state  : for reproducibility

    Returns
    -------
    data_high          : (N, new_dimension)
    projection_matrix  : (new_dimension, D)  — W such that data_high ≈ data @ W.T
    """
    rng = _make_rng(random_state)
    N, D = data.shape

    if new_dimension < D:
        raise ValueError("new_dimension must be >= original dimension")

    if random_rotation:
        A = rng.normal(size=(new_dimension, D))
        Q, _ = np.linalg.qr(A)
        W = Q[:, :D]
    else:
        W = np.zeros((new_dimension, D))
        W[:D, :D] = np.eye(D)

    data_high = data @ W.T
    if noise > 0:
        data_high += rng.normal(scale=noise, size=data_high.shape)

    return data_high, W


# ── Label utilities ───────────────────────────────────────────────────────────

def make_label_mask(n, labeled_fraction=0.1, random_state=None):
    """
    Randomly select a fraction of points to be 'labeled'.
    Returns a boolean mask of shape (n,).

    Use this to simulate semi-supervised settings:
        data, labels = torus(n=1000)
        mask = make_label_mask(1000, labeled_fraction=0.1)
        # only data[mask] has known labels
    """
    rng = _make_rng(random_state)
    mask = np.zeros(n, dtype=bool)
    k = max(1, int(n * labeled_fraction))
    idx = rng.choice(n, size=k, replace=False)
    mask[idx] = True
    return mask


def assign_soft_labels_to_components(data, labels, posteriors, label_key):
    """
    Transfer point-level labels to Gaussian components via MFA posteriors.

    Each component gets a soft label = weighted average of the labels
    of all data points, weighted by p(component | point).

    Parameters
    ----------
    data       : (N, D) — not used directly, kept for clarity
    labels     : dict, labels[label_key] has shape (N,)
                 Use np.nan for unlabeled points.
    posteriors : (N, C) — p(component | point), rows sum to 1
                 (this is what VAMM gives you after training)
    label_key  : str, which label to transfer

    Returns
    -------
    component_labels : (C,) soft labels per component
                       np.nan if no labeled point contributed
    component_confidence : (C,) total posterior mass from labeled points
                           (how reliable the soft label is)
    """
    y = np.array(labels[label_key], dtype=float)   # (N,)
    labeled = ~np.isnan(y)                          # boolean mask

    if labeled.sum() == 0:
        raise ValueError(f"No labeled points found for label '{label_key}'")

    # only use labeled points
    y_labeled  = y[labeled]               # (N_labeled,)
    post_labeled = posteriors[labeled]    # (N_labeled, C)

    # total posterior mass per component from labeled points
    confidence = post_labeled.sum(axis=0)             # (C,)

    # weighted average label per component
    component_labels = (post_labeled * y_labeled[:, None]).sum(axis=0)
    nonzero = confidence > 1e-12
    component_labels[nonzero] /= confidence[nonzero]
    component_labels[~nonzero] = np.nan

    return component_labels, confidence

