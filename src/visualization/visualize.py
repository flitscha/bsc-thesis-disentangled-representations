import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def visualize_data(data):
    N, D = data.shape
    if D == 2:
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], s=5)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()
    elif D == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5)
        plt.tight_layout()
        plt.show()
    else:
        print("visualisation of dimension higher than 3 is not supported yet.")


def visualize_embedded_data(data, projection_matrix):
    projected_data = data @ projection_matrix
    N, D = projected_data.shape
    visualize_data(projected_data)


def plot_ellipse(ax, mean, cov, prior):
    # get tha angle of the ellipse, by calculating the eigenvalues of cov
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    vals = np.clip(vals, 1e-8, None)  # avoid negative eigenvalues
    width, height = 4 * np.sqrt(vals)

    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        color="orange",
        alpha=min(0.6, 0.2 + prior * 3),  # scale visibility by prior
    )

    ax.add_patch(ell)


def plot_line(ax, mean, cov, prior, length=0.5):
    vals, vecs = np.linalg.eigh(cov)

    # use the eigenvector with the biggest eigenvalue as direction
    idx = np.argmax(vals)
    direction = vecs[:, idx]

    # fixed length
    direction = direction * (length / 2.0)

    p1 = mean - direction
    p2 = mean + direction

    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color="orange",
        linewidth=3.0,
        alpha=min(0.8, 0.3 + prior * 2)
    )


def visualize_gmm_2d(
    ax, data, means, covariances, priors, draw_points=True,
    visualisation_mode="ellipsoid", draw_means=True
):
    """
    Plot 2D data and Gaussian components as ellipses.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if draw_points:
        ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.8, label="Data")

    if draw_means:
        ax.scatter(means[:, 0], means[:, 1], s=20, c="red", label="Means")

    # set limits
    x_min, y_min = data.min(axis=0)
    x_max, y_max = data.max(axis=0)
    padding = 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect("equal")

    # draw the lines to visualize the covariances
    plot_functions = {
        "ellipsoid": plot_ellipse,
        "line": plot_line,
        "plane": plot_line,  # just draw the line, since we are in a 2d-plot
        "none": lambda ax, mean, cov, prior: None,
    }

    plot_function = plot_functions[visualisation_mode]

    for mean, cov, prior in zip(means, covariances, priors):
        plot_function(ax, mean, cov, prior)


def plot_ellipsoid(ax, mean, cov, resolution=20, color="orange", alpha=0.2):
    """
    Plot a 3D Gaussian ellipsoid based on mean and covariance

    Parameters
    ----------
    ax : matplotlib 3D axis
    mean : shape (3,)
        Center of the ellipsoid
    cov : shape (3,3)
        Covariance matrix
    resolution : int
        Number of grid points for sphere parameterization
    """

    # Eigen decomposition
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-10, None)
    radii = 2.0 * np.sqrt(vals)

    # Parametric sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([x, y, z], axis=0).reshape(3, -1)

    # make the sphere to an ellipsoid by scaling and rotating it
    ellipsoid = radii[:, np.newaxis] * sphere
    ellipsoid = vecs @ ellipsoid
    ellipsoid = ellipsoid + mean[:, np.newaxis]

    x_e = ellipsoid[0].reshape(resolution, resolution)
    y_e = ellipsoid[1].reshape(resolution, resolution)
    z_e = ellipsoid[2].reshape(resolution, resolution)

    ax.plot_surface(x_e, y_e, z_e, color=color, alpha=alpha, linewidth=0)


def plot_plane(ax, mean, cov, size=3.0, color="orange", alpha=0.2):
    vals, vecs = np.linalg.eigh(cov)

    # use the two directions with the largest eigenvalues
    idx = np.argsort(vals)[::-1]
    vec1 = vecs[:, idx[0]]
    vec2 = vecs[:, idx[1]]

    # scale with fixed size
    vec1 = vec1 * size
    vec2 = vec2 * size

    # four vertices of the plane
    p1 = mean - vec1 - vec2
    p2 = mean - vec1 + vec2
    p3 = mean + vec1 - vec2
    p4 = mean + vec1 + vec2

    # plot the surface
    X = np.array([[p1[0], p2[0]],
                  [p3[0], p4[0]]])

    Y = np.array([[p1[1], p2[1]],
                  [p3[1], p4[1]]])

    Z = np.array([[p1[2], p2[2]],
                  [p3[2], p4[2]]])

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)


def plot_line_in_3d(ax, mean, cov, length=3.0, color="orange", alpha=0.8):
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argmax(vals)
    direction = vecs[:, idx]
    direction = direction * (length / 2.0)

    p1 = mean - direction
    p2 = mean + direction

    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        color="orange",
        linewidth=3.0,
        alpha=alpha
    )


def visualize_gmm_3d(
    ax, data, means, covariances, priors, draw_points=True,
    visualisation_mode="ellipsoid", draw_means=True
):
    if draw_points:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                   s=5, alpha=0.8, label="Data")

    if draw_means:
        ax.scatter(means[:, 0], means[:, 1], means[:, 2],
                   s=10, label="Means", c="red")

    # set limits
    x_min, y_min, z_min = data.min(axis=0)
    x_max, y_max, z_max = data.max(axis=0)
    padding = 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_zlim(z_min - padding, z_max + padding)
    ax.set_aspect("equal")

    plot_functions = {
        "ellipsoid": plot_ellipsoid,
        "line": plot_line_in_3d,
        "plane": plot_plane,
        "none": lambda ax, mean, cov: None,
    }

    plot_function = plot_functions[visualisation_mode]

    for mean, cov, prior in zip(means, covariances, priors):
        plot_function(ax, mean, cov)


def _visualize_gmm_higher_dimension(
    ax, data, means, covariances, priors, projection_matrix,
    draw_points=True, visualisation_mode="ellipsoid", draw_means=True
):
    """
    visualize data in higher dimensions by projecting it down using the projection_matrix
    """
    projected_data = data @ projection_matrix
    projected_means = means @ projection_matrix
    projected_covariances = [
        projection_matrix.T @ sigma @ projection_matrix
        for sigma in covariances
    ]

    N, D = projected_data.shape
    if D == 2:
        visualize_gmm_2d(
            ax,
            projected_data,
            projected_means,
            projected_covariances,
            priors,
            draw_points=draw_points,
            visualisation_mode=visualisation_mode,
            draw_means=draw_means
        )
    if D == 3:
        visualize_gmm_3d(
            ax,
            projected_data,
            projected_means,
            projected_covariances,
            priors,
            draw_points=draw_points,
            visualisation_mode=visualisation_mode,
            draw_means=draw_means
        )


def visualize_gmm(
    ax,
    data,
    means,
    covariances,
    priors,
    projection_matrix=None,
    draw_points=True,
    visualisation_mode="ellipsoid",
    draw_means=True
):
    if projection_matrix is not None:
        _visualize_gmm_higher_dimension(
            ax, data, means, covariances, priors, projection_matrix,
            draw_points, visualisation_mode, draw_means
        )
        return

    N, D = data.shape
    if D == 2:
        visualize_gmm_2d(
            ax, data, means, covariances, priors, draw_points=draw_points,
            visualisation_mode=visualisation_mode, draw_means=draw_means
        )
    if D == 3:
        visualize_gmm_3d(
            ax, data, means, covariances, priors, draw_points=draw_points,
            visualisation_mode=visualisation_mode, draw_means=draw_means
        )


def visualize_graph_on_mfa(
    ax,
    means,
    covariances,
    priors,
    edges=None,  # list of tuples [(i,j), ...]
    draw_nodes=True,
    visualisation_mode="line",
    node_color="red",
    edge_color="blue",
    edge_alpha=0.6
):
    """
    Visualize MFA components (means + covariances) and optionally a graph connecting them.

    Parameters
    ----------
    ax : matplotlib axis
    means : ndarray (N, D)
    covariances : list of ndarray (N, D, D)
    priors : ndarray (N,)
    edges : list of tuples (i,j) optional
        Each tuple is a connection between mean i and mean j
    """
    N, D = means.shape

    # set limits based on means
    mins = means.min(axis=0)
    maxs = means.max(axis=0)
    padding = 0.1

    if D == 2:
        ax.set_xlim(mins[0] - padding, maxs[0] + padding)
        ax.set_ylim(mins[1] - padding, maxs[1] + padding)
        ax.set_aspect("equal")

    elif D == 3:
        ax.set_xlim(mins[0] - padding, maxs[0] + padding)
        ax.set_ylim(mins[1] - padding, maxs[1] + padding)
        ax.set_zlim(mins[2] - padding, maxs[2] + padding)
        ax.set_box_aspect([1, 1, 1])

    # Draw MFA components
    if D == 2:
        for mean, cov, prior in zip(means, covariances, priors):
            if visualisation_mode == "line":
                plot_line(ax, mean, cov, prior)
            elif visualisation_mode == "ellipsoid":
                plot_ellipse(ax, mean, cov, prior)
        if draw_nodes:
            ax.scatter(means[:, 0], means[:, 1], s=20, c=node_color, zorder=3)

        # Draw edges
        if edges is not None:
            for i, j in edges:
                ax.plot(
                    [means[i, 0], means[j, 0]],
                    [means[i, 1], means[j, 1]],
                    color=edge_color,
                    alpha=edge_alpha,
                    linewidth=2
                )
    elif D == 3:
        for mean, cov, prior in zip(means, covariances, priors):
            if visualisation_mode == "line":
                plot_line_in_3d(ax, mean, cov)
            elif visualisation_mode == "ellipsoid":
                plot_ellipsoid(ax, mean, cov)
        if draw_nodes:
            ax.scatter(means[:, 0], means[:, 1], means[:, 2],
                       s=20, c=node_color, zorder=3)

        # Draw edges
        if edges is not None:
            for i, j in edges:
                ax.plot(
                    [means[i, 0], means[j, 0]],
                    [means[i, 1], means[j, 1]],
                    [means[i, 2], means[j, 2]],
                    color=edge_color,
                    alpha=edge_alpha,
                    linewidth=2
                )


def visualize_traversal(ax, means, order):
    """
    Visualize traversal order by drawing indices at each node
    """
    N, D = means.shape

    # set limits based on means
    mins = means.min(axis=0)
    maxs = means.max(axis=0)
    padding = 0.1

    if D == 2:
        ax.set_xlim(mins[0] - padding, maxs[0] + padding)
        ax.set_ylim(mins[1] - padding, maxs[1] + padding)
        ax.set_aspect("equal")

    elif D == 3:
        ax.set_xlim(mins[0] - padding, maxs[0] + padding)
        ax.set_ylim(mins[1] - padding, maxs[1] + padding)
        ax.set_zlim(mins[2] - padding, maxs[2] + padding)
        ax.set_box_aspect([1, 1, 1])

    if order is None or len(order) == 0:
        return

    ordered_means = means[order]

    D = means.shape[1]

    if D == 2:
        ax.scatter(ordered_means[:, 0], ordered_means[:, 1],
                   c="black", s=30, zorder=5)

        ax.plot(
            ordered_means[:, 0],
            ordered_means[:, 1],
            color="gray",
            linewidth=1.5,
            alpha=0.7,
            zorder=4
        )

        for idx, (x, y) in enumerate(ordered_means):
            ax.text(
                x+0.04, y,
                str(idx),
                fontsize=9,
                color="blue",
                zorder=6
            )

    elif D == 3:
        ax.scatter(ordered_means[:, 0], ordered_means[:, 1], ordered_means[:, 2],
                   c="black", s=30, zorder=5)

        ax.plot(
            ordered_means[:, 0],
            ordered_means[:, 1],
            ordered_means[:, 2],
            color="gray",
            linewidth=1.5,
            alpha=0.7,
            zorder=4
        )

        for idx, (x, y, z) in enumerate(ordered_means):
            ax.text(
                x+0.04, y, z,
                str(idx),
                fontsize=9,
                color="blue",
                zorder=6
            )


def visualize_spline(
    ax,
    data,
    spline,
    t,
    draw_points=True,
    colors=None,
    cmap="hsv",
    colorbar=False,
):
    """Visualisiert den berechneten Spline-Pfad bis zum Zeitpunkt t in 2D oder 3D.

    Sorgt für ein stabiles Achsenverhältnis (Aspect Ratio) ohne Springen oder
    Verzerren.
    """
    # Bestimme die Zieldimension (2D oder 3D) anhand des Matplotlib-Achsenobjekts
    is_3d = hasattr(ax, "get_zlim")
    D = 3 if is_3d else 2

    # Achsenbegrenzungen starr anhand der benötigten Dimensionen fixieren
    # Wenn data 3D ist, wir aber 2D plotten, ignorieren wir die 3. Spalte
    plot_data = data[:, :D]
    mins = plot_data.min(axis=0)
    maxs = plot_data.max(axis=0)

    # Padding relativ zur Skalierung berechnen, um Verzerrungen zu vermeiden
    ranges = maxs - mins
    padding = 0.1 * (ranges if np.all(ranges > 0) else 1.0)

    # 1. Datenpunkte im Hintergrund zeichnen
    if draw_points:
        if colors is None:
            if not is_3d:
                ax.scatter(
                    plot_data[:, 0],
                    plot_data[:, 1],
                    c="grey",
                    alpha=0.2,
                    s=5,
                )
            else:
                ax.scatter(
                    plot_data[:, 0],
                    plot_data[:, 1],
                    plot_data[:, 2],
                    c="grey",
                    alpha=0.15,
                    s=4,
                )
        else:
            if not is_3d:
                scatter = ax.scatter(
                    plot_data[:, 0],
                    plot_data[:, 1],
                    c=colors,
                    cmap=cmap,
                    s=15,
                    alpha=0.5,
                    vmin=0,
                    vmax=360,
                )
            else:
                scatter = ax.scatter(
                    plot_data[:, 0],
                    plot_data[:, 1],
                    plot_data[:, 2],
                    c=colors,
                    cmap=cmap,
                    s=15,
                    alpha=0.5,
                    vmin=0,
                    vmax=360,
                )

            if colorbar:
                # Verhindert, dass mehrfach Colorbars an dasselbe Ax-Objekt gehängt werden
                if not hasattr(ax, "_has_colorbar"):
                    cbar = ax.figure.colorbar(
                        scatter,
                        ax=ax,
                        pad=0.04,
                        shrink=0.75,
                    )
                    cbar.set_label("Rotation angle")
                    cbar.set_ticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
                    ax._has_colorbar = True

    # 2. Spline-Kurve berechnen (von 0 bis aktuellen Schieberegler-Wert t)
    if spline is not None and t > 0:
        t_vals = np.linspace(0, t, 200)

        # Sicheres Auswerten der Kurvenpunkte
        curve = spline(t_vals)[:, :D]
        current_p = spline(t)
        # Falls spline(t) ein 1D-Array zurückgibt, stelle 1D sicher
        if current_p.ndim > 1:
            current_p = current_p[0]
        current_p = current_p[:D]

        if not is_3d:
            # Kurve zeichnen
            ax.plot(
                curve[:, 0], curve[:, 1], c="blue", linewidth=2.5, zorder=10
            )
            # Aktuellen Punkt (Kopf der Kurve) markieren
            ax.scatter(
                [current_p[0]],
                [current_p[1]],
                s=180,
                c="crimson",
                marker="*",
                edgecolors="black",
                zorder=20,
            )
        else:
            # Kurve im 3D-Raum zeichnen
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                curve[:, 2],
                c="blue",
                linewidth=2.5,
                zorder=10,
            )
            # Aktuellen 3D-Punkt markieren
            ax.scatter(
                [current_p[0]],
                [current_p[1]],
                [current_p[2]],
                s=180,
                c="crimson",
                marker="*",
                edgecolors="black",
                zorder=20,
            )

    # 3. Aspect Ratio & Limits setzen
    if not is_3d:
        ax.set_xlim(mins[0] - padding[0], maxs[0] + padding[0])
        ax.set_ylim(mins[1] - padding[1], maxs[1] + padding[1])
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_xlim(mins[0] - padding[0], maxs[0] + padding[0])
        ax.set_ylim(mins[1] - padding[1], maxs[1] + padding[1])
        ax.set_zlim(mins[2] - padding[2], maxs[2] + padding[2])
        ax.set_box_aspect([1, 1, 1])




# -------------- Visualisation for MNIST demo ----------------------------
def render_samples_frame(fig, X, angles, digit):
    n_show = 20
    idx = np.linspace(0, len(X) - 1, n_show, dtype=int)
    cols, rows = 10, 2
    axes = fig.subplots(rows, cols)

    # Pixel-Mittelwert fuer Rekonstruktion (falls zentriert, hier direkt berechnet)
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

