"""
Render modes:
  Samples  — grid of training images (sanity check)
  PCA      — 2D projection, circle check
  Spline   — slider: generated image + position in PCA plot side by side
"""

import sys
import os
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data.mnist_rotation import make_rotation_dataset
from core.experiment import Experiment
from core.atlas import extract_tangent_frame, atlas_summary
from core.graph import compute_score_matrix, build_knn_graph, graph_diagnostics
from core.traversal import traverse_graph
from core.interpolation import build_closed_spline

matplotlib.use("TkAgg")


class MNISTDemo:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MNIST Rotation — Manifold Demo")
        self.root.resizable(True, True)

        # state
        self.X = None           # (N, 784) centered pixel space
        self.pixel_mean = None  # (784,) for display
        self.angles = None      # (N,)
        self.exp = None
        self.spline = None
        self.pca_basis = None   # (784, 2)  for the *visualisation* PCA plot
        self.pca_data = None    # (N, 2)    cached 2-D projection for plotting

        left = ttk.Frame(self.root, padding=10)
        right = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._build_controls(left)
        self._build_canvas(right)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _to_img(self, flat):
        """Centered flat vector (pixel space, 900) -> displayable (30, 30) in [0,1]."""
        img = (flat + self.pixel_mean).reshape(30, 30)
        img = np.clip(img, 0.0, 1.0)
        return img

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_controls(self, frame):
        r = 0

        def section(text):
            nonlocal r
            ttk.Separator(frame, orient="horizontal").grid(
                row=r, column=0, columnspan=2, sticky="ew", pady=(10, 2))
            r += 1
            ttk.Label(frame, text=text, font=("", 9, "bold")).grid(
                row=r, column=0, columnspan=2, sticky="w")
            r += 1

        def entry(label, var, width=8):
            nonlocal r
            ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w")
            ttk.Entry(frame, textvariable=var, width=width).grid(
                row=r, column=1, sticky="w")
            r += 1

        def btn(label, cmd):
            nonlocal r
            ttk.Button(frame, text=label, command=cmd).grid(
                row=r, column=0, columnspan=2, sticky="ew", pady=3)
            r += 1

        def status(init="—"):
            nonlocal r
            lbl = ttk.Label(frame, text=init, foreground="gray",
                            wraplength=220, justify="left")
            lbl.grid(row=r, column=0, columnspan=2, sticky="w")
            r += 1
            return lbl

        # ---- data --------------------------------------------------------
        section("Data")
        self.digit_var    = tk.IntVar(value=3)
        self.n_angles_var = tk.IntVar(value=360)
        self.n_images_var = tk.IntVar(value=1)
        entry("Digit (0–9):", self.digit_var)
        entry("Angles:", self.n_angles_var)
        entry("Source images:", self.n_images_var)
        btn("Generate data", self._generate_data)
        self.data_lbl = status()

        # ---- model -------------------------------------------------------
        section("Model")
        self.n_comp_var  = tk.IntVar(value=24)
        self.k_var       = tk.IntVar(value=2)
        self.pca_dim_var = tk.IntVar(value=0)   # 0 = disabled
        entry("# components:", self.n_comp_var)
        entry("k (graph):", self.k_var)
        entry("PCA dim (0=off):", self.pca_dim_var)
        btn("Train + build spline", self._train_threaded)
        self.train_lbl = status()

        # ---- render mode -------------------------------------------------
        section("Render mode")
        self.mode_var = tk.StringVar(value="samples")
        for val, txt in [("samples", "Samples"),
                         ("pca",     "PCA projection"),
                         ("spline",  "Spline + PCA (use slider)")]:
            ttk.Radiobutton(frame, text=txt, variable=self.mode_var,
                            value=val, command=self._render).grid(
                row=r, column=0, columnspan=2, sticky="w")
            r += 1

        # ---- slider ------------------------------------------------------
        section("Spline parameter t")
        self.t_var  = tk.DoubleVar(value=0.0)
        self.t_disp = ttk.Label(frame, text="t = 0.000  (~0°)")
        self.t_disp.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        ttk.Scale(frame, from_=0.0, to=1.0, orient="horizontal",
                  variable=self.t_var,
                  command=self._on_slider).grid(
            row=r, column=0, columnspan=2, sticky="ew")
        r += 1

        for child in frame.winfo_children():
            child.grid_configure(padx=4, pady=2)

    # ------------------------------------------------------------------
    # canvas
    # ------------------------------------------------------------------

    def _build_canvas(self, frame):
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._blank("Generate data to begin.")

    def _blank(self, msg=""):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        if msg:
            ax.set_title(msg)
        self.canvas.draw()

    # ------------------------------------------------------------------
    # data / train
    # ------------------------------------------------------------------

    def _generate_data(self):
        self.data_lbl.config(text="Loading…", foreground="orange")
        self.root.update()

        self.X, self.angles, self.pixel_mean = make_rotation_dataset(
            digit=self.digit_var.get(),
            n_angles=self.n_angles_var.get(),
            n_images=self.n_images_var.get(),
            center=True,
        )

        # 2-D PCA for the *visualisation* scatter plot (always in pixel space)
        _, _, Vt = np.linalg.svd(self.X, full_matrices=False)
        self.pca_basis = Vt[:2].T     # (784, 2)
        self.pca_data  = self.X @ self.pca_basis  # (N, 2)

        N = len(self.X)
        self.data_lbl.config(
            text=f"digit={self.digit_var.get()}, "
                 f"{self.n_images_var.get()} img × {self.n_angles_var.get()} "
                 f"rot = {N} pts",
            foreground="green")
        print(f"[data] X.shape={self.X.shape}")

        self.spline = None
        self.exp    = None
        self.mode_var.set("samples")
        self._render()

    def _train_threaded(self):
        if self.X is None:
            self.train_lbl.config(text="Generate data first.", foreground="red")
            return
        self.train_lbl.config(text="Training… (see terminal)", foreground="orange")
        self.root.update()
        threading.Thread(target=self._train, daemon=True).start()

    def _train(self):
        C       = self.n_comp_var.get()
        k       = self.k_var.get()
        pca_dim = self.pca_dim_var.get()   # 0 means disabled

        exp = Experiment(
            data_type="external",
            N=len(self.X),
            C=C, H=1,
            cov_type="mfa",
            shared=False,
            pca_dim=pca_dim if pca_dim > 0 else None,
        )
        exp.data = self.X.copy()   # Experiment.train() may overwrite self.data
        exp.train()
        self.exp = exp
        print(f"[train] obj={exp.obj:.4f}")

        means = exp.model.means
        covs  = exp.model.covariances

        tangents, variances, noise_var = extract_tangent_frame(covs, n_tangents=1)
        atlas_summary(means, tangents, variances, noise_var)

        score = compute_score_matrix(means, tangents, variances)
        graph = build_knn_graph(score, k=k)
        diag  = graph_diagnostics(graph)
        print(f"[graph] {diag}")

        order = traverse_graph(graph["adjacency"])

        # Build spline in whatever space the model lives in (pixel or PCA).
        # _spline_to_pixel() will back-project as needed.
        self.spline = build_closed_spline(means[order])
        print(f"[traversal] {len(order)}/{C}")

        pca_info = f", pca_dim={pca_dim}" if pca_dim > 0 else ""
        txt = (f"C={C}{pca_info}, obj={exp.obj:.3f} | "
               f"comps={diag['n_components']}, "
               f"cycle={diag['is_cycle_like']}, "
               f"traversal={len(order)}/{C}")
        self.root.after(0, lambda: self.train_lbl.config(
            text=txt, foreground="green"))
        self.root.after(0, lambda: self.mode_var.set("spline"))
        self.root.after(0, self._render)

    # ------------------------------------------------------------------
    # spline → pixel-space helper
    # ------------------------------------------------------------------

    def _spline_to_pixel(self, t: float) -> np.ndarray:
        """
        Evaluate spline at t and return a (784,) centred pixel-space vector.

        If PCA pre-projection was used, the spline lives in pca_dim-space and
        we back-project via exp.reconstruct().  Otherwise the spline lives
        directly in 784-D pixel space.
        """
        point = self.spline(t)   # (pca_dim,) or (784,)
        if self.exp is not None and self.exp.pca_components is not None:
            point = self.exp.reconstruct(point)   # -> (784,)  (mean-centred)
            # exp.reconstruct adds back pca_mean which is in centred pixel
            # space, so the result is still centred relative to pixel_mean.
        return point   # (784,) centred

    # ------------------------------------------------------------------
    # render dispatch
    # ------------------------------------------------------------------

    def _on_slider(self, val):
        t = float(val)
        self.t_disp.config(text=f"t = {t:.3f}  (~{t * 360:.0f}°)")
        if self.mode_var.get() == "spline":
            self._render_spline(t)

    def _render(self, *_):
        mode = self.mode_var.get()
        if mode == "samples":
            self._render_samples()
        elif mode == "pca":
            self._render_pca()
        elif mode == "spline":
            self._render_spline(self.t_var.get())

    # ------------------------------------------------------------------
    # render: samples
    # ------------------------------------------------------------------

    def _render_samples(self):
        if self.X is None:
            return
        self.fig.clear()
        n_show = 20
        idx    = np.linspace(0, len(self.X) - 1, n_show, dtype=int)
        cols, rows = 10, 2
        axes = np.array(self.fig.subplots(rows, cols)).reshape(rows, cols)

        for k, i in enumerate(idx):
            r, c = divmod(k, cols)
            axes[r, c].imshow(self._to_img(self.X[i]),
                              cmap="gray", vmin=0, vmax=1,
                              interpolation="nearest")
            axes[r, c].set_title(f"{self.angles[i]:.0f}°", fontsize=6)
            axes[r, c].axis("off")

        self.fig.suptitle(
            f"Samples — digit {self.digit_var.get()}, "
            f"{self.n_images_var.get()} source image(s)", fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # render: PCA scatter
    # ------------------------------------------------------------------

    def _render_pca(self):
        if self.X is None:
            return
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._draw_pca_background(ax)
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_pca_background(self, ax, highlight_point=None):
        """
        Draw the 2-D PCA scatter of the *pixel-space* data.

        Component means and spline are back-projected to pixel space before
        being projected onto the 2-D visualisation basis so everything lives
        in the same space regardless of whether PCA pre-projection was used
        during training.
        """
        sc = ax.scatter(self.pca_data[:, 0], self.pca_data[:, 1],
                        c=self.angles, cmap="hsv", s=15, alpha=0.5,
                        vmin=0, vmax=360)

        cbar = self.fig.colorbar(sc, ax=ax)
        cbar.set_label("angle (°)", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        if self.exp is not None:
            # back-project means to pixel space for the visualisation
            means_px = self.exp.reconstruct(self.exp.model.means)  # (C, 784)
            means_2d = means_px @ self.pca_basis                   # (C, 2)
            ax.scatter(means_2d[:, 0], means_2d[:, 1],
                       c=np.arange(len(means_2d)), cmap="plasma",
                       s=60, zorder=4,
                       edgecolors="k", linewidths=0.4)

        if self.spline is not None:
            t_fine  = np.linspace(0, 1, 500)
            # back-project each spline point to pixel space
            sp_px   = np.stack([self._spline_to_pixel(t) for t in t_fine])  # (500, 784)
            sp_2d   = sp_px @ self.pca_basis                                 # (500, 2)
            ax.plot(sp_2d[:, 0], sp_2d[:, 1],
                    color="crimson", lw=1.5, zorder=5, label="spline")

        if highlight_point is not None:
            ax.scatter(*highlight_point, s=180, color="crimson",
                       zorder=10, marker="*", label=f"t={self.t_var.get():.2f}")
            ax.legend(fontsize=8)

        ax.set_title("PCA projection", fontsize=15)
        ax.set_xlabel("PC 1", fontsize=13)
        ax.set_ylabel("PC 2", fontsize=13)

    # ------------------------------------------------------------------
    # render: spline + PCA side by side
    # ------------------------------------------------------------------

    def _render_spline(self, t):
        if self.spline is None:
            self._render_pca()
            return

        t     = float(t)
        point = self._spline_to_pixel(t)   # (784,) centred, pixel space
        img   = self._to_img(point)        # (28,28) in [0,1]
        pt_2d = point @ self.pca_basis     # (2,) position in vis-PCA

        self.fig.clear()
        ax_img = self.fig.add_subplot(1, 2, 1)
        ax_pca = self.fig.add_subplot(1, 2, 2)

        ax_img.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax_img.axis("off")
        ax_img.set_title(
            f"Spline reconstruction\nt = {t:.3f}  (~{t*360:.0f}°)",
            fontsize=10)

        self._draw_pca_background(ax_pca, highlight_point=pt_2d)

        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    MNISTDemo().run()
