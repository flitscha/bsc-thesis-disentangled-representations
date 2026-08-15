"""
Tab 3: Topology Explorer (TDA)
"""

import sys
import os
import time
import threading

import numpy as np
import dearpygui.dearpygui as dpg

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pipeline import ManifoldPipeline
from data.basic_manifolds import (
    curve_in_3d,
    half_circle,
    circle_and_half_circle,
    two_circles,
    linked_circles_3d,
    embed_data_to_dimension,
)
from visualization.geometry import plot_transform


DPI = 100
MIN_TEX = 250
MAX_TEX = 1600
RESIZE_THRESHOLD = 25
ROTATE_SENSITIVITY = 0.4
RENDER_INTERVAL = 0.033

# fn: generator function, has_labels: supports return_labels=True (ground truth, for later evaluation)
DATASETS = {
    "loop_3d": {"fn": curve_in_3d, "has_labels": False},
    "half_circle": {"fn": half_circle, "has_labels": False},
    "circle_and_half_circle": {"fn": circle_and_half_circle, "has_labels": True},
    "two_circles": {"fn": two_circles, "has_labels": True},
    "linked_circles_3d": {"fn": linked_circles_3d, "has_labels": True},
}

VIS_MODES = ("data", "graph", "spline", "persistence")
COMPONENT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
DIM_COLORS = {0: "tab:blue", 1: "tab:red"}


class TopologyDemoTab:
    _instance_counter = 0

    def __init__(self):
        TopologyDemoTab._instance_counter += 1
        uid = TopologyDemoTab._instance_counter
        self._texture_tag_base = f"topology_plot_texture_{uid}"
        self._texture_counter = 0
        self.texture_tag = None
        self.texture_registry_tag = f"topology_texture_registry_{uid}"
        self.image_tag = f"topology_plot_image_{uid}"
        self.plot_panel_tag = f"topology_plot_panel_{uid}"
        self.curve_controls_tag = f"{self.plot_panel_tag}_curve_controls_{uid}"
        self.curve_selector_tag = f"{self.plot_panel_tag}_curve_selector_{uid}"

        self.raw_data = None       # (N, D) point cloud, ground-truth data
        self.projection = None     # map back to the intrinsic coordinates, if embedded
        self.pipe = None           # fitted ManifoldPipeline
        self.topology = None       # result dict from pipe.detect()
        self.selected_curve = 0    # index into self.topology["curves"]
        self.t = 0.0               # position along the selected curve

        self.tex_w, self.tex_h = 700, 700
        self.azim = -60.0
        self.elev = 30.0
        self._dragging = False
        self._last_mouse_pos = None
        self._last_render_time = 0.0
        self._is_rendering = False
        self._render_requested = False

    def build_tab_ui(self):
        with dpg.group(horizontal=True):
            self._build_settings_panel()
            self._build_plot_panel()

        self._register_mouse_handlers()

    # ------------------ Setup: Settings ---------------------------
    def _build_settings_panel(self):
        _width = 150
        with dpg.child_window(width=400, autosize_y=True):
            dpg.add_text(
                "Fits an MFA, builds a tangent-aware distance from it, and "
                "detects connected components and loops with persistent homology "
                "(TDA). Unlike the traversal baseline, TDA needs only the "
                "distance graph, so there is a single k here.",
                wrap=380, color=[150, 150, 150],
            )
            dpg.add_separator()

            dpg.add_text("Data")
            self.data_type = dpg.add_combo(
                list(DATASETS.keys()),
                default_value="two_circles", label="Dataset", width=_width,
            )
            self.num_points = dpg.add_input_int(
                label="# data points", default_value=300, min_value=3, width=_width,
            )
            self.embed_dim = dpg.add_input_int(
                label="Embed into dimension (0 = off)", default_value=0, min_value=0, width=_width,
            )

            dpg.add_separator()
            dpg.add_text("MFA model")
            self.num_components = dpg.add_input_int(
                label="# components", default_value=25, min_value=3, width=_width,
            )
            self.pca_dim = dpg.add_input_int(
                label="PCA dim (0 = off)", default_value=0, min_value=0, width=_width,
            )

            dpg.add_separator()
            dpg.add_text("Distance metric")
            self.lambda_aniso = dpg.add_input_float(
                label="off-manifold penalty", default_value=30.0,
                min_value=0.0, width=_width,
            )
            with dpg.tooltip(self.lambda_aniso):
                dpg.add_text(
                    "Penalty lambda for moving off the tangent space. 0 = plain "
                    "Euclidean, larger stretches the normal directions more.",
                    wrap=260,
                )
            self.k_distance = dpg.add_input_int(
                label="k - distance graph", default_value=5, min_value=1, width=_width,
            )
            with dpg.tooltip(self.k_distance):
                dpg.add_text(
                    "Neighbors of the k-NN graph whose shortest paths give the "
                    "geodesic distance.",
                    wrap=260,
                )

            dpg.add_separator()
            dpg.add_text("TDA and interpolation")
            self.h0_factor = dpg.add_input_float(
                label="H0 threshold (0 = auto)", default_value=0.0,
                min_value=0.0, step=0.1, format="%.2f", width=_width,
            )
            with dpg.tooltip(self.h0_factor):
                dpg.add_text(
                    "How persistent a connected component has to be to count. "
                    "0 uses the automatic largest-gap rule; a value > 0 replaces "
                    "it by an explicit threshold at that multiple of the median "
                    "H0 merge scale, which keeps it free of the data scale.",
                    wrap=260,
                )
            self.h1_factor = dpg.add_input_float(
                label="H1 threshold (0 = auto)", default_value=0.0,
                min_value=0.0, step=0.1, format="%.2f", width=_width,
            )
            with dpg.tooltip(self.h1_factor):
                dpg.add_text(
                    "How persistent a loop has to be to count. 0 uses the "
                    "automatic prominence rule (death >= 1.8 x birth); a value "
                    "> 0 replaces it by an explicit threshold at that multiple "
                    "of the median H0 merge scale. Raise it to suppress "
                    "spurious loops.",
                    wrap=260,
                )
            self.interp_weight = dpg.add_input_float(
                label="tangent weight", default_value=3.0, min_value=0.0,
                step=0.5, width=_width,
            )
            with dpg.tooltip(self.interp_weight):
                dpg.add_text(
                    "Strength of the soft chart-tangent alignment of the "
                    "spline. 0 = pure minimal-curvature interpolation.",
                    wrap=260,
                )

            dpg.add_separator()
            dpg.add_button(label="Train", callback=self._on_train, width=-1)

            dpg.add_separator()
            dpg.add_text("Render mode")
            self.visualisation_mode = dpg.add_radio_button(
                VIS_MODES, default_value="data", callback=self._on_viz_change,
            )

            dpg.add_separator()
            dpg.add_text("Component to traverse")
            with dpg.group(tag=self.curve_controls_tag):
                dpg.add_text("(train first)", tag=f"{self.curve_controls_tag}_placeholder")

            dpg.add_separator()
            dpg.add_text("Spline parameter t")
            self.t_display = dpg.add_text("t = 0.000")
            self.t_slider = dpg.add_slider_float(
                default_value=0.0, min_value=0.0, max_value=1.0,
                width=-1, callback=self._on_slider_change,
            )

            dpg.add_separator()
            self.status_text = dpg.add_text("No training has been conducted yet.")

    # -------------------- Setup: Plot-Panel + Texture -----------------------
    def _build_plot_panel(self):
        self.texture_tag = f"{self._texture_tag_base}_{self._texture_counter}"
        self._texture_counter += 1

        with dpg.child_window(tag=self.plot_panel_tag, autosize_y=True):
            blank = [0.12, 0.12, 0.14, 1.0] * (self.tex_w * self.tex_h)
            with dpg.texture_registry(tag=self.texture_registry_tag):
                dpg.add_dynamic_texture(
                    width=self.tex_w, height=self.tex_h,
                    default_value=blank, tag=self.texture_tag,
                )
            dpg.add_image(
                self.texture_tag, tag=self.image_tag,
                width=self.tex_w, height=self.tex_h,
            )

        with dpg.item_handler_registry(tag=f"{self.plot_panel_tag}_resize_reg"):
            dpg.add_item_resize_handler(callback=self._on_panel_resize)
        dpg.bind_item_handler_registry(self.plot_panel_tag, f"{self.plot_panel_tag}_resize_reg")

    def _register_mouse_handlers(self):
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_down)
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_up)

    # ------------------------- Training -----------------------------
    def _on_train(self):
        dpg.set_value(self.status_text, "Training...")
        threading.Thread(target=self._run_training_thread, daemon=True).start()

    def _run_training_thread(self):
        key = dpg.get_value(self.data_type)
        n = dpg.get_value(self.num_points)
        spec = DATASETS[key]

        if spec["has_labels"]:
            data, _ = spec["fn"](n=n, return_labels=True)
        else:
            data = spec["fn"](n=n)

        embed_dim = dpg.get_value(self.embed_dim)
        self.projection = None
        if embed_dim and embed_dim > data.shape[1]:
            # keep the embedding map: it takes the plots back to the intrinsic
            # coordinates instead of showing three arbitrary ambient axes
            data, self.projection = embed_data_to_dimension(data, embed_dim)

        self.raw_data = data

        self.pipe = ManifoldPipeline(
            n_components=dpg.get_value(self.num_components),
            latent_dim=1, cov_type="mfa", shared=False,
            pca_dim=dpg.get_value(self.pca_dim) or None,
            lambda_aniso=dpg.get_value(self.lambda_aniso),
            n_neighbors=dpg.get_value(self.k_distance),
            detection="tda",
            interp_tangent_weight=dpg.get_value(self.interp_weight),
        )
        self.pipe.fit(data)

        # the automatic rules run first: the panel's thresholds are multiples of
        # the barcode's own merge scale, which only a finished detection knows
        self.pipe.detect()
        self.pipe.apply_persistence_thresholds(
            h0_factor=dpg.get_value(self.h0_factor),
            h1_factor=dpg.get_value(self.h1_factor),
        )
        self.topology = self.pipe.result()

        self.selected_curve = 0
        self._rebuild_curve_selector()

        n_components = len(self._component_ids())
        n_loops = len(self.topology["loops"])
        dpg.set_value(
            self.status_text,
            f"objective: {self.pipe.obj:.4f} | components: {n_components} | loops: {n_loops}",
        )
        dpg.set_value(self.visualisation_mode, "spline" if self.topology["curves"] else "graph")
        self._request_async_plot()

    def _component_ids(self):
        """The detected component labels (-1 marks charts pruned as near-empty)."""
        labels = np.unique(self.topology["components"])
        return labels[labels >= 0]

    def _rebuild_curve_selector(self):
        dpg.delete_item(self.curve_controls_tag, children_only=True)
        curves = self.topology["curves"] if self.topology else []

        if not curves:
            dpg.add_text("No component found.", parent=self.curve_controls_tag)
            return

        labels = []
        for i, c in enumerate(curves):
            if c["type"] == "loop":
                labels.append(f"{i}: loop (persistence={c['persistence']:.3f})")
            else:
                labels.append(f"{i}: path (component {c['component']})")
        dpg.add_radio_button(
            labels, default_value=labels[0], tag=self.curve_selector_tag,
            parent=self.curve_controls_tag, callback=self._on_curve_change,
        )

    def _on_curve_change(self, sender, app_data):
        self.selected_curve = int(app_data.split(":")[0])
        self._request_async_plot()

    def _on_viz_change(self):
        self._request_async_plot()

    def _on_slider_change(self, sender, app_data):
        self.t = app_data
        dpg.set_value(self.t_display, f"t = {self.t:.3f}")
        if dpg.get_value(self.visualisation_mode) == "spline":
            self._request_async_plot()

    def _request_async_plot(self):
        if self._is_rendering:
            self._render_requested = True
            return
        self._is_rendering = True
        threading.Thread(target=self._async_render_worker, daemon=True).start()

    def _async_render_worker(self):
        while True:
            self._render_requested = False
            self._update_plot()
            if not self._render_requested:
                break
        self._is_rendering = False

    def _throttled_update_plot(self):
        now = time.time()
        if now - self._last_render_time > RENDER_INTERVAL:
            self._last_render_time = now
            self._update_plot()

    # ------------ mouse dragging for 3d plots --------------------
    def _is_3d(self):
        return self.raw_data is not None and self._plot_data().shape[1] >= 3

    def _on_mouse_down(self, sender, app_data):
        if not dpg.is_item_hovered(self.image_tag):
            return
        if not self._is_3d():
            return
        self._dragging = True
        self._last_mouse_pos = None

    def _on_mouse_move(self, sender, app_data):
        if not self._dragging:
            return
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            self._dragging = False
            self._last_mouse_pos = None
            return

        x, y = dpg.get_mouse_pos(local=False)
        if self._last_mouse_pos is None:
            self._last_mouse_pos = (x, y)
            return

        dx = x - self._last_mouse_pos[0]
        dy = y - self._last_mouse_pos[1]
        self._last_mouse_pos = (x, y)

        self.azim -= dx * ROTATE_SENSITIVITY
        self.elev = float(np.clip(self.elev + dy * ROTATE_SENSITIVITY, -89.0, 89.0))
        self._throttled_update_plot()

    def _on_mouse_up(self, sender, app_data):
        self._dragging = False
        self._last_mouse_pos = None

    # -------------------- Texture management -------------------------
    def _swap_texture(self, width, height, data=None):
        new_tag = f"{self._texture_tag_base}_{self._texture_counter}"
        self._texture_counter += 1

        if data is None:
            data = [0.12, 0.12, 0.14, 1.0] * (width * height)

        dpg.add_dynamic_texture(
            width=width, height=height, default_value=data,
            tag=new_tag, parent=self.texture_registry_tag,
        )
        dpg.configure_item(self.image_tag, texture_tag=new_tag, width=width, height=height)

        old_tag = self.texture_tag
        self.texture_tag = new_tag
        self.tex_w, self.tex_h = width, height

        if old_tag is not None and dpg.does_item_exist(old_tag):
            dpg.delete_item(old_tag)

    def _on_panel_resize(self, sender, app_data):
        w, h = dpg.get_item_rect_size(self.plot_panel_tag)
        if w <= 0 or h <= 0:
            return
        new_w = int(np.clip(w - 15, MIN_TEX, MAX_TEX))
        new_h = int(np.clip(h - 15, MIN_TEX, MAX_TEX))

        if abs(new_w - self.tex_w) < RESIZE_THRESHOLD and abs(new_h - self.tex_h) < RESIZE_THRESHOLD:
            return

        self._swap_texture(new_w, new_h)
        self._update_plot()

    # ------------------- Plot space ------------------------
    def _plot_transform(self):
        """Affine map from the model's space to the plotted one (see geometry)."""
        pre = self.pipe.pre if self.pipe is not None else None
        return plot_transform(pre, self.projection, self.raw_data.shape[1])

    def _plot_points(self, points):
        """Map model-space points (means, spline samples) into the plot."""
        M, b = self._plot_transform()
        return np.asarray(points) @ M + b

    def _plot_data(self):
        """The observations in the plotted coordinates."""
        if self.projection is None:
            return self.raw_data
        return self.raw_data @ self.projection

    # ------------------- Plotting ------------------------
    def _update_plot(self):
        if self.raw_data is None:
            return

        mode = dpg.get_value(self.visualisation_mode)
        is_3d = self._is_3d() and mode != "persistence"
        figsize = (self.tex_w / DPI, self.tex_h / DPI)

        fig = Figure(figsize=figsize, dpi=DPI)
        ax = fig.add_subplot(111, projection="3d" if is_3d else None)
        if is_3d:
            ax.view_init(elev=self.elev, azim=self.azim)

        if mode == "data":
            self._draw_data(ax, is_3d)
        elif mode == "graph":
            self._draw_graph(ax, is_3d)
        elif mode == "spline":
            self._draw_spline(ax, is_3d)
        elif mode == "persistence":
            self._draw_persistence(ax)

        self._render_to_texture(fig)

    def _scatter(self, ax, points, is_3d, **kwargs):
        coords = [points[:, 0], points[:, 1]] + ([points[:, 2]] if is_3d else [])
        ax.scatter(*coords, **kwargs)

    def _line(self, ax, points, is_3d, **kwargs):
        coords = [points[:, 0], points[:, 1]] + ([points[:, 2]] if is_3d else [])
        ax.plot(*coords, **kwargs)

    def _draw_data(self, ax, is_3d):
        self._scatter(ax, self._plot_data(), is_3d, s=8, c="tab:blue")
        ax.set_title(dpg.get_value(self.data_type))

    def _draw_graph(self, ax, is_3d):
        if self.topology is None:
            ax.set_title("train first")
            return

        means = self._plot_points(self.pipe.model.means)
        labels = self.topology["components"]

        for c in self._component_ids():
            mask = labels == c
            color = COMPONENT_COLORS[int(c) % len(COMPONENT_COLORS)]
            self._scatter(ax, means[mask], is_3d, s=20, c=color, label=f"component {int(c)}")

        for curve in self.topology["curves"]:
            path_points = means[curve["order"]]
            style = "-" if curve["type"] == "loop" else "--"
            self._line(ax, path_points, is_3d, c="black", linewidth=1.5, alpha=0.7, linestyle=style)

        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{len(self._component_ids())} components, "
                     f"{len(self.topology['curves'])} curves")

    def _draw_spline(self, ax, is_3d):
        """The selected curve, traversed up to t (the star marks that point)."""
        curves = self.topology["curves"] if self.topology else []
        if not curves:
            ax.set_title("no component to interpolate")
            return

        self._scatter(ax, self._plot_data(), is_3d, s=6, c="lightgray")

        curve = curves[self.selected_curve]
        spline = curve["spline"]
        full = self._plot_points(spline(np.linspace(0, 1, 300)))
        self._line(ax, full, is_3d, c="tab:red", linewidth=1.0, alpha=0.3)

        if self.t > 0:
            walked = self._plot_points(spline(np.linspace(0, self.t, 300)))
            self._line(ax, walked, is_3d, c="tab:red", linewidth=2.5)
        current = self._plot_points(spline(self.t))[None, :]
        self._scatter(ax, current, is_3d, s=180, c="crimson", marker="*",
                      edgecolors="black", zorder=20)

        means = self._plot_points(self.pipe.model.means[curve["order"]])
        self._scatter(ax, means, is_3d, s=25, c="black")

        ax.set_title(f"{self.selected_curve}: {curve['type']} spline, t = {self.t:.3f}")

    def _draw_persistence(self, ax):
        diagram = self.topology["diagram"] if self.topology else []
        if not diagram:
            ax.set_title("train first")
            return

        finite = [(dim, b, d) for dim, (b, d) in diagram if np.isfinite(d)]
        if not finite:
            ax.set_title("no finite bars")
            return

        max_val = max(d for _, _, d in finite) * 1.1
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5)

        for dim in (0, 1):
            pts = [(b, d) for (dd, b, d) in finite if dd == dim]
            if pts:
                births, deaths = zip(*pts)
                ax.scatter(births, deaths, c=DIM_COLORS[dim], label=f"H{dim}", s=25)

        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        ax.legend(loc="lower right")
        ax.set_title("Persistence diagram")

    def _render_to_texture(self, fig):
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.asarray(canvas.buffer_rgba(), dtype=np.float32) / 255.0

        if (w, h) != (self.tex_w, self.tex_h):
            self._swap_texture(w, h, data=buf.flatten())
            return

        dpg.set_value(self.texture_tag, buf.flatten())
