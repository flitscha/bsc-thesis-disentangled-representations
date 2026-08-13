import sys
import os
import queue
import threading
import numpy as np
import dearpygui.dearpygui as dpg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from data.faces import (
    make_multi_face_dataset, FaceRenderer, FACES, FACTOR_NAMES, FACTOR_UNITS,
)
from core.pipeline import ManifoldPipeline
from core.mfa import atlas_summary
from visualization.faces import render_samples_frame, render_spline_frame
from visualization.mnist import render_pca_frame, render_persistence_frame
from experiments.face_factor_eval import (
    run_evaluation as run_face_evaluation,
    _spec_label,
    _format_value,
)
from experiments.multi_factor_eval import _component_labels as _observation_components

DPI = 100
MIN_TEX = 250
MAX_TEX = 1600
RESIZE_THRESHOLD = 25
ROTATE_SENSITIVITY = 0.4
PLOT_THROTTLE_SEC = 0.05 # Throttling interval for mouse dragging (20 FPS)

# One table row per face of the catalogue; the preset fills the first three.
PRESET_FACTORS = ("yaw", "smile", "jaw_open")
DEFAULT_FACTORS = ("yaw", "smile", "jaw_open", "brow_raise", "pucker", "blink")


class FacesDemoTab:
    """
    Faces from the ICT morphable model (the catalogue 'data.faces.FACES'), each
    sweeping one generative factor (head rotation, smile, jaw, ...) over its full range.
    """

    def __init__(self):
        # State variables
        self.X = None
        self.pixel_mean = None
        self.pixel_std = None
        self.values = None # ground-truth factor value per frame
        self.colors = None # position within its own sweep, in [0, 1]
        self.captions = None # formatted factor value per frame
        self.component_gt = None # which spec (face) a frame comes from
        self.meta = None
        self.specs = None
        self.exp = None
        self.spline = None
        self.pca_basis = None
        self.pca_data = None
        self.renderer = None
        self.image_shape = None # resolution the current data was rendered at

        # Initial camera rotation (for 3D plots)
        self.azim = -60.0
        self.elev = 30.0
        self._dragging = False
        self._last_mouse_pos = None

        # Throttling control attributes
        self._throttle_timer = None
        self._pending_plot_request = False

        # Threading & Dynamic Texture Pipeline
        self.plot_queue = queue.Queue(maxsize=1)
        self.tex_w, self.tex_h = 800, 600

        self._texture_tag_base = "faces_plot_texture"
        self._texture_counter = 0
        self.texture_tag = None
        self.texture_registry_tag = "faces_texture_registry"

        self.image_tag = "faces_plot_image"
        self.plot_panel_tag = "faces_canvas_container"
        self.handler_registry_tag = "faces_handler_registry"
        self.window_handler_tag = "faces_window_handler"

        self.export_dialog_tag = "faces_export_dialog"
        self.export_status_lbl = None

        # Per-face data table widgets (include + factor + samples).
        self.row_include = {}
        self.row_factor = {}
        self.row_samples = {}
        self.eval_lbl = None
        self.data_label = "-"

        # Multi-component visualization state
        self.diagram = None # persistence diagram of the detection
        self.curves = None # detected curves (with splines)
        self.curve_projections = None # each curve sampled + projected to PCA space
        self.curve_faces = [] # face name per detected curve (majority vote)
        self.selected_component = 0
        self.component_var = None
        self.component_label_to_idx = {}

    def start_workers(self):
        threading.Thread(target=self._plot_worker, daemon=True).start()

    def build_tab_ui(self):
        dpg.add_texture_registry(tag=self.texture_registry_tag, show=False)

        with dpg.group(horizontal=True):
            self._build_settings_panel()
            self._build_plot_panel()

        with dpg.item_handler_registry(tag=self.window_handler_tag):
            dpg.add_item_resize_handler(callback=self._on_window_resize)
        dpg.bind_item_handler_registry(self.plot_panel_tag, self.window_handler_tag)

        self._register_mouse_handlers()

    def _build_settings_panel(self):
        with dpg.child_window(width=380, border=True):
            _width = 125

            dpg.add_text(
                "Rendered faces: every face is a separate component and sweeps "
                "one factor over its full range (rotation in degrees, "
                "expressions in blendshape units), so each face is an open arc. "
                "Detected with TDA (Sec. 5.6).",
                wrap=340, color=[150, 150, 150],
            )
            dpg.add_spacer(height=8)

            # Data Section: per-face table
            dpg.add_text("Data (faces and their factor)", color=[0, 255, 255])
            dpg.add_separator()
            dpg.add_button(
                label="Preset: 3 faces (yaw + smile + jaw)",
                callback=self._preset_three_faces, width=-1,
            )
            with dpg.group(horizontal=True):
                dpg.add_text("use", color=[150, 150, 150])
                dpg.add_text("face", color=[150, 150, 150])
                dpg.add_text("  factor", color=[150, 150, 150])
                dpg.add_text("      samples", color=[150, 150, 150])
            for row, face in enumerate(FACES):
                factor = (PRESET_FACTORS[row] if row < len(PRESET_FACTORS)
                          else DEFAULT_FACTORS[row])
                with dpg.group(horizontal=True):
                    self.row_include[row] = dpg.add_checkbox(
                        default_value=row < len(PRESET_FACTORS))
                    dpg.add_text(f" {face['name']} ")
                    self.row_factor[row] = dpg.add_combo(
                        list(FACTOR_NAMES), default_value=factor, width=100)
                    self.row_samples[row] = dpg.add_input_int(
                        default_value=120, width=62, step=0)

            dpg.add_spacer(height=6)
            self.image_size_in = dpg.add_input_int(
                label="Image size (px)", default_value=64, width=_width)
            with dpg.tooltip(self.image_size_in):
                dpg.add_text(
                    "Faces are rendered square at this resolution; the data "
                    "dimension is its square. Larger is slower to render.",
                    wrap=260)
            self.noise_in = dpg.add_input_float(
                label="Pixel noise (std)", default_value=0.0, min_value=0.0,
                step=0.01, format="%.3f", width=_width,
            )
            with dpg.tooltip(self.noise_in):
                dpg.add_text(
                    "Std of per-pixel Gaussian noise. Keep it well below the "
                    "MNIST setting: an expression moves far fewer pixels than a "
                    "rotating digit.", wrap=260)
            self.seed_in = dpg.add_input_int(
                label="RNG seed (-1 = random)", default_value=0, width=_width)
            with dpg.tooltip(self.seed_in):
                dpg.add_text(
                    "Seeds the PCA random rotation, the MFA fit and the data "
                    "noise. -1 = random each run", wrap=260)
            dpg.add_button(label="Generate Data", callback=self._generate_data_threaded,
                           width=-1)
            self.data_lbl = dpg.add_text("-", color=[150, 150, 150], wrap=340)

            dpg.add_spacer(height=6)

            # MFA Model Section
            dpg.add_text("MFA model", color=[0, 255, 255])
            dpg.add_separator()
            self.n_comp_in = dpg.add_input_int(
                label="# components", default_value=90, width=_width)
            with dpg.tooltip(self.n_comp_in):
                dpg.add_text(
                    "Total charts across all faces. Raise it with the number of "
                    "faces so each component stays well sampled.", wrap=260)
            self.pca_dim_in = dpg.add_input_int(
                label="PCA dim (0=off)", default_value=40, width=_width)
            with dpg.tooltip(self.pca_dim_in):
                dpg.add_text(
                    "The rotation sweep dominates the leading directions; keep "
                    "enough dimensions for the smaller expression arcs.", wrap=260)

            dpg.add_spacer(height=6)

            # Distance Section (Sec. 5.4)
            dpg.add_text("Distance metric (Sec. 5.4)", color=[0, 255, 255])
            dpg.add_separator()
            self.lambda_in = dpg.add_input_float(
                label="off-manifold penalty", default_value=30.0, min_value=0.0,
                format="%.2f", step=0.5, width=_width,
            )
            with dpg.tooltip(self.lambda_in):
                dpg.add_text("Penalty for moving off the tangent space.", wrap=260)
            self.k_distance_in = dpg.add_input_int(
                label="k - distance graph", default_value=4, width=_width)
            with dpg.tooltip(self.k_distance_in):
                dpg.add_text(
                    "Neighbors of the k-NN graph whose shortest paths give the "
                    "geodesic distance (Sec. 5.4).", wrap=260)

            dpg.add_spacer(height=6)

            # Interpolation Section (Sec. 5.7); detection is always TDA
            dpg.add_text("Interpolation (Sec. 5.7)", color=[0, 255, 255])
            dpg.add_separator()
            self.interp_w_in = dpg.add_input_float(
                label="tangent weight", default_value=3.0, min_value=0.0,
                step=0.5, width=_width,
            )
            with dpg.tooltip(self.interp_w_in):
                dpg.add_text(
                    "Strength of the soft chart-tangent alignment of the spline "
                    "0 = pure minimal-curvature interpolation.", wrap=260)

            dpg.add_spacer(height=6)
            dpg.add_button(label="Train + Detect (TDA)", callback=self._train_threaded,
                           width=-1)
            self.train_lbl = dpg.add_text("-", color=[150, 150, 150], wrap=340)

            dpg.add_spacer(height=6)

            # Evaluation Section (offline, saved to disk)
            dpg.add_text("Evaluation", color=[0, 255, 255])
            dpg.add_separator()
            eval_btn = dpg.add_button(label="Run evaluation (save to disk)",
                                      callback=self._run_evaluation_threaded, width=-1)
            with dpg.tooltip(eval_btn):
                dpg.add_text(
                    "Offline evaluation of the current table: topology (M1), ARI "
                    "against the face labels (M4) and the per-face factor error "
                    "in its own unit (M2), plus figures, into results/faces/.",
                    wrap=260,
                )
            self.eval_lbl = dpg.add_text("-", color=[150, 150, 150], wrap=340)

            dpg.add_spacer(height=6)

            # Render Mode Section
            dpg.add_text("Render Mode", color=[0, 255, 255])
            dpg.add_separator()
            self.mode_var = dpg.add_radio_button(
                ("samples", "pca", "spline", "persistence"),
                default_value="samples", horizontal=True,
                callback=self._on_mode_change,
            )

            dpg.add_spacer(height=6)

            # Component selector. Picks which detected curve the spline view reconstructs.
            dpg.add_text("Component to traverse", color=[0, 255, 255])
            dpg.add_separator()
            self.component_var = dpg.add_radio_button(
                ("-",), default_value="-", callback=self._on_component_change)

            dpg.add_spacer(height=8)

            # Slider Section
            dpg.add_text("Spline Parameter t", color=[0, 255, 255])
            dpg.add_separator()
            self.t_disp = dpg.add_text("t = 0.000")
            self.t_slider = dpg.add_slider_double(
                default_value=0.0, min_value=0.0, max_value=1.0,
                callback=self._on_slider, width=-1,
            )

    def _build_plot_panel(self):
        with dpg.group():
            with dpg.child_window(border=False, tag=self.plot_panel_tag, height=-40):
                self._swap_texture(self.tex_w, self.tex_h)

            dpg.add_button(
                label="Export current plot as PNG",
                callback=lambda: dpg.show_item(self.export_dialog_tag),
                width=-1,
            )
            self.export_status_lbl = dpg.add_text("", color=[150, 150, 150])

            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=self._on_export_path_chosen,
                tag=self.export_dialog_tag,
                width=600,
                height=400,
                default_filename="faces_plot.png",
            ):
                dpg.add_file_extension(".png")
                dpg.add_file_extension(".*")

    # ------------------ Interactions & Callbacks -----------------------
    def _register_mouse_handlers(self):
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                        callback=self._on_mouse_down)
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                          callback=self._on_mouse_up)

    def _on_mouse_down(self, sender, app_data):
        # drag to orbit the 3D PCA projection
        if not dpg.is_item_hovered(self.image_tag):
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

    def _throttled_update_plot(self):
        """Throttles high-frequency plot updates during mouse dragging."""
        if self._throttle_timer is None:
            self._request_async_plot()
            self._start_throttle_timer()
        else:
            self._pending_plot_request = True

    def _start_throttle_timer(self):
        self._throttle_timer = threading.Timer(PLOT_THROTTLE_SEC,
                                               self._on_throttle_timer_expiry)
        self._throttle_timer.start()

    def _on_throttle_timer_expiry(self):
        self._throttle_timer = None
        if self._pending_plot_request:
            self._pending_plot_request = False
            self._request_async_plot()
            self._start_throttle_timer()

    def _on_mode_change(self):
        """Validates if the model is trained before allowing spline selection."""
        chosen_mode = dpg.get_value(self.mode_var)
        needs_detection = {"spline": self.spline, "persistence": self.diagram}
        if chosen_mode in needs_detection and not needs_detection[chosen_mode]:
            dpg.set_value(self.mode_var, "samples")
            dpg.set_value(self.train_lbl,
                          f"Please train + detect first before selecting '{chosen_mode}'!")
            dpg.configure_item(self.train_lbl, color=[255, 50, 50])
            return

        self._request_async_plot()

    def _on_slider(self):
        t = dpg.get_value(self.t_slider)
        dpg.set_value(self.t_disp, f"t = {t:.3f}")
        if dpg.get_value(self.mode_var) == "spline":
            self._request_async_plot()

    # ---------------- Texture Management (Responsive Swapping) -----------------
    def _swap_texture(self, width, height, data=None):
        new_tag = f"{self._texture_tag_base}_{self._texture_counter}"
        self._texture_counter += 1

        if data is None:
            data = [0.12, 0.12, 0.14, 1.0] * (width * height)

        dpg.add_dynamic_texture(
            width=width, height=height, default_value=data,
            tag=new_tag, parent=self.texture_registry_tag,
        )

        if not dpg.does_item_exist(self.image_tag):
            dpg.add_image(new_tag, tag=self.image_tag, width=width, height=height,
                          parent=self.plot_panel_tag)
        else:
            dpg.configure_item(self.image_tag, texture_tag=new_tag,
                               width=width, height=height)

        old_tag = self.texture_tag
        self.texture_tag = new_tag
        self.tex_w, self.tex_h = width, height

        if old_tag is not None and dpg.does_item_exist(old_tag):
            dpg.delete_item(old_tag)

    def _on_window_resize(self):
        w, h = dpg.get_item_rect_size(self.plot_panel_tag)
        if w <= 0 or h <= 0:
            return

        new_w = int(np.clip(w - 15, MIN_TEX, MAX_TEX))
        new_h = int(np.clip(h - 15, MIN_TEX, MAX_TEX))

        if (abs(new_w - self.tex_w) < RESIZE_THRESHOLD and
                abs(new_h - self.tex_h) < RESIZE_THRESHOLD):
            return

        self._swap_texture(new_w, new_h)
        self._request_async_plot()

    def _on_export_path_chosen(self, sender, app_data):
        path = app_data["file_path_name"]
        if not path.lower().endswith(".png"):
            path += ".png"
        dpg.set_value(self.export_status_lbl, "Exporting...")
        dpg.configure_item(self.export_status_lbl, color=[255, 165, 0])
        threading.Thread(target=self._export_plot, args=(path,), daemon=True).start()

    def _export_plot(self, path):
        if self.X is None:
            dpg.set_value(self.export_status_lbl,
                          "Nothing to export - generate data first.")
            dpg.configure_item(self.export_status_lbl, color=[255, 0, 0])
            return

        export_dpi = 200
        fig = Figure(figsize=(10, 7.5), dpi=export_dpi)
        canvas = FigureCanvasAgg(fig)
        self._draw_state(fig, self._plot_state())
        canvas.draw()
        fig.savefig(path, dpi=export_dpi, bbox_inches="tight")
        fig.clf()

        dpg.set_value(self.export_status_lbl, f"Saved: {path}")
        dpg.configure_item(self.export_status_lbl, color=[0, 255, 0])

    # ---------------- Pipeline & Training ----------------------
    def _seed_value(self):
        """Read the seed field; -1 (or any negative) means random (None)."""
        seed = dpg.get_value(self.seed_in)
        return None if seed < 0 else seed

    def _selected_image_shape(self):
        """The resolution currently set in the panel"""
        size = int(dpg.get_value(self.image_size_in))
        return (size, size)

    def _current_specs(self):
        """
        Read the per-face table into a list of dataset specs.
        """
        specs = []
        for row in range(len(FACES)):
            if dpg.get_value(self.row_include[row]):
                specs.append({
                    "face": row,
                    "factor": dpg.get_value(self.row_factor[row]),
                    "samples": int(dpg.get_value(self.row_samples[row])),
                })
        return specs

    def _preset_three_faces(self):
        """The configuration of the thesis experiment: three faces, three factors."""
        for row in range(len(FACES)):
            use = row < len(PRESET_FACTORS)
            factor = PRESET_FACTORS[row] if use else DEFAULT_FACTORS[row]
            dpg.set_value(self.row_include[row], use)
            dpg.set_value(self.row_factor[row], factor)
            dpg.set_value(self.row_samples[row], 120)
        dpg.set_value(self.image_size_in, 64)
        dpg.set_value(self.noise_in, 0.02)
        dpg.set_value(self.seed_in, 0)
        dpg.set_value(self.n_comp_in, 90)
        dpg.set_value(self.pca_dim_in, 40)
        dpg.set_value(self.lambda_in, 30.0)
        dpg.set_value(self.k_distance_in, 4)
        dpg.set_value(self.interp_w_in, 3.0)

    def _generate_data_threaded(self):
        specs = self._current_specs()
        if not specs:
            dpg.set_value(self.data_lbl, "Select at least one face in the table.")
            dpg.configure_item(self.data_lbl, color=[255, 0, 0])
            return
        dpg.set_value(self.data_lbl, "Rendering faces...")
        dpg.configure_item(self.data_lbl, color=[255, 165, 0])
        threading.Thread(target=self._generate_data, args=(specs,), daemon=True).start()

    def _generate_data(self, specs):
        image_size = self._selected_image_shape()[0]
        if self.renderer is None or self.renderer.image_size != image_size:
            self.renderer = FaceRenderer(image_size=image_size)

        def report(done, total):
            dpg.set_value(self.data_lbl,
                          f"Rendering faces... {100 * done // total}%  "
                          f"({done}/{total} frames)")

        (self.X, self.values, self.component_gt, self.meta,
         self.pixel_mean, self.pixel_std) = make_multi_face_dataset(
            specs, image_size=image_size, add_noise=dpg.get_value(self.noise_in),
            random_state=self._seed_value(), renderer=self.renderer,
            progress=report,
        )
        self.specs = specs
        # the frames are drawn at the resolution they were rendered with
        self.image_shape = (image_size, image_size)

        # colour by the position within a frame's own sweep
        spans = np.array([[m["start"], m["end"]] for m in self.meta])[self.component_gt]
        width = np.where(spans[:, 1] != spans[:, 0], spans[:, 1] - spans[:, 0], 1.0)
        self.colors = (self.values - spans[:, 0]) / width
        self.captions = [
            f"{self.meta[i]['name']} "
            f"{_format_value(v, FACTOR_UNITS.get(self.meta[i]['factor'], 'units'))}"
            for v, i in zip(self.values, self.component_gt)
        ]

        _, _, Vt = np.linalg.svd(self.X, full_matrices=False)
        self.pca_basis = Vt[:3].T
        self.pca_data = self.X @ self.pca_basis

        self.data_label = ", ".join(_spec_label(s) for s in specs)
        dpg.set_value(self.data_lbl,
                      f"{len(self.X)} frames from {len(specs)} face(s): "
                      f"{self.data_label}.")
        dpg.configure_item(self.data_lbl, color=[0, 255, 0])

        # detection artifacts are stale once the data changes
        self.spline = None
        self.exp = None
        self.curves = None
        self.curve_projections = None
        self.curve_faces = []
        self.diagram = None
        self.selected_component = 0
        dpg.configure_item(self.component_var, items=("-",), default_value="-")
        dpg.set_value(self.mode_var, "samples")
        self._request_async_plot()

    def _train_threaded(self):
        if self.X is None:
            dpg.set_value(self.train_lbl, "Generate data first.")
            dpg.configure_item(self.train_lbl, color=[255, 0, 0])
            return
        dpg.set_value(self.train_lbl, "Training...")
        dpg.configure_item(self.train_lbl, color=[255, 165, 0])
        threading.Thread(target=self._train, daemon=True).start()

    def _train(self):
        pca_dim = dpg.get_value(self.pca_dim_in)

        exp = ManifoldPipeline(
            n_components=dpg.get_value(self.n_comp_in), latent_dim=1,
            cov_type="mfa", shared=False,
            pca_dim=pca_dim if pca_dim > 0 else None,
            lambda_aniso=dpg.get_value(self.lambda_in),
            n_neighbors=dpg.get_value(self.k_distance_in),
            detection="tda",
            interp_tangent_weight=dpg.get_value(self.interp_w_in),
            seed=self._seed_value(),
        )
        exp.fit(self.X.copy())
        self.exp = exp

        atlas_summary(exp.model.means, exp.tangents, exp.variances, exp.noise_)

        res = exp.detect()
        self.curves = res["curves"]
        self.diagram = exp.structure_.get("diagram")
        self._build_component_projections()
        self._set_component_selector()

        self.selected_component = 0
        self.spline = self.curves[0]["spline"] if self.curves else None

        # H0 = how many separate faces were found, H1 = loops (none expected)
        n_components = int(np.unique(_observation_components(self.exp, self.X)).size)
        n_loops = sum(c["type"] == "loop" for c in self.curves)
        n_paths = sum(c["type"] == "path" for c in self.curves)
        txt = (f"training done. H0: {n_components} component(s), "
               f"H1: {n_paths} arc(s)")
        if n_loops:
            txt += f" (+{n_loops} loop(s), none expected)"
        txt += "."
        dpg.set_value(self.train_lbl, txt)
        dpg.configure_item(self.train_lbl, color=[0, 255, 0])

        dpg.set_value(self.mode_var, "spline" if self.spline is not None else "pca")
        self._request_async_plot()

    def _run_evaluation_threaded(self):
        dpg.set_value(self.eval_lbl, "Starting face evaluation...")
        dpg.configure_item(self.eval_lbl, color=[255, 165, 0])
        threading.Thread(target=self._run_evaluation, daemon=True).start()

    def _run_evaluation(self):
        specs = self._current_specs()
        if not specs:
            dpg.set_value(self.eval_lbl, "Select at least one face.")
            dpg.configure_item(self.eval_lbl, color=[255, 0, 0])
            return

        seed = self._seed_value()
        if seed is None: # -1 = "random each run"
            seed = int(np.random.randint(2**31))

        try:
            _, out_dir = run_face_evaluation(
                specs=specs,
                image_size=self._selected_image_shape()[0],
                noise=dpg.get_value(self.noise_in),
                n_components=dpg.get_value(self.n_comp_in),
                pca_dim=dpg.get_value(self.pca_dim_in),
                lambda_aniso=dpg.get_value(self.lambda_in),
                n_neighbors=dpg.get_value(self.k_distance_in),
                interp_tangent_weight=dpg.get_value(self.interp_w_in),
                seed=seed,
                progress=lambda msg: dpg.set_value(self.eval_lbl, msg),
            )
            dpg.set_value(self.eval_lbl, f"Done. Results saved to:\n{out_dir}")
            dpg.configure_item(self.eval_lbl, color=[0, 255, 0])
        except Exception as exc:
            dpg.set_value(self.eval_lbl, f"Evaluation failed: {exc}")
            dpg.configure_item(self.eval_lbl, color=[255, 0, 0])

    def _component_labels(self):
        """
        One label per detected structure, naming the face it mostly covers.
        """
        if not self.curves:
            return []
        _, cid = self.exp.transform(self.X)
        labels = []
        self.curve_faces = []
        for j, curve in enumerate(self.curves):
            comp = curve.get("component")
            where = f" in H0 comp {comp}" if comp is not None else ""
            members = self.component_gt[cid == j] if self.component_gt is not None else []
            if len(members):
                vals, counts = np.unique(members, return_counts=True)
                face = _spec_label(self.specs[int(vals[np.argmax(counts)])])
                labels.append(f"{j}: {curve['type']}{where} ({face})")
            else:
                face = None
                labels.append(f"{j}: {curve['type']}{where}")
            self.curve_faces.append(face)
        return labels

    def _set_component_selector(self):
        labels = self._component_labels() or ["-"]
        self.component_label_to_idx = {lab: i for i, lab in enumerate(labels)}
        dpg.configure_item(self.component_var, items=labels, default_value=labels[0])

    def _build_component_projections(self):
        """Sample every detected curve and project it into the 3D PCA space."""
        self.curve_projections = []
        if not self.curves:
            return
        ts = np.linspace(0.0, 1.0, 200)
        for curve in self.curves:
            spline = curve.get("spline")
            if spline is None:
                self.curve_projections.append(None)
                continue
            pts_px = self.exp.reconstruct(np.asarray(spline(ts)))
            self.curve_projections.append(pts_px @ self.pca_basis)

    def _on_component_change(self, sender=None, app_data=None):
        if not self.curves:
            return
        idx = self.component_label_to_idx.get(dpg.get_value(self.component_var), 0)
        self.selected_component = idx
        self.spline = self.curves[idx].get("spline")
        self._request_async_plot()

    def _plot_state(self):
        return {
            "mode": dpg.get_value(self.mode_var),
            "t": dpg.get_value(self.t_slider),
            "width": self.tex_w,
            "height": self.tex_h,
            "title": self.data_label,
            "elev": self.elev,
            "azim": self.azim,
            "overlay_curves": self.curve_projections,
            "selected_component": self.selected_component,
            "cmap": "viridis",
            "color_label": "position along sweep",
            "color_scale": (0.0, 1.0),
            "factor_label": (self.curve_faces[self.selected_component]
                             if self.selected_component < len(self.curve_faces)
                             else None),
        }

    def _request_async_plot(self):
        if self.plot_queue.full():
            try:
                self.plot_queue.get_nowait()
            except queue.Empty:
                pass
        self.plot_queue.put(self._plot_state())

    def _spline_to_pixel(self, t: float) -> np.ndarray:
        if self.spline is None:
            return np.zeros(self.X.shape[1] if self.X is not None else 1)
        point = self.spline(t)
        if self.exp is not None and self.exp.pre is not None:
            point = self.exp.reconstruct(point)
        return point

    def _draw_state(self, fig, state):
        """Render one frame of the currently selected mode into `fig`."""
        mode = state["mode"]
        shape = self.image_shape
        if mode == "samples":
            render_samples_frame(fig, self.X, self.captions,
                                 f"Samples - {state['title']}",
                                 self.pixel_mean, self.pixel_std, shape)
        elif mode == "pca":
            render_pca_frame(fig, state, self.pca_data, self.colors, self.exp,
                             self.pca_basis, self._spline_to_pixel)
        elif mode == "spline":
            render_spline_frame(fig, state, self.pca_data, self.colors, self.exp,
                                self.pca_basis, self.spline, self._spline_to_pixel,
                                self.pixel_mean, self.pixel_std, shape)
        elif mode == "persistence":
            render_persistence_frame(fig, self.diagram, self.curves,
                                     self.selected_component)

    def _plot_worker(self):
        while True:
            state = self.plot_queue.get()
            if self.X is None:
                continue

            fig = Figure(figsize=(state["width"] / DPI, state["height"] / DPI), dpi=DPI)
            canvas = FigureCanvasAgg(fig)
            self._draw_state(fig, state)

            canvas.draw()
            w, h = canvas.get_width_height()
            buf = np.asarray(canvas.buffer_rgba(), dtype=np.float32) / 255.0
            flat_data = buf.flatten()

            if (w, h) != (self.tex_w, self.tex_h):
                self._swap_texture(w, h, data=flat_data)
            else:
                if self.texture_tag and dpg.does_item_exist(self.texture_tag):
                    dpg.set_value(self.texture_tag, flat_data)

            fig.clf()

    def run(self):
        dpg.setup_dearpygui()
        dpg.create_viewport(title="Faces - DPG Manifold Demo", width=1150, height=660)
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

