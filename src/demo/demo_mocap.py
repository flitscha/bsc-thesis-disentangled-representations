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

from data.mocap import make_multi_motion_dataset, MOTIONS, FACTOR_UNIT
from core.pipeline import ManifoldPipeline
from core.mfa import atlas_summary
from visualization.mocap import render_samples_frame, render_spline_frame
from visualization.mnist import render_pca_frame, render_persistence_frame
from experiments.mocap_eval import (
    run_evaluation as run_mocap_evaluation,
    _apply_h0_threshold,
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

# The verified configuration of the thesis experiment.
PRESET_MOTIONS = ("walk", "run", "wave", "sit_down")


class MocapDemoTab:
    """
    Motion capture poses from the CMU database (the catalogue 'data.mocap.MOTIONS'):
    every motion of the same person is a separate component, periodic motions
    (walking, running) are loops, one-way movements (sitting down) are arcs.
    The observations are 93-dimensional poses, not images.
    """

    def __init__(self):
        # State variables
        self.X = None
        self.pose_mean = None
        self.pose_std = None
        self.values = None # ground-truth progress per frame, in percent
        self.colors = None # progress in [0, 1], for colouring
        self.captions = None # formatted progress per frame
        self.component_gt = None # which spec (motion) a frame comes from
        self.meta = None
        self.specs = None
        self.exp = None
        self.spline = None
        self.pca_basis = None
        self.pca_data = None

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

        self._texture_tag_base = "mocap_plot_texture"
        self._texture_counter = 0
        self.texture_tag = None
        self.texture_registry_tag = "mocap_texture_registry"

        self.image_tag = "mocap_plot_image"
        self.plot_panel_tag = "mocap_canvas_container"
        self.handler_registry_tag = "mocap_handler_registry"
        self.window_handler_tag = "mocap_window_handler"

        self.export_dialog_tag = "mocap_export_dialog"
        self.export_status_lbl = None

        # Per-motion data table widgets (include + samples).
        self.row_include = {}
        self.row_samples = {}
        self.eval_lbl = None
        self.data_label = "-"

        # Multi-component visualization state
        self.diagram = None # persistence diagram of the detection
        self.curves = None # detected curves (with splines)
        self.curve_projections = None # each curve sampled + projected to PCA space
        self.curve_motions = [] # motion name per detected curve (majority vote)
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
                "Motion capture (CMU database, one subject): an observation is a "
                "pose, not an image. Every motion is a separate component; a "
                "periodic one (walking, running) closes into a loop, a one-way "
                "movement (sitting down) stays an arc. The factor is the "
                "progress within one repetition, in percent. "
                "Detected with TDA (Sec. 5.6).",
                wrap=340, color=[150, 150, 150],
            )
            dpg.add_spacer(height=8)

            # Data Section: per-motion table
            dpg.add_text("Data (motions of one person)", color=[0, 255, 255])
            dpg.add_separator()
            dpg.add_button(
                label="Preset: walk + run + sit down",
                callback=self._preset_three_motions, width=-1,
            )
            with dpg.group(horizontal=True):
                dpg.add_text("use", color=[150, 150, 150])
                dpg.add_text("  motion", color=[150, 150, 150])
                dpg.add_text("            type", color=[150, 150, 150])
                dpg.add_text("   samples", color=[150, 150, 150])
            for row, motion in enumerate(MOTIONS):
                with dpg.group(horizontal=True):
                    self.row_include[row] = dpg.add_checkbox(
                        default_value=motion["name"] in PRESET_MOTIONS)
                    dpg.add_text(f" {motion['name']:<15}")
                    dpg.add_text(f"{motion['kind']:<5}", color=[150, 150, 150])
                    self.row_samples[row] = dpg.add_input_int(
                        default_value=400, width=62, step=0)
                    with dpg.tooltip(dpg.last_item()):
                        dpg.add_text(motion["description"], wrap=260)

            dpg.add_spacer(height=6)
            self.noise_in = dpg.add_input_float(
                label="Joint noise (m)", default_value=0.0, min_value=0.0,
                step=0.005, format="%.3f", width=_width,
            )
            with dpg.tooltip(self.noise_in):
                dpg.add_text(
                    "Std of Gaussian noise on the joint coordinates, in metres "
                    "(0.01 = 1 cm). The recordings already carry the natural "
                    "variation between repetitions.", wrap=260)
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
                label="# components", default_value=100, width=_width)
            with dpg.tooltip(self.n_comp_in):
                dpg.add_text(
                    "Total charts across all motions. Raise it with the number "
                    "of motions so each component stays well covered.", wrap=260)
            self.pca_dim_in = dpg.add_input_int(
                label="PCA dim (0=off)", default_value=50, width=_width)
            with dpg.tooltip(self.pca_dim_in):
                dpg.add_text(
                    "A pose has 93 coordinates but a single motion moves in far "
                    "fewer directions.", wrap=260)

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
                    "geodesic distance (Sec. 5.4). Higher than in the image "
                    "demos: a motion is recorded several times, and the charts "
                    "of the different repetitions have to be linked.", wrap=260)

            dpg.add_spacer(height=6)

            # Detection / interpolation (Sec. 5.6 / 5.7)
            dpg.add_text("Detection (Sec. 5.6/5.7)", color=[0, 255, 255])
            dpg.add_separator()
            self.h0_factor_in = dpg.add_input_float(
                label="H0 threshold (0=auto)", default_value=0.0, min_value=0.0,
                step=0.1, format="%.2f", width=_width,
            )
            with dpg.tooltip(self.h0_factor_in):
                dpg.add_text(
                    "0 uses the automatic largest-gap rule of Sec. 5.6. A value "
                    "> 0 replaces it by an explicit persistence threshold at "
                    "that multiple of the median H0 merge scale, which is what "
                    "it takes once one motion sits much further away from the "
                    "others than they do from each other (try 2.2).", wrap=260)
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
                    "against the motion labels (M4) and the per-motion progress "
                    "error in percent (M2), plus figures, into results/mocap/.",
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
                default_filename="mocap_plot.png",
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

    def _current_specs(self):
        """Read the per-motion table into a list of dataset specs."""
        specs = []
        for row, motion in enumerate(MOTIONS):
            if dpg.get_value(self.row_include[row]):
                specs.append({
                    "motion": motion["name"],
                    "samples": int(dpg.get_value(self.row_samples[row])),
                })
        return specs

    def _preset_three_motions(self):
        """The configuration of the thesis experiment: two loops and one arc."""
        for row, motion in enumerate(MOTIONS):
            dpg.set_value(self.row_include[row], motion["name"] in PRESET_MOTIONS)
            dpg.set_value(self.row_samples[row], 400)
        dpg.set_value(self.noise_in, 0.0)
        dpg.set_value(self.seed_in, 0)
        dpg.set_value(self.n_comp_in, 100)
        dpg.set_value(self.pca_dim_in, 50)
        dpg.set_value(self.lambda_in, 30.0)
        dpg.set_value(self.k_distance_in, 4)
        dpg.set_value(self.h0_factor_in, 0.0)
        dpg.set_value(self.interp_w_in, 3.0)

    def _generate_data_threaded(self):
        specs = self._current_specs()
        if not specs:
            dpg.set_value(self.data_lbl, "Select at least one motion in the table.")
            dpg.configure_item(self.data_lbl, color=[255, 0, 0])
            return
        dpg.set_value(self.data_lbl, "Reading recordings...")
        dpg.configure_item(self.data_lbl, color=[255, 165, 0])
        threading.Thread(target=self._generate_data, args=(specs,), daemon=True).start()

    def _generate_data(self, specs):
        def report(done, total):
            dpg.set_value(self.data_lbl,
                          f"Reading recordings... ({done}/{total} motions)")

        try:
            (self.X, self.values, self.component_gt, self.meta,
             self.pose_mean, self.pose_std) = make_multi_motion_dataset(
                specs, add_noise=dpg.get_value(self.noise_in),
                random_state=self._seed_value(), progress=report,
            )
        except FileNotFoundError as exc:
            dpg.set_value(self.data_lbl, f"{exc}")
            dpg.configure_item(self.data_lbl, color=[255, 0, 0])
            return
        self.specs = specs

        # colour by the progress within the frame's own repetition
        self.colors = self.values / 100.0
        self.captions = [f"{self.meta[i]['motion']} {_format_value(v)}"
                         for v, i in zip(self.values, self.component_gt)]

        _, _, Vt = np.linalg.svd(self.X, full_matrices=False)
        self.pca_basis = Vt[:3].T
        self.pca_data = self.X @ self.pca_basis

        self.data_label = ", ".join(_spec_label(s) for s in specs)
        dpg.set_value(self.data_lbl,
                      f"{len(self.X)} poses from {len(specs)} motion(s): "
                      f"{self.data_label}.")
        dpg.configure_item(self.data_lbl, color=[0, 255, 0])

        # detection artifacts are stale once the data changes
        self.spline = None
        self.exp = None
        self.curves = None
        self.curve_projections = None
        self.curve_motions = []
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
        h0_factor = dpg.get_value(self.h0_factor_in)
        if h0_factor > 0:
            _apply_h0_threshold(exp, h0_factor)
            res = {"curves": exp.curves_}
        self.curves = res["curves"]
        self.diagram = exp.structure_.get("diagram")
        self._build_component_projections()
        self._set_component_selector()

        self.selected_component = 0
        self.spline = self.curves[0]["spline"] if self.curves else None

        # H0 = how many separate motions were found, H1 = loops
        n_components = int(np.unique(_observation_components(self.exp, self.X)).size)
        n_loops = sum(c["type"] == "loop" for c in self.curves)
        n_paths = sum(c["type"] == "path" for c in self.curves)
        dpg.set_value(self.train_lbl,
                      f"training done. H0: {n_components} component(s), "
                      f"H1: {n_loops} loop(s) + {n_paths} arc(s).")
        dpg.configure_item(self.train_lbl, color=[0, 255, 0])

        dpg.set_value(self.mode_var, "spline" if self.spline is not None else "pca")
        self._request_async_plot()

    def _run_evaluation_threaded(self):
        dpg.set_value(self.eval_lbl, "Starting mocap evaluation...")
        dpg.configure_item(self.eval_lbl, color=[255, 165, 0])
        threading.Thread(target=self._run_evaluation, daemon=True).start()

    def _run_evaluation(self):
        specs = self._current_specs()
        if not specs:
            dpg.set_value(self.eval_lbl, "Select at least one motion.")
            dpg.configure_item(self.eval_lbl, color=[255, 0, 0])
            return

        seed = self._seed_value()
        if seed is None: # -1 = "random each run"
            seed = int(np.random.randint(2**31))

        try:
            _, out_dir = run_mocap_evaluation(
                specs=specs,
                noise=dpg.get_value(self.noise_in),
                n_components=dpg.get_value(self.n_comp_in),
                pca_dim=dpg.get_value(self.pca_dim_in),
                lambda_aniso=dpg.get_value(self.lambda_in),
                n_neighbors=dpg.get_value(self.k_distance_in),
                interp_tangent_weight=dpg.get_value(self.interp_w_in),
                h0_persistence_factor=dpg.get_value(self.h0_factor_in),
                seed=seed,
                progress=lambda msg: dpg.set_value(self.eval_lbl, msg),
            )
            dpg.set_value(self.eval_lbl, f"Done. Results saved to:\n{out_dir}")
            dpg.configure_item(self.eval_lbl, color=[0, 255, 0])
        except Exception as exc:
            dpg.set_value(self.eval_lbl, f"Evaluation failed: {exc}")
            dpg.configure_item(self.eval_lbl, color=[255, 0, 0])

    def _component_labels(self):
        """One label per detected structure, naming the motion it mostly covers."""
        if not self.curves:
            return []
        _, cid = self.exp.transform(self.X)
        labels = []
        self.curve_motions = []
        for j, curve in enumerate(self.curves):
            comp = curve.get("component")
            where = f" in H0 comp {comp}" if comp is not None else ""
            members = self.component_gt[cid == j] if self.component_gt is not None else []
            if len(members):
                vals, counts = np.unique(members, return_counts=True)
                motion = _spec_label(self.specs[int(vals[np.argmax(counts)])])
                labels.append(f"{j}: {curve['type']}{where} ({motion})")
            else:
                motion = None
                labels.append(f"{j}: {curve['type']}{where}")
            self.curve_motions.append(motion)
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
            points = self.exp.reconstruct(np.asarray(spline(ts)))
            self.curve_projections.append(points @ self.pca_basis)

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
            "color_label": f"progress ({FACTOR_UNIT} of the repetition)",
            "color_scale": (0.0, 1.0),
            "factor_label": (self.curve_motions[self.selected_component]
                             if self.selected_component < len(self.curve_motions)
                             else None),
        }

    def _request_async_plot(self):
        if self.plot_queue.full():
            try:
                self.plot_queue.get_nowait()
            except queue.Empty:
                pass
        self.plot_queue.put(self._plot_state())

    def _spline_to_pose(self, t: float) -> np.ndarray:
        if self.spline is None:
            return np.zeros(self.X.shape[1] if self.X is not None else 1)
        point = self.spline(t)
        if self.exp is not None and self.exp.pre is not None:
            point = self.exp.reconstruct(point)
        return point

    def _draw_state(self, fig, state):
        """Render one frame of the currently selected mode into `fig`."""
        mode = state["mode"]
        if mode == "samples":
            render_samples_frame(fig, self.X, self.captions,
                                 f"Samples - {state['title']}",
                                 self.pose_mean, self.pose_std)
        elif mode == "pca":
            render_pca_frame(fig, state, self.pca_data, self.colors, self.exp,
                             self.pca_basis, self._spline_to_pose)
        elif mode == "spline":
            render_spline_frame(fig, state, self.pca_data, self.colors, self.exp,
                                self.pca_basis, self.spline, self._spline_to_pose,
                                self.pose_mean, self.pose_std)
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
        dpg.create_viewport(title="Mocap - DPG Manifold Demo", width=1150, height=660)
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
