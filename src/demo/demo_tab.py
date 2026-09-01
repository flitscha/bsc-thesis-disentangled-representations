"""
The machinery shared by the three dataset tabs: MNIST, faces and motion capture.

'ManifoldDemoTab' owns everything that is not specific to a dataset: the texture handling, the
background plot thread with its drag throttling, the mouse orbit, the PNG export and the common
settings blocks. A subclass supplies its own dataset and figures by overriding the hooks marked
below, and inherits the rest.

Every widget tag is prefixed with the subclass's 'name', so that several tabs can live in one
DearPyGui context.
"""

import queue
import threading

import numpy as np
import dearpygui.dearpygui as dpg

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from core.pipeline import ManifoldPipeline
from core.mfa import atlas_summary
from experiments.eval import component_labels

DPI = 100
MIN_TEX = 250
MAX_TEX = 1600
RESIZE_THRESHOLD = 25
ROTATE_SENSITIVITY = 0.4
PLOT_THROTTLE_SEC = 0.05 # 20 fps while the mouse is dragged

_LABEL_WIDTH = 125
_GREY = [150, 150, 150]
_CYAN = [0, 255, 255]
_GREEN = [0, 255, 0]
_ORANGE = [255, 165, 0]
_RED = [255, 0, 0]


class ManifoldDemoTab:

    # --- per-tab identity, set by the subclass ---
    name = "demo" # tag prefix, has to be unique per tab
    title = "Manifold Demo" # window title when the tab is run on its own
    intro = "" # the text at the top of the settings panel
    eval_tooltip = "" # describes what the evaluation button writes to disk
    h0_hint = "" # advice on the H0 threshold, specific to the dataset
    panel_width = 380
    wrap = 340

    def __init__(self):
        self.X = None
        self.exp = None
        self.spline = None
        self.pca_basis = None
        self.pca_data = None
        self.specs = None
        self.data_label = "-"

        # camera of the 3D PCA view
        self.azim = -60.0
        self.elev = 30.0
        self._dragging = False
        self._last_mouse_pos = None

        self._throttle_timer = None
        self._pending_plot_request = False

        self.plot_queue = queue.Queue(maxsize=1)
        self.tex_w, self.tex_h = 800, 600
        self._texture_counter = 0
        self.texture_tag = None

        self._texture_tag_base = f"{self.name}_plot_texture"
        self.texture_registry_tag = f"{self.name}_texture_registry"
        self.image_tag = f"{self.name}_plot_image"
        self.plot_panel_tag = f"{self.name}_canvas_container"
        self.window_handler_tag = f"{self.name}_window_handler"
        self.export_dialog_tag = f"{self.name}_export_dialog"

        # detection artifacts
        self.diagram = None
        self.curves = None
        self.curve_projections = None # each curve sampled and projected to PCA space
        self.curve_labels = [] # majority ground-truth class per curve
        self.selected_component = 0
        self.component_label_to_idx = {}

        # widgets filled in while the panel is built
        self.component_var = None
        self.export_status_lbl = None
        self.eval_lbl = None

    # --- hooks a subclass must implement ---
    def _build_data_section(self):
        """The dataset table plus '_add_noise_seed_generate'."""
        raise NotImplementedError

    def _build_model_section(self):
        """The MFA block. Call '_add_model_inputs' with this dataset's defaults."""
        raise NotImplementedError

    def _current_specs(self):
        """Read the table into a list of dataset specs."""
        raise NotImplementedError

    def _load_data(self, specs, report):
        """
        Generate the data for 'specs' and fill self.X and the per-tab arrays.

        'report(done, total)' updates the status line. Returns the message to show once loading
        succeeded, or None to abort quietly, in which case the hook reports the failure itself.
        """
        raise NotImplementedError

    def _draw_state(self, fig, state):
        """Render the currently selected mode into 'fig'."""
        raise NotImplementedError

    def _evaluate(self, specs, seed, progress):
        """Run the offline evaluation and return the output directory."""
        raise NotImplementedError

    # --- hooks with a usable default ---
    def _member_ids(self):
        """The ground-truth class of every observation, used to name the detected curves."""
        return None

    def _member_name(self, value):
        """Readable name of the ground-truth class 'value'."""
        return str(value)

    def _detect(self, exp):
        """
        Run the detection and return the list of curves.

        The automatic rules go first, because the panel's H0 and H1 thresholds are multiples of
        the barcode's own merge scale, and that is only known once a detection has run. While both
        are 0 the second call does nothing.
        """
        exp.detect()
        exp.apply_persistence_thresholds(
            h0_factor=dpg.get_value(self.h0_factor_in),
            h1_factor=dpg.get_value(self.h1_factor_in),
        )
        return exp.curves_

    def _training_summary(self):
        n_components, n_loops, n_paths = self._topology_counts()
        return (f"training done. H0: {n_components} component(s), "
                f"H1: {n_loops} loop(s) + {n_paths} arc(s).")

    def _format_t(self, t):
        return f"t = {t:.3f}"

    def _extra_plot_state(self):
        """Additional keys the tab's render functions read."""
        return {}

    def _build_detection_section(self):
        """The TDA thresholds and the interpolation strength."""
        dpg.add_text("TDA and interpolation", color=_CYAN)
        dpg.add_separator()
        self._add_tda_inputs()
        self._add_interp_input()

    # --- panel construction ---
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
        with dpg.child_window(width=self.panel_width, border=True):
            dpg.add_text(self.intro, wrap=self.wrap, color=_GREY)
            dpg.add_spacer(height=8)

            self._build_data_section()
            dpg.add_spacer(height=6)
            self._build_model_section()
            dpg.add_spacer(height=6)
            self._build_distance_section()
            dpg.add_spacer(height=6)
            self._build_detection_section()

            dpg.add_spacer(height=6)
            dpg.add_button(label="Train", callback=self._train_threaded, width=-1)
            self.train_lbl = dpg.add_text("-", color=_GREY, wrap=self.wrap)

            dpg.add_spacer(height=6)
            dpg.add_text("Evaluation", color=_CYAN)
            dpg.add_separator()
            eval_btn = dpg.add_button(label="Run evaluation (save to disk)",
                                      callback=self._run_evaluation_threaded, width=-1)
            with dpg.tooltip(eval_btn):
                dpg.add_text(self.eval_tooltip, wrap=260)
            self.eval_lbl = dpg.add_text("-", color=_GREY, wrap=self.wrap)

            dpg.add_spacer(height=6)
            self._build_view_sections()

    def _add_noise_seed_generate(self, noise_label, noise_default, noise_tooltip,
                                 noise_step=0.01, noise_format=None, seed_default=0):
        """The noise field, the seed field and the 'Generate Data' button."""
        kwargs = {} if noise_format is None else {"format": noise_format}
        self.noise_in = dpg.add_input_float(
            label=noise_label, default_value=noise_default, min_value=0.0,
            step=noise_step, width=_LABEL_WIDTH, **kwargs,
        )
        with dpg.tooltip(self.noise_in):
            dpg.add_text(noise_tooltip, wrap=260)

        self.seed_in = dpg.add_input_int(
            label="RNG seed (-1 = random)", default_value=seed_default, width=_LABEL_WIDTH)
        with dpg.tooltip(self.seed_in):
            dpg.add_text("Seeds the PCA random rotation, the MFA fit and the data "
                         "noise. -1 = random each run", wrap=260)

        dpg.add_button(label="Generate Data", callback=self._generate_data_threaded,
                       width=-1)
        self.data_lbl = dpg.add_text("-", color=_GREY, wrap=self.wrap)

    def _add_model_inputs(self, n_components, pca_dim, n_comp_tooltip, pca_tooltip=None):
        """The '# components' and 'PCA dim' fields of the MFA block."""
        dpg.add_text("MFA model", color=_CYAN)
        dpg.add_separator()
        self.n_comp_in = dpg.add_input_int(
            label="# components", default_value=n_components, width=_LABEL_WIDTH)
        with dpg.tooltip(self.n_comp_in):
            dpg.add_text(n_comp_tooltip, wrap=260)
        self.pca_dim_in = dpg.add_input_int(
            label="PCA dim (0 = off)", default_value=pca_dim, width=_LABEL_WIDTH)
        if pca_tooltip:
            with dpg.tooltip(self.pca_dim_in):
                dpg.add_text(pca_tooltip, wrap=260)

    def _build_distance_section(self, k=4, k_tooltip=None):
        dpg.add_text("Distance metric", color=_CYAN)
        dpg.add_separator()
        self.lambda_in = dpg.add_input_float(
            label="off-manifold penalty", default_value=30.0, min_value=0.0,
            format="%.2f", step=0.5, width=_LABEL_WIDTH,
        )
        with dpg.tooltip(self.lambda_in):
            dpg.add_text("Penalty lambda for moving off the tangent space. "
                         "0 = plain Euclidean, larger stretches the normal "
                         "directions more.", wrap=260)
        self.k_distance_in = dpg.add_input_int(
            label="k - distance graph", default_value=k, width=_LABEL_WIDTH)
        with dpg.tooltip(self.k_distance_in):
            dpg.add_text(k_tooltip or
                         "Neighbors of the k-NN graph whose shortest paths give "
                         "the geodesic distance.", wrap=260)

    def _add_tda_inputs(self):
        """The two persistence thresholds, given as multiples of the merge scale."""
        self.h0_factor_in = dpg.add_input_float(
            label="H0 threshold (0 = auto)", default_value=0.0, min_value=0.0,
            step=0.1, format="%.2f", width=_LABEL_WIDTH,
        )
        with dpg.tooltip(self.h0_factor_in):
            dpg.add_text(
                "How persistent a connected component has to be to count. "
                "0 uses the automatic largest-gap rule; a value > 0 replaces it "
                "by an explicit threshold at that multiple of the median H0 "
                "merge scale, which keeps it free of the data scale. " +
                self.h0_hint, wrap=260)

        self.h1_factor_in = dpg.add_input_float(
            label="H1 threshold (0 = auto)", default_value=0.0, min_value=0.0,
            step=0.1, format="%.2f", width=_LABEL_WIDTH,
        )
        with dpg.tooltip(self.h1_factor_in):
            dpg.add_text(
                "How persistent a loop has to be to count. 0 uses the automatic "
                "prominence rule (death >= 1.8 x birth); a value > 0 replaces it "
                "by an explicit threshold at that multiple of the median H0 "
                "merge scale. Raise it to suppress spurious loops.", wrap=260)

    def _add_interp_input(self):
        self.interp_w_in = dpg.add_input_float(
            label="tangent weight", default_value=3.0, min_value=0.0,
            step=0.5, width=_LABEL_WIDTH,
        )
        with dpg.tooltip(self.interp_w_in):
            dpg.add_text("Strength of the soft chart-tangent alignment of the "
                         "spline. 0 = pure minimal-curvature interpolation.", wrap=260)

    def _build_view_sections(self):
        """The render mode, the component selector and the t slider."""
        dpg.add_text("Render mode", color=_CYAN)
        dpg.add_separator()
        self.mode_var = dpg.add_radio_button(
            ("samples", "pca", "spline", "persistence"),
            default_value="samples", horizontal=True, callback=self._on_mode_change,
        )

        dpg.add_spacer(height=6)
        dpg.add_text("Component to traverse", color=_CYAN)
        dpg.add_separator()
        self.component_var = dpg.add_radio_button(
            ("-",), default_value="-", callback=self._on_component_change)

        dpg.add_spacer(height=8)
        dpg.add_text("Spline parameter t", color=_CYAN)
        dpg.add_separator()
        self.t_disp = dpg.add_text(self._format_t(0.0))
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
                callback=lambda: dpg.show_item(self.export_dialog_tag), width=-1,
            )
            self.export_status_lbl = dpg.add_text("", color=_GREY)

            with dpg.file_dialog(
                directory_selector=False, show=False,
                callback=self._on_export_path_chosen, tag=self.export_dialog_tag,
                width=600, height=400, default_filename=f"{self.name}_plot.png",
            ):
                dpg.add_file_extension(".png")
                dpg.add_file_extension(".*")

    # --- mouse orbit and throttling ---
    def _register_mouse_handlers(self):
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                        callback=self._on_mouse_down)
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                          callback=self._on_mouse_up)

    def _on_mouse_down(self, sender, app_data):
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
        """Plot at most once per PLOT_THROTTLE_SEC while the mouse is dragged."""
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
        """Fall back to 'samples' when the chosen mode would need a detection first."""
        chosen = dpg.get_value(self.mode_var)
        needs_detection = {"spline": self.spline, "persistence": self.diagram}
        if chosen in needs_detection and not needs_detection[chosen]:
            dpg.set_value(self.mode_var, "samples")
            dpg.set_value(self.train_lbl,
                          f"Please train + detect first before selecting '{chosen}'!")
            dpg.configure_item(self.train_lbl, color=[255, 50, 50])
            return
        self._request_async_plot()

    def _on_slider(self):
        t = dpg.get_value(self.t_slider)
        dpg.set_value(self.t_disp, self._format_t(t))
        if dpg.get_value(self.mode_var) == "spline":
            self._request_async_plot()

    # --- texture handling ---
    def _swap_texture(self, width, height, data=None):
        """Replace the plot texture, since a DearPyGui texture cannot be resized in place."""
        new_tag = f"{self._texture_tag_base}_{self._texture_counter}"
        self._texture_counter += 1

        if data is None:
            data = [0.12, 0.12, 0.14, 1.0] * (width * height)

        dpg.add_dynamic_texture(width=width, height=height, default_value=data,
                                tag=new_tag, parent=self.texture_registry_tag)

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

    # --- export ---
    def _on_export_path_chosen(self, sender, app_data):
        path = app_data["file_path_name"]
        if not path.lower().endswith(".png"):
            path += ".png"
        self._status(self.export_status_lbl, "Exporting...", _ORANGE)
        threading.Thread(target=self._export_plot, args=(path,), daemon=True).start()

    def _export_plot(self, path):
        if self.X is None:
            self._status(self.export_status_lbl,
                         "Nothing to export - generate data first.", _RED)
            return

        export_dpi = 200
        fig = Figure(figsize=(10, 7.5), dpi=export_dpi)
        canvas = FigureCanvasAgg(fig)
        self._draw_state(fig, self._plot_state())
        canvas.draw()
        fig.savefig(path, dpi=export_dpi, bbox_inches="tight")
        fig.clf()

        self._status(self.export_status_lbl, f"Saved: {path}", _GREEN)

    # --- data, training and evaluation ---
    @staticmethod
    def _status(label, text, color):
        dpg.set_value(label, text)
        dpg.configure_item(label, color=color)

    def _seed_value(self):
        """Read the seed field, where a negative value means random."""
        seed = dpg.get_value(self.seed_in)
        return None if seed < 0 else seed

    def _concrete_seed(self):
        """The seed to save with a run. Draws one if the field is set to random."""
        seed = self._seed_value()
        return int(np.random.randint(2**31)) if seed is None else seed

    def _generate_data_threaded(self):
        specs = self._current_specs()
        if not specs:
            self._status(self.data_lbl, "Select at least one entry in the table.", _RED)
            return
        self._status(self.data_lbl, "Loading...", _ORANGE)
        threading.Thread(target=self._generate_data, args=(specs,), daemon=True).start()

    def _generate_data(self, specs):
        def report(done, total):
            dpg.set_value(self.data_lbl, f"Loading... ({done}/{total})")

        message = self._load_data(specs, report)
        if message is None:
            return
        self.specs = specs

        _, _, Vt = np.linalg.svd(self.X, full_matrices=False)
        self.pca_basis = Vt[:3].T
        self.pca_data = self.X @ self.pca_basis
        self._status(self.data_lbl, message, _GREEN)

        # detection artifacts are stale once the data changes
        self.spline = None
        self.exp = None
        self.curves = None
        self.curve_projections = None
        self.curve_labels = []
        self.diagram = None
        self.selected_component = 0
        dpg.configure_item(self.component_var, items=("-",), default_value="-")
        dpg.set_value(self.mode_var, "samples")
        self._request_async_plot()

    def _train_threaded(self):
        if self.X is None:
            self._status(self.train_lbl, "Generate data first.", _RED)
            return
        self._status(self.train_lbl, "Training...", _ORANGE)
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

        self.curves = self._detect(exp)
        self.diagram = exp.structure_.get("diagram")
        self._build_component_projections()
        self._set_component_selector()

        self.selected_component = 0
        self.spline = self.curves[0]["spline"] if self.curves else None

        self._status(self.train_lbl, self._training_summary(), _GREEN)
        dpg.set_value(self.mode_var, "spline" if self.spline is not None else "pca")
        self._request_async_plot()

    def _topology_counts(self):
        """The number of components backed by data, of detected loops and of detected paths."""
        n_components = int(np.unique(component_labels(self.exp, self.X)).size)
        n_loops = sum(c["type"] == "loop" for c in self.curves)
        n_paths = sum(c["type"] == "path" for c in self.curves)
        return n_components, n_loops, n_paths

    def _run_evaluation_threaded(self):
        self._status(self.eval_lbl, "Starting evaluation...", _ORANGE)
        threading.Thread(target=self._run_evaluation, daemon=True).start()

    def _run_evaluation(self):
        specs = self._current_specs()
        if not specs:
            self._status(self.eval_lbl, "Select at least one entry.", _RED)
            return
        try:
            out_dir = self._evaluate(
                specs, self._concrete_seed(),
                progress=lambda msg: dpg.set_value(self.eval_lbl, msg),
            )
            if out_dir is not None:
                self._status(self.eval_lbl, f"Done. Results saved to:\n{out_dir}", _GREEN)
        except Exception as exc:
            self._status(self.eval_lbl, f"Evaluation failed: {exc}", _RED)

    # --- detected curves ---
    def _component_labels(self):
        """One label per detected curve, naming the class it mostly covers."""
        if not self.curves:
            return []
        _, cid = self.exp.transform(self.X)
        member_ids = self._member_ids()

        labels = []
        self.curve_labels = []
        for j, curve in enumerate(self.curves):
            comp = curve.get("component")
            where = f" in H0 comp {comp}" if comp is not None else ""
            members = member_ids[cid == j] if member_ids is not None else []
            if len(members):
                values, counts = np.unique(members, return_counts=True)
                name = self._member_name(values[np.argmax(counts)])
                labels.append(f"{j}: {curve['type']}{where} ({name})")
            else:
                name = None
                labels.append(f"{j}: {curve['type']}{where}")
            self.curve_labels.append(name)
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

    def _selected_curve_label(self):
        if self.selected_component < len(self.curve_labels):
            return self.curve_labels[self.selected_component]
        return None

    # --- plotting ---
    def _plot_state(self):
        state = {
            "mode": dpg.get_value(self.mode_var),
            "t": dpg.get_value(self.t_slider),
            "width": self.tex_w,
            "height": self.tex_h,
            "title": self.data_label,
            "elev": self.elev,
            "azim": self.azim,
            "overlay_curves": self.curve_projections,
            "selected_component": self.selected_component,
        }
        state.update(self._extra_plot_state())
        return state

    def _request_async_plot(self):
        if self.plot_queue.full():
            try:
                self.plot_queue.get_nowait()
            except queue.Empty:
                pass
        self.plot_queue.put(self._plot_state())

    def _spline_point(self, t):
        """The ambient-space point the spline reaches at t."""
        if self.spline is None:
            return np.zeros(self.X.shape[1] if self.X is not None else 1)
        point = self.spline(t)
        if self.exp is not None and self.exp.pre is not None:
            point = self.exp.reconstruct(point)
        return point

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
            flat_data = (np.asarray(canvas.buffer_rgba(), dtype=np.float32) / 255.0).flatten()

            if (w, h) != (self.tex_w, self.tex_h):
                self._swap_texture(w, h, data=flat_data)
            elif self.texture_tag and dpg.does_item_exist(self.texture_tag):
                dpg.set_value(self.texture_tag, flat_data)

            fig.clf()

    def run(self):
        """Show this tab alone in its own viewport."""
        dpg.setup_dearpygui()
        dpg.create_viewport(title=self.title, width=1150, height=660)
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

