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

from data.mnist_rotation import make_rotation_dataset
from core.experiment import Experiment
from core.atlas import extract_tangent_frame, atlas_summary
from core.graph import compute_score_matrix, build_knn_graph
from core.traversal import traverse_graph
from core.interpolation import build_closed_spline
from visualization.visualize import render_samples_frame, render_pca_frame, render_spline_frame

DPI = 100
MIN_TEX = 250
MAX_TEX = 1600
RESIZE_THRESHOLD = 25
ROTATE_SENSITIVITY = 0.4
PLOT_THROTTLE_SEC = 0.05 # Throttling interval for mouse dragging (20 FPS)

class MNISTDemoDPG:
    def __init__(self):
        # State variables
        self.X = None
        self.pixel_mean = None
        self.angles = None
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
        self._texture_tag_base = "mnist_plot_texture"
        self._texture_counter = 0
        self.texture_tag = None
        self.texture_registry_tag = "mnist_texture_registry"

        # Tags for UI control
        self.image_tag = "plot_image"
        self.plot_panel_tag = "CanvasContainer"

        # GUI Setup
        dpg.create_context()
        self._build_gui()

        # Start asynchronous plot worker thread
        threading.Thread(target=self._plot_worker, daemon=True).start()

        self._register_mouse_handlers()

    def _to_img(self, flat):
        img = (flat + self.pixel_mean).reshape(30, 30)
        return np.clip(img, 0.0, 1.0)

    def _spline_to_pixel(self, t: float) -> np.ndarray:
        if self.spline is None:
            return np.zeros(784)
        point = self.spline(t)
        if self.exp is not None and self.exp.pca_components is not None:
            point = self.exp.reconstruct(point)
        return point


    # ---------------- GUI Construction -------------------------
    def _build_gui(self):
        dpg.add_texture_registry(tag=self.texture_registry_tag, show=False)

        with dpg.window(tag="PrimaryWindow"):
            with dpg.group(horizontal=True):

                # Left Panel: Controls
                with dpg.child_window(width=300, border=True):
                    _width = 150

                    # Data Section
                    dpg.add_text("Data", color=[0, 255, 255])
                    dpg.add_separator()
                    self.digit_in = dpg.add_input_int(
                        label="Digit (0-9)", default_value=3, min_value=0, max_value=9, width=_width
                    )
                    self.n_angles_in = dpg.add_input_int(label="Number of samples", default_value=360, width=_width)
                    dpg.add_button(label="Generate Data", callback=self._generate_data, width=-1)
                    self.data_lbl = dpg.add_text("-", color=[150, 150, 150])

                    dpg.add_spacer(height=10)

                    # Model Section
                    dpg.add_text("Model", color=[0, 255, 255])
                    dpg.add_separator()
                    self.n_comp_in = dpg.add_input_int(label="# components", default_value=24, width=_width)
                    self.k_in = dpg.add_input_int(label="k (graph)", default_value=2, width=_width)
                    self.pca_dim_in = dpg.add_input_int(label="PCA dim (0=off)", default_value=0, width=_width)
                    dpg.add_button(label="Train + Build Spline", callback=self._train_threaded, width=-1)
                    self.train_lbl = dpg.add_text("-", color=[150, 150, 150], wrap=280)

                    dpg.add_spacer(height=10)

                    # Render Mode Section
                    dpg.add_text("Render Mode", color=[0, 255, 255])
                    dpg.add_separator()
                    self.mode_var = dpg.add_radio_button(
                        ("samples", "pca", "spline"),
                        default_value="samples",
                        callback=self._on_mode_change
                    )

                    # Visualization (2D / 3D)
                    dpg.add_spacer(height=5)
                    dpg.add_text("Visualization dimension:")
                    self.dim_var = dpg.add_radio_button(
                        ("2D", "3D"),
                        default_value="2D",
                        horizontal=True,
                        callback=self._request_async_plot
                    )

                    dpg.add_spacer(height=10)

                    # Slider Section
                    dpg.add_text("Spline Parameter t", color=[0, 255, 255])
                    dpg.add_separator()
                    self.t_disp = dpg.add_text("t = 0.000  (0°)")
                    self.t_slider = dpg.add_slider_double(
                        default_value=0.0, min_value=0.0, max_value=1.0,
                        callback=self._on_slider, width=-1
                    )

                # Right Panel: Canvas Plot Area
                with dpg.child_window(border=False, tag=self.plot_panel_tag):
                    self._swap_texture(self.tex_w, self.tex_h)

        dpg.set_primary_window("PrimaryWindow", True)

        # Register window resize handler
        with dpg.item_handler_registry(tag="window_handler"):
            dpg.add_item_resize_handler(callback=self._on_window_resize)
        dpg.bind_item_handler_registry(self.plot_panel_tag, "window_handler")


    # ------------------ Interactions & Callbacks -----------------------
    def _register_mouse_handlers(self):
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_down)
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_up)

    def _is_3d(self):
        return dpg.get_value(self.dim_var) == "3D"

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

        # Use the requested throttled plot update mechanism
        self._throttled_update_plot()

    def _on_mouse_up(self, sender, app_data):
        self._dragging = False
        self._last_mouse_pos = None

    def _throttled_update_plot(self):
        """Throttles high-frequency plot updates during mouse dragging."""
        if self._throttle_timer is None:
            # First call or timer expired: execute plot immediately
            self._request_async_plot()
            self._start_throttle_timer()
        else:
            # Timer active: mark request as pending to process at expiration
            self._pending_plot_request = True

    def _start_throttle_timer(self):
        self._throttle_timer = threading.Timer(PLOT_THROTTLE_SEC, self._on_throttle_timer_expiry)
        self._throttle_timer.start()

    def _on_throttle_timer_expiry(self):
        self._throttle_timer = None
        if self._pending_plot_request:
            self._pending_plot_request = False
            self._request_async_plot()
            # Restart timer to handle continuous dragging
            self._start_throttle_timer()

    def _on_mode_change(self):
        """Validates if the model is trained before allowing spline selection."""
        chosen_mode = dpg.get_value(self.mode_var)
        if chosen_mode == "spline" and self.spline is None:
            # Force UI fallback to 'samples' mode
            dpg.set_value(self.mode_var, "samples")
            dpg.set_value(self.train_lbl, "Please train the model first before selecting Spline!")
            dpg.configure_item(self.train_lbl, color=[255, 50, 50])
            return

        self._request_async_plot()

    def _on_slider(self):
        t = dpg.get_value(self.t_slider)
        dpg.set_value(self.t_disp, f"t = {t:.3f}  ({t * 360:.0f}°)")
        if dpg.get_value(self.mode_var) == "spline":
            self._request_async_plot()


    # ---------------- Texture Management (Responsive Swapping) -----------------------
    def _swap_texture(self, width, height, data=None):
        new_tag = f"{self._texture_tag_base}_{self._texture_counter}"
        self._texture_counter += 1

        if data is None:
            data = [1.0, 1.0, 1.0, 1.0] * (width * height)

        dpg.add_dynamic_texture(
            width=width, height=height, default_value=data,
            tag=new_tag, parent=self.texture_registry_tag,
        )

        if not dpg.does_item_exist(self.image_tag):
            dpg.add_image(new_tag, tag=self.image_tag, width=width, height=height, parent=self.plot_panel_tag)
        else:
            dpg.configure_item(self.image_tag, texture_tag=new_tag, width=width, height=height)

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

        if abs(new_w - self.tex_w) < RESIZE_THRESHOLD and abs(new_h - self.tex_h) < RESIZE_THRESHOLD:
            return

        self._swap_texture(new_w, new_h)
        self._request_async_plot()


    # ---------------- Pipeline & Training ----------------------
    def _generate_data(self):
        dpg.set_value(self.data_lbl, "Loading...")
        dpg.configure_item(self.data_lbl, color=[255, 165, 0])

        self.X, self.angles, self.pixel_mean = make_rotation_dataset(
            digit=dpg.get_value(self.digit_in),
            n_angles=dpg.get_value(self.n_angles_in),
            n_images=1,
            center=True,
        )

        _, _, Vt = np.linalg.svd(self.X, full_matrices=False)
        self.pca_basis = Vt[:3].T                 
        self.pca_data  = self.X @ self.pca_basis  

        N = len(self.X)
        dpg.set_value(self.data_lbl, f"Digit {dpg.get_value(self.digit_in)}: {N} pts generated.")
        dpg.configure_item(self.data_lbl, color=[0, 255, 0])

        self.spline = None
        self.exp = None
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
        C = dpg.get_value(self.n_comp_in)
        k = dpg.get_value(self.k_in)
        pca_dim = dpg.get_value(self.pca_dim_in)

        exp = Experiment(
            data_type="external", N=len(self.X), C=C, H=1,
            cov_type="mfa", shared=False,
            pca_dim=pca_dim if pca_dim > 0 else None,
        )
        exp.data = self.X.copy()
        exp.train()
        self.exp = exp

        means = exp.model.means
        covs  = exp.model.covariances
        tangents, variances, noise_var = extract_tangent_frame(covs, n_tangents=1)
        atlas_summary(means, tangents, variances, noise_var)

        score = compute_score_matrix(means, tangents, variances)
        graph = build_knn_graph(score, k=k)

        order = traverse_graph(graph["adjacency"])
        self.spline = build_closed_spline(means[order])

        txt = "training done."
        dpg.set_value(self.train_lbl, txt)
        dpg.configure_item(self.train_lbl, color=[0, 255, 0])

        dpg.set_value(self.mode_var, "spline")
        self._request_async_plot()

    def _request_async_plot(self):
        if self.plot_queue.full():
            try:
                self.plot_queue.get_nowait()
            except queue.Empty:
                pass

        state = {
            "mode": dpg.get_value(self.mode_var),
            "is_3d": dpg.get_value(self.dim_var) == "3D",
            "t": dpg.get_value(self.t_slider),
            "width": self.tex_w,
            "height": self.tex_h,
            "digit": dpg.get_value(self.digit_in),
            "elev": self.elev,
            "azim": self.azim
        }
        self.plot_queue.put(state)

    def _plot_worker(self):
        while True:
            state = self.plot_queue.get()
            if self.X is None:
                continue

            fig = Figure(figsize=(state["width"] / DPI, state["height"] / DPI), dpi=DPI)
            canvas = FigureCanvasAgg(fig)

            mode = state["mode"]
            if mode == "samples":
                render_samples_frame(fig, self.X, self.angles, state["digit"])
            elif mode == "pca":
                render_pca_frame(fig, state, self.pca_data, self.angles, self.exp, self.pca_basis, self._spline_to_pixel)
            elif mode == "spline":
                render_spline_frame(fig, state, self.X, self.pca_data, self.angles, self.exp, self.pca_basis, self.spline, self._spline_to_pixel)

            canvas.draw()
            w, h = canvas.get_width_height()
            buf = np.asarray(canvas.buffer_rgba(), dtype=np.float32) / 255.0

            if (w, h) != (self.tex_w, self.tex_h):
                self._swap_texture(w, h, data=buf.flatten())
            else:
                dpg.set_value(self.texture_tag, buf.flatten())

            fig.clf()


    def run(self):
        dpg.setup_dearpygui()
        dpg.create_viewport(title="MNIST Rotation - DPG Manifold Demo", width=1150, height=660)
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


if __name__ == "__main__":
    MNISTDemoDPG().run()

