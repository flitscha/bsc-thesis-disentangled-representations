"""
Tab 1: Synthetic Explorer
"""

import sys
import os
import time

import numpy as np
import dearpygui.dearpygui as dpg

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pipeline import ManifoldPipeline
from data.synthetic import make_dataset
from visualization.gmm import visualize_gmm


DPI = 100

# Bounds for resolution of plot (the resolution changes dynamicaly depending on the window size)
MIN_TEX = 250
MAX_TEX = 1600
RESIZE_THRESHOLD = 25 # avoid rerendering spam during window-resizing

ROTATE_SENSITIVITY = 0.4


class SyntheticDemoTab:
    _instance_counter = 0

    def __init__(self):
        SyntheticDemoTab._instance_counter += 1
        uid = SyntheticDemoTab._instance_counter
        self._texture_tag_base = f"synthetic_plot_texture_{uid}"
        self._texture_counter = 0
        self.texture_tag = None # set in _build_plot_panel
        self.texture_registry_tag = f"synthetic_texture_registry_{uid}"
        self.image_tag = f"synthetic_plot_image_{uid}"
        self.plot_panel_tag = f"synthetic_plot_panel_{uid}"

        self.pipe = None
        self.data = None
        self.projection = None

        # initial texture resolution
        self.tex_w, self.tex_h = 700, 700

        # initial camera rotation (for 3D plots)
        self.azim = -60.0
        self.elev = 30.0
        self._dragging = False
        self._last_mouse_pos = None
        self._last_render_time = 0.0

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
                "Fits a mixture model to a dataset and visualizes the "
                "components and their tangent estimates. This tab explores the "
                "fit itself (covariance type, number of components); it does "
                "not run the distance or detection steps.",
                wrap=380, color=[150, 150, 150],
            )
            dpg.add_separator()

            dpg.add_text("Data")
            self.data_type = dpg.add_combo(
                ("line", "circle", "swiss_roll", "torus", "curve_in_3d"),
                default_value="circle", label="Data", width=_width
            )
            self.num_points = dpg.add_input_int(
                label="Number of data points", default_value=100, min_value=1, width=_width
            )
            self.embed_dim = dpg.add_input_int(
                label="Embed into dimension (0 = off)", default_value=0, min_value=0, width=_width
            )

            dpg.add_separator()
            dpg.add_text("Model")
            self.cov_type = dpg.add_combo(
                ("isotropic", "diagonal", "mfa", "full"),
                default_value="mfa", label="Model", width=_width
            )
            self.shared_cov = dpg.add_checkbox(label="Shared Covariances", default_value=False)
            self.num_components = dpg.add_input_int(
                label="Number of Components", default_value=15, min_value=1, width=_width
            )
            self.manifold_dim = dpg.add_input_int(
                label="Manifold Dimension (H)", default_value=1, min_value=1, width=_width
            )

            dpg.add_separator()
            dpg.add_button(label="Train", callback=self._on_train, width=-1)

            dpg.add_separator()
            dpg.add_text("Visualization")
            self.draw_points = dpg.add_checkbox(
                label="Draw Points", default_value=True, callback=self._on_viz_change,
            )
            self.draw_means = dpg.add_checkbox(
                label="Draw Means", default_value=True, callback=self._on_viz_change,
            )
            self.visualisation_mode = dpg.add_radio_button(
                ("ellipsoid", "line", "plane", "none"),
                default_value="ellipsoid", callback=self._on_viz_change,
            )

            dpg.add_separator()
            self.status_text = dpg.add_text("No training has been conducted yet.")


    # -------------------- Setup: Plot-Panel + Textur -----------------------
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


    def _on_train(self):
        dpg.set_value(self.status_text, "training is running...")
        self.data, self.projection = make_dataset(
            dpg.get_value(self.data_type),
            dpg.get_value(self.num_points),
            embed_dim=dpg.get_value(self.embed_dim) or None,
        )
        self.pipe = ManifoldPipeline(
            n_components=dpg.get_value(self.num_components),
            latent_dim=dpg.get_value(self.manifold_dim),
            cov_type=dpg.get_value(self.cov_type),
            shared=dpg.get_value(self.shared_cov),
        )
        self.pipe.fit(self.data)

        dpg.set_value(self.status_text, f"objective: {self.pipe.obj:.4f}")
        self._update_plot()


    def _on_viz_change(self):
        self._update_plot()


    # ------------ mouse dragging for 3d plots --------------------
    def _is_3d(self):
        return dpg.get_value(self.data_type) in ("swiss_roll", "torus", "curve_in_3d")

    def _on_mouse_down(self, sender, app_data):
        if not dpg.is_item_hovered(self.image_tag):
            return
        if not self._is_3d() or self.pipe is None:
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

        current_time = time.time()
        if current_time - self._last_render_time > 0.033:  
            self._last_render_time = current_time
            self._update_plot()

    def _on_mouse_up(self, sender, app_data):
        self._dragging = False
        self._last_mouse_pos = None


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


    # ------------------- Plotting ------------------------
    def _update_plot(self):
        if self.pipe is None:
            return

        is_3d = self._is_3d()
        figsize = (self.tex_w / DPI, self.tex_h / DPI)

        fig = Figure(figsize=figsize, dpi=DPI)
        ax = fig.add_subplot(111, projection="3d" if is_3d else None)
        if is_3d:
            ax.view_init(elev=self.elev, azim=self.azim)

        visualize_gmm(
            ax=ax,
            data=self.data,
            means=self.pipe.model.means,
            covariances=self.pipe.model.covariances,
            priors=self.pipe.model.prior,
            projection_matrix=self.projection,
            draw_points=dpg.get_value(self.draw_points),
            visualisation_mode=dpg.get_value(self.visualisation_mode),
            draw_means=dpg.get_value(self.draw_means),
        )

        self._render_to_texture(fig)

    def _render_to_texture(self, fig):
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.asarray(canvas.buffer_rgba(), dtype=np.float32) / 255.0

        if (w, h) != (self.tex_w, self.tex_h):
            self._swap_texture(w, h, data=buf.flatten())
            return

        dpg.set_value(self.texture_tag, buf.flatten())


# TODO: move starting point to main.py
def main():
    dpg.create_context()
    dpg.create_viewport(title="MFA Demo", width=1150, height=820)

    with dpg.window(label="MFA Demo", tag="primary_window"):
        with dpg.tab_bar():
            SyntheticDemoTab(parent=dpg.last_item())

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()

