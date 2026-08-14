import sys
import os

import numpy as np
import dearpygui.dearpygui as dpg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.faces import (
    make_multi_face_dataset, FaceRenderer, FACES, FACTOR_NAMES, FACTOR_UNITS,
)
from visualization.faces import render_samples_frame, render_spline_frame
from visualization.mnist import render_pca_frame, render_persistence_frame
from experiments.face_factor_eval import (
    run_evaluation as run_face_evaluation,
    _spec_label,
    _format_value,
)
from demo.demo_tab import ManifoldDemoTab, _CYAN, _GREY

# One table row per face of the catalogue; the preset fills the first three.
PRESET_FACTORS = ("yaw", "smile", "jaw_open")
DEFAULT_FACTORS = ("yaw", "smile", "jaw_open", "brow_raise", "pucker", "blink")


class FacesDemoTab(ManifoldDemoTab):
    """
    Faces from the ICT morphable model (the catalogue 'data.faces.FACES'), each
    sweeping one generative factor (head rotation, smile, jaw, ...) over its full range.
    """

    name = "faces"
    title = "Faces - DPG Manifold Demo"
    intro = (
        "Rendered faces: every face is a separate component and sweeps one "
        "factor over its full range (rotation in degrees, expressions in "
        "blendshape units), so each face is an open arc. Detected with TDA "
        "(Sec. 5.6)."
    )
    eval_tooltip = (
        "Offline evaluation of the current table: topology (M1), ARI against the "
        "face labels (M4) and the per-face factor error in its own unit (M2), "
        "plus figures, into results/faces/."
    )

    def __init__(self):
        super().__init__()
        self.pixel_mean = None
        self.pixel_std = None
        self.values = None        # ground-truth factor value per frame
        self.colors = None        # position within its own sweep, in [0, 1]
        self.captions = None      # formatted factor value per frame
        self.component_gt = None  # which spec (face) a frame comes from
        self.meta = None
        self.renderer = None
        self.image_shape = None   # resolution the current data was rendered at
        self.row_include = {}
        self.row_factor = {}
        self.row_samples = {}

    # ------------------------------------------------------------------
    # settings panel
    # ------------------------------------------------------------------
    def _build_data_section(self):
        dpg.add_text("Data (faces and their factor)", color=_CYAN)
        dpg.add_separator()
        dpg.add_button(label="Preset: 3 faces (yaw + smile + jaw)",
                       callback=self._preset_three_faces, width=-1)

        with dpg.group(horizontal=True):
            dpg.add_text("use", color=_GREY)
            dpg.add_text("face", color=_GREY)
            dpg.add_text("  factor", color=_GREY)
            dpg.add_text("      samples", color=_GREY)
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
            label="Image size (px)", default_value=128, width=125)
        with dpg.tooltip(self.image_size_in):
            dpg.add_text("Faces are rendered square at this resolution; the data "
                         "dimension is its square. Larger is slower to render.",
                         wrap=260)
        self._add_noise_seed_generate(
            "Pixel noise (std)", 0.0,
            "Std of per-pixel Gaussian noise. Keep it well below the MNIST "
            "setting: an expression moves far fewer pixels than a rotating digit.",
            noise_format="%.3f",
        )

    def _build_model_section(self):
        self._add_model_inputs(
            n_components=90, pca_dim=40,
            n_comp_tooltip="Total charts across all faces. Raise it with the "
                           "number of faces so each component stays well sampled.",
            pca_tooltip="The rotation sweep dominates the leading directions; "
                        "keep enough dimensions for the smaller expression arcs.",
        )

    def _preset_three_faces(self):
        """The configuration of the thesis experiment: three faces, three factors."""
        for row in range(len(FACES)):
            use = row < len(PRESET_FACTORS)
            factor = PRESET_FACTORS[row] if use else DEFAULT_FACTORS[row]
            dpg.set_value(self.row_include[row], use)
            dpg.set_value(self.row_factor[row], factor)
            dpg.set_value(self.row_samples[row], 120)
        dpg.set_value(self.image_size_in, 128)
        dpg.set_value(self.noise_in, 0.0)
        dpg.set_value(self.seed_in, 0)
        dpg.set_value(self.n_comp_in, 90)
        dpg.set_value(self.pca_dim_in, 40)
        dpg.set_value(self.lambda_in, 30.0)
        dpg.set_value(self.k_distance_in, 4)
        dpg.set_value(self.interp_w_in, 3.0)

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def _current_specs(self):
        """Read the per-face table into a list of dataset specs."""
        return [{"face": row,
                 "factor": dpg.get_value(self.row_factor[row]),
                 "samples": int(dpg.get_value(self.row_samples[row]))}
                for row in range(len(FACES))
                if dpg.get_value(self.row_include[row])]

    def _load_data(self, specs, report):
        image_size = int(dpg.get_value(self.image_size_in))
        if self.renderer is None or self.renderer.image_size != image_size:
            self.renderer = FaceRenderer(image_size=image_size)

        (self.X, self.values, self.component_gt, self.meta,
         self.pixel_mean, self.pixel_std) = make_multi_face_dataset(
            specs, image_size=image_size, add_noise=dpg.get_value(self.noise_in),
            random_state=self._seed_value(), renderer=self.renderer,
            progress=lambda done, total: dpg.set_value(
                self.data_lbl,
                f"Rendering faces... {100 * done // total}%  ({done}/{total} frames)"),
        )
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

        self.data_label = ", ".join(_spec_label(s) for s in specs)
        return (f"{len(self.X)} frames from {len(specs)} face(s): "
                f"{self.data_label}.")

    # ------------------------------------------------------------------
    # detection and evaluation
    # ------------------------------------------------------------------
    def _member_ids(self):
        return self.component_gt

    def _member_name(self, value):
        return _spec_label(self.specs[int(value)])

    def _training_summary(self):
        n_components, n_loops, n_paths = self._topology_counts()
        txt = f"training done. H0: {n_components} component(s), H1: {n_paths} arc(s)"
        if n_loops:
            txt += f" (+{n_loops} loop(s), none expected)"
        return txt + "."

    def _evaluate(self, specs, seed, progress):
        _, out_dir = run_face_evaluation(
            specs=specs,
            image_size=int(dpg.get_value(self.image_size_in)),
            noise=dpg.get_value(self.noise_in),
            n_components=dpg.get_value(self.n_comp_in),
            pca_dim=dpg.get_value(self.pca_dim_in),
            lambda_aniso=dpg.get_value(self.lambda_in),
            n_neighbors=dpg.get_value(self.k_distance_in),
            interp_tangent_weight=dpg.get_value(self.interp_w_in),
            seed=seed, progress=progress,
        )
        return out_dir

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _extra_plot_state(self):
        return {
            "cmap": "viridis",
            "color_label": "position along sweep",
            "color_scale": (0.0, 1.0),
            "factor_label": self._selected_curve_label(),
        }

    def _draw_state(self, fig, state):
        mode = state["mode"]
        if mode == "samples":
            render_samples_frame(fig, self.X, self.captions,
                                 f"Samples - {state['title']}",
                                 self.pixel_mean, self.pixel_std, self.image_shape)
        elif mode == "pca":
            render_pca_frame(fig, state, self.pca_data, self.colors, self.exp,
                             self.pca_basis, self._spline_point)
        elif mode == "spline":
            render_spline_frame(fig, state, self.pca_data, self.colors, self.exp,
                                self.pca_basis, self.spline, self._spline_point,
                                self.pixel_mean, self.pixel_std, self.image_shape)
        elif mode == "persistence":
            render_persistence_frame(fig, self.diagram, self.curves,
                                     self.selected_component)
