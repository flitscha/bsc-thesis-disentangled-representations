import sys
import os

import dearpygui.dearpygui as dpg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.mocap import make_multi_motion_dataset, MOTIONS, FACTOR_UNIT
from visualization.mocap import render_samples_frame, render_spline_frame
from visualization.mnist import render_pca_frame, render_persistence_frame
from experiments.mocap_eval import (
    run_evaluation as run_mocap_evaluation,
    _spec_label,
    _format_value,
)
from demo.demo_tab import ManifoldDemoTab, _CYAN, _GREY, _RED

# The verified configuration of the thesis experiment.
PRESET_MOTIONS = ("walk", "run", "wave", "sit_down")


class MocapDemoTab(ManifoldDemoTab):
    """
    Motion capture poses from the CMU database (the catalogue 'data.mocap.MOTIONS'):
    every motion of the same person is a separate component, periodic motions
    (walking, running) are loops, one-way movements (sitting down) are arcs.
    The observations are 93-dimensional poses, not images.
    """

    name = "mocap"
    title = "Mocap - DPG Manifold Demo"
    intro = (
        "Motion capture (CMU database, one subject): an observation is a pose, "
        "not an image. Every motion is a separate component; a periodic one "
        "(walking, running) closes into a loop, a one-way movement (sitting "
        "down) stays an arc. The factor is the progress within one repetition, "
        "in percent. Detected with TDA."
    )
    eval_tooltip = (
        "Offline evaluation of the current table: topology (M1), ARI against the "
        "motion labels (M4) and the per-motion progress error in percent (M2), "
        "plus figures, into results/mocap/."
    )
    h0_hint = (
        "Needed once one motion sits much further away from the others than they "
        "do from each other (try 2.2)."
    )

    def __init__(self):
        super().__init__()
        self.pose_mean = None
        self.pose_std = None
        self.values = None        # ground-truth progress per frame, in percent
        self.colors = None        # progress in [0, 1], for colouring
        self.captions = None      # formatted progress per frame
        self.component_gt = None  # which spec (motion) a frame comes from
        self.meta = None
        self.row_include = {}
        self.row_samples = {}

    # ------------------------------------------------------------------
    # settings panel
    # ------------------------------------------------------------------
    def _build_data_section(self):
        dpg.add_text("Data (motions of one person)", color=_CYAN)
        dpg.add_separator()
        dpg.add_button(label="Preset: walk + run + wave + sit down",
                       callback=self._preset_motions, width=-1)

        with dpg.group(horizontal=True):
            dpg.add_text("use", color=_GREY)
            dpg.add_text("  motion", color=_GREY)
            dpg.add_text("            type", color=_GREY)
            dpg.add_text("   samples", color=_GREY)
        for row, motion in enumerate(MOTIONS):
            with dpg.group(horizontal=True):
                self.row_include[row] = dpg.add_checkbox(
                    default_value=motion["name"] in PRESET_MOTIONS)
                dpg.add_text(f" {motion['name']:<15}")
                dpg.add_text(f"{motion['kind']:<5}", color=_GREY)
                self.row_samples[row] = dpg.add_input_int(
                    default_value=100, width=62, step=0)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text(motion["description"], wrap=260)

        dpg.add_spacer(height=6)
        self._add_noise_seed_generate(
            "Joint noise (m)", 0.0,
            "Std of Gaussian noise on the joint coordinates, in metres "
            "(0.01 = 1 cm). The recordings already carry the natural variation "
            "between repetitions.",
            noise_step=0.005, noise_format="%.3f",
        )

    def _build_model_section(self):
        self._add_model_inputs(
            n_components=100, pca_dim=50,
            n_comp_tooltip="Total charts across all motions. Raise it with the "
                           "number of motions so each component stays well covered.",
            pca_tooltip="A pose has 93 coordinates but a single motion moves in "
                        "far fewer directions.",
        )

    def _build_distance_section(self):
        super()._build_distance_section(
            k=4,
            k_tooltip="Neighbors of the k-NN graph whose shortest paths give the "
                      "geodesic distance. Small values keep the recordings "
                      "apart: with k = 4 the graph falls into one piece per "
                      "motion, which is what lets the automatic H0 rule "
                      "separate them.",
        )

    def _preset_motions(self):
        """The configuration of the thesis experiment: two loops and two arcs."""
        for row, motion in enumerate(MOTIONS):
            dpg.set_value(self.row_include[row], motion["name"] in PRESET_MOTIONS)
            dpg.set_value(self.row_samples[row], 100)
        dpg.set_value(self.noise_in, 0.0)
        dpg.set_value(self.seed_in, 0)
        dpg.set_value(self.n_comp_in, 100)
        dpg.set_value(self.pca_dim_in, 50)
        dpg.set_value(self.lambda_in, 30.0)
        dpg.set_value(self.k_distance_in, 4)
        dpg.set_value(self.h0_factor_in, 0.0)
        dpg.set_value(self.h1_factor_in, 0.0)
        dpg.set_value(self.interp_w_in, 3.0)

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def _current_specs(self):
        """Read the per-motion table into a list of dataset specs."""
        return [{"motion": motion["name"],
                 "samples": int(dpg.get_value(self.row_samples[row]))}
                for row, motion in enumerate(MOTIONS)
                if dpg.get_value(self.row_include[row])]

    def _load_data(self, specs, report):
        try:
            (self.X, self.values, self.component_gt, self.meta,
             self.pose_mean, self.pose_std) = make_multi_motion_dataset(
                specs, add_noise=dpg.get_value(self.noise_in),
                random_state=self._seed_value(),
                progress=lambda done, total: dpg.set_value(
                    self.data_lbl, f"Reading recordings... ({done}/{total} motions)"),
            )
        except FileNotFoundError as exc:
            self._status(self.data_lbl, f"{exc}", _RED)
            return None

        self.colors = self.values / 100.0  # colour by progress in the repetition
        self.captions = [f"{self.meta[i]['motion']} {_format_value(v)}"
                         for v, i in zip(self.values, self.component_gt)]

        self.data_label = ", ".join(_spec_label(s) for s in specs)
        return (f"{len(self.X)} poses from {len(specs)} motion(s): "
                f"{self.data_label}.")

    # ------------------------------------------------------------------
    # detection and evaluation
    # ------------------------------------------------------------------
    def _member_ids(self):
        return self.component_gt

    def _member_name(self, value):
        return _spec_label(self.specs[int(value)])

    def _evaluate(self, specs, seed, progress):
        _, out_dir = run_mocap_evaluation(
            specs=specs,
            noise=dpg.get_value(self.noise_in),
            n_components=dpg.get_value(self.n_comp_in),
            pca_dim=dpg.get_value(self.pca_dim_in),
            lambda_aniso=dpg.get_value(self.lambda_in),
            n_neighbors=dpg.get_value(self.k_distance_in),
            interp_tangent_weight=dpg.get_value(self.interp_w_in),
            h0_persistence_factor=dpg.get_value(self.h0_factor_in),
            h1_persistence_factor=dpg.get_value(self.h1_factor_in),
            seed=seed, progress=progress,
        )
        return out_dir

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _extra_plot_state(self):
        return {
            "cmap": "viridis",
            "color_label": f"progress ({FACTOR_UNIT} of the repetition)",
            "color_scale": (0.0, 1.0),
            "factor_label": self._selected_curve_label(),
        }

    def _draw_state(self, fig, state):
        mode = state["mode"]
        if mode == "samples":
            render_samples_frame(fig, self.X, self.captions,
                                 f"Samples - {state['title']}",
                                 self.pose_mean, self.pose_std)
        elif mode == "pca":
            render_pca_frame(fig, state, self.pca_data, self.colors, self.exp,
                             self.pca_basis, self._spline_point)
        elif mode == "spline":
            render_spline_frame(fig, state, self.pca_data, self.colors, self.exp,
                                self.pca_basis, self.spline, self._spline_point,
                                self.pose_mean, self.pose_std)
        elif mode == "persistence":
            render_persistence_frame(fig, self.diagram, self.curves,
                                     self.selected_component)
