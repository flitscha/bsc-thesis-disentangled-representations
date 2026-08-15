import sys
import os

import dearpygui.dearpygui as dpg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.mnist_rotation import make_multi_rotation_dataset
from visualization.mnist import (
    render_samples_frame, render_pca_frame, render_spline_frame,
    render_persistence_frame,
)
from experiments.multi_factor_eval import run_evaluation as run_multi_evaluation
from experiments.mnist_rotation_eval import run_evaluation as run_single_evaluation
from demo.demo_tab import ManifoldDemoTab, _CYAN, _GREY, _RED


class MNISTDemoTab(ManifoldDemoTab):
    """
    Rotating MNIST digits: every digit is a separate component, a full 0-360
    sweep closes into a loop and a partial sweep stays an open arc.
    """

    name = "mnist"
    title = "MNIST Rotation - DPG Manifold Demo"
    panel_width = 360
    wrap = 320
    intro = (
        "Rotating MNIST digits: each digit is a separate component, a full 0-360 "
        "sweep a loop, a partial sweep an arc. Detected with TDA."
    )
    eval_tooltip = (
        "Offline evaluation of the current table. One selected digit runs the "
        "single-component experiment (capacity + denoising, TDA vs. MST) into "
        "results/mnist_rotation/. Two or more digits run the multi-factor "
        "evaluation (topology M1, ARI M4, per-component angle M2 + figures) into "
        "results/mnist_multi/."
    )

    def __init__(self):
        super().__init__()
        self.pixel_mean = None
        self.pixel_std = None
        self.angles = None
        self.digit_id = None
        self.meta = None
        self.multi_include = {}
        self.multi_start = {}
        self.multi_end = {}
        self.multi_samples = {}

    # ------------------------------------------------------------------
    # settings panel
    # ------------------------------------------------------------------
    def _build_data_section(self):
        dpg.add_text("Data (digits to rotate)", color=_CYAN)
        dpg.add_separator()
        dpg.add_button(label="Preset: multi-factor (1, 3, 9 loop + 6 arc)",
                       callback=self._multi_preset_clean, width=-1)

        with dpg.group(horizontal=True):
            dpg.add_text("use", color=_GREY)
            dpg.add_text("digit", color=_GREY)
            dpg.add_text("  start", color=_GREY)
            dpg.add_text("   end", color=_GREY)
            dpg.add_text("  samples", color=_GREY)
        for d in range(10):
            with dpg.group(horizontal=True):
                self.multi_include[d] = dpg.add_checkbox(default_value=(d == 6))
                dpg.add_text(f"  {d}  ")
                self.multi_start[d] = dpg.add_input_int(
                    default_value=0, width=70, step=0)
                self.multi_end[d] = dpg.add_input_int(
                    default_value=360, width=70, step=0)
                self.multi_samples[d] = dpg.add_input_int(
                    default_value=360, width=70, step=0)

        dpg.add_spacer(height=6)
        self._add_noise_seed_generate(
            "Pixel noise (std)", 0.25,
            "Std of uniform per-pixel, per-image Gaussian noise",
            seed_default=-1,
        )

    def _build_model_section(self):
        self._add_model_inputs(
            n_components=24, pca_dim=20,
            n_comp_tooltip="Total charts across all digits. With several digits "
                           "raise this (e.g. 90 for three) so each component is "
                           "well sampled.",
        )

    def _multi_preset_clean(self):
        """
        Load the multi-digit experiment described in the thesis: digits 1, 3, 9 as
        full loops and 6 as a half arc, with its fitted parameters.
        """
        for d in range(10):
            dpg.set_value(self.multi_include[d], d in (1, 3, 6, 9))
            dpg.set_value(self.multi_start[d], 0)
            dpg.set_value(self.multi_end[d], 180 if d == 6 else 360)
            dpg.set_value(self.multi_samples[d], 360)
        dpg.set_value(self.n_comp_in, 200)
        dpg.set_value(self.pca_dim_in, 60)
        dpg.set_value(self.noise_in, 0.15)
        dpg.set_value(self.seed_in, 0)
        dpg.set_value(self.lambda_in, 30.0)
        dpg.set_value(self.k_distance_in, 4)
        dpg.set_value(self.h0_factor_in, 0.0)
        dpg.set_value(self.h1_factor_in, 0.0)
        dpg.set_value(self.interp_w_in, 3.0)

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def _current_specs(self):
        """Read the per-digit table into a list of dataset specs."""
        return [{"digit": d,
                 "start": float(dpg.get_value(self.multi_start[d])),
                 "end": float(dpg.get_value(self.multi_end[d])),
                 "samples": int(dpg.get_value(self.multi_samples[d]))}
                for d in range(10) if dpg.get_value(self.multi_include[d])]

    def _load_data(self, specs, report):
        (self.X, self.angles, self.digit_id, self.meta,
         self.pixel_mean, self.pixel_std) = make_multi_rotation_dataset(
            specs, add_noise=dpg.get_value(self.noise_in),
            random_state=self._seed_value(),
        )
        self.data_label = ",".join(str(s["digit"]) for s in specs)
        kinds = ", ".join(
            f"{s['digit']}({'loop' if s['start'] == 0 and s['end'] == 360 else 'arc'})"
            for s in specs)
        return f"{len(self.X)} pts from {len(specs)} digit(s): {kinds}."

    # ------------------------------------------------------------------
    # detection and evaluation
    # ------------------------------------------------------------------
    def _member_ids(self):
        return self.digit_id

    def _member_name(self, value):
        return f"digit {int(value)}"

    def _training_summary(self):
        n_components, n_loops, n_paths = self._topology_counts()
        txt = f"training done. H0: {n_components} component(s), H1: {n_loops} loop(s)"
        if n_paths:
            txt += f" (+{n_paths} open path(s))"
        return txt + "."

    def _evaluate(self, specs, seed, progress):
        """
        One selected digit runs the single-component experiment, two or more the
        multi-factor one. The single-digit case is defined for a full loop only.
        """
        if len(specs) == 1:
            spec = specs[0]
            if spec["start"] != 0.0 or spec["end"] != 360.0:
                self._status(self.eval_lbl,
                             "Single-digit evaluation covers a full 0-360 loop only.",
                             _RED)
                return None
            _, out_dir = run_single_evaluation(
                digit=int(spec["digit"]),
                n_angles=int(spec.get("samples", 360)),
                noise=dpg.get_value(self.noise_in),
                n_components=dpg.get_value(self.n_comp_in),
                pca_dim=dpg.get_value(self.pca_dim_in),
                lambda_aniso=dpg.get_value(self.lambda_in),
                n_neighbors=dpg.get_value(self.k_distance_in),
                interp_tangent_weight=dpg.get_value(self.interp_w_in),
                seed=seed, progress=progress,
            )
            return out_dir

        _, out_dir = run_multi_evaluation(
            specs=specs,
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
    def _format_t(self, t):
        return f"t = {t:.3f}  ({t * 360:.0f}°)"

    def _draw_state(self, fig, state):
        mode = state["mode"]
        if mode == "samples":
            render_samples_frame(fig, self.X, self.angles, state["title"],
                                 self.pixel_mean, self.pixel_std)
        elif mode == "pca":
            render_pca_frame(fig, state, self.pca_data, self.angles, self.exp,
                             self.pca_basis, self._spline_point)
        elif mode == "spline":
            render_spline_frame(fig, state, self.pca_data, self.angles, self.exp,
                                self.pca_basis, self.spline, self._spline_point,
                                self.pixel_mean, self.pixel_std)
        elif mode == "persistence":
            render_persistence_frame(fig, self.diagram, self.curves,
                                     self.selected_component)
