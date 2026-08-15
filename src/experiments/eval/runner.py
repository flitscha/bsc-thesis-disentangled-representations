"""
Single-run evaluation orchestrator.

'evaluate_run' takes a fitted and detected pipeline, a ground-truth factor and
the data, computes M1-M4, renders the figures and writes everything into one
self-contained run directory:

    results/<experiment>/<run_name>/
        summary.json   hyperparameters + scalar metrics
        arrays.npz     raw per-observation arrays
        fig_*.pdf      figures
"""

import os
import json
import datetime

import numpy as np

from experiments.eval.metrics import topology_report, angle_error, discrete_ari
from experiments.eval.reconstruct import reconstruction_errors
from experiments.eval import figures as F


def persistence_tag(h0_factor=0.0, h1_factor=0.0):
    """
    Filename fragment for explicit H0/H1 persistence thresholds, e.g. '_h0f2.2'.

    The thresholds are part of a run's identity: the same data with and without
    them are two different results, not one overwriting the other. Empty string
    if both are 0 (the automatic rules), which keeps the tags of the default
    runs unchanged.
    """
    parts = ""
    if h0_factor and h0_factor > 0:
        parts += f"_h0f{h0_factor:g}"
    if h1_factor and h1_factor > 0:
        parts += f"_h1f{h1_factor:g}"
    return parts


def _project_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _jsonify(obj):
    """Recursively convert numpy types / arrays to plain JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if callable(obj):
        return None
    return obj


def _nearest_index(factor, target, periodic):
    """Index of the observation whose factor is closest to 'target'."""
    if periodic:
        d = np.abs((factor - target + 180.0) % 360.0 - 180.0)
    else:
        d = np.abs(factor - target)
    return int(np.argmin(d))


def evaluate_run(
    pipe, X_input, factor, *, kind="loop", image_shape=None,
    pixel_mean=None, pixel_std=None, X_target=None, classes=None, expected_n=None,
    expected_types=None, experiment="experiment", run_name=None,
    results_root=None, curve_index=0, n_strip=4, n_recovery_strip=8,
    extra=None, make_figures=True
):
    """
    Evaluate one fitted+detected pipeline against a ground-truth factor and save
    the results. Returns (summary_dict, run_dir).
    """
    if pipe.curves_ is None:
        pipe.detect()
    factor = np.asarray(factor, dtype=float)
    periodic = kind == "loop"

    # encode + metrics
    t, cid = pipe.transform(X_input)
    m1 = topology_report(pipe.structure_, expected_n, expected_types)
    m2 = angle_error(t, factor, kind=kind)
    m3 = reconstruction_errors(pipe, X_input, X_target)
    m4 = discrete_ari(cid, classes) if classes is not None else None

    al = m2["alignment"]
    if periodic:
        factor_of_t = al["theta_of_t"]
        align_summary = {"s": al["s"], "delta_deg": al["delta_deg"]}
    else:
        factor_of_t = al["factor_of_t"]
        align_summary = {"a": al["a"], "b": al["b"]}
    factor_pred = factor_of_t(t)

    # run directory
    run_name = run_name or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_root = results_root or os.path.join(_project_root(), "results")
    run_dir = os.path.join(results_root, experiment, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # summary + raw arrays
    summary = {
        "experiment": experiment,
        "run_name": run_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "params": pipe.get_params(),
        "extra": extra or {},
        "metrics": {
            "topology": m1,
            "angle_error": {"mean": m2["mean"], "median": m2["median"],
                            "max": m2["max"], **align_summary},
            "reconstruction": {k: float(np.mean(m3[k])) for k in m3},
            "ari": m4,
        },
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(_jsonify(summary), fh, indent=2)

    np.savez(
        os.path.join(run_dir, "arrays.npz"),
        t=t, curve_id=cid, factor=factor, factor_pred=factor_pred,
        angle_error=m2["error"], **{k: m3[k] for k in m3},
    )

    # figures
    if make_figures:
        flabel = "angle (deg)" if periodic else "factor"
        figs = {
            "fig_reconstruction_decomp.pdf": F.plot_reconstruction_decomposition(
                factor, m3, factor_label=flabel),
            "fig_factor_residual.pdf": F.plot_factor_residual(
                factor % 360.0 if periodic else factor,
                factor_pred % 360.0 if periodic else factor_pred,
                periodic=periodic, factor_label=flabel),
        }
        if pipe.structure_.get("diagram"):
            figs["fig_persistence.pdf"] = F.plot_persistence(pipe.structure_["diagram"])

        if image_shape is not None:
            figs.update(_strip_figures(
                pipe, X_input, X_target, factor, t, periodic, curve_index, al,
                image_shape, pixel_mean, pixel_std, n_strip, n_recovery_strip))

        for name, fig in figs.items():
            fig.savefig(os.path.join(run_dir, name), bbox_inches="tight")

    return summary, run_dir


def _strip_figures(
    pipe, X_input, X_target, factor, t, periodic, ci, alignment,
    image_shape, pixel_mean, pixel_std, n_strip, n_recovery_strip
):
    """
    Two image strips over a grid of ground-truth factor values. They differ in
    what the bottom row is, and the difference matters:

    fig_reconstruction : reconstruction / denoising view. The bottom row is the
        round trip of each observation -- encode it, then decode it at its own
        coordinate t. This is the only strip in which an observation enters the
        model.
            capacity  : true  | recon
            denoising : noisy | recon
    fig_recovery_strip : factor recovery, the same figure the multi-structure
        experiments build. The bottom row is the decoder evaluated at the
        coordinate the alignment maps each *ground-truth* grid value to, so the
        ground truth never enters the model: only the two alignment freedoms
        (direction and offset for a loop, the affine map for an arc) connect the
        two rows. A wrong angle would show up as a wrongly rotated digit.
            capacity / denoising : true / noisy | model
    """
    spline = pipe.curves_[ci]["spline"]

    # invert the standardization (v * std + mean) to display a [0, 1] image
    if pixel_mean is not None:
        pm = np.asarray(pixel_mean)
        ps = np.asarray(pixel_std) if pixel_std is not None else 1.0
        def to_image(v):
            return np.clip(np.asarray(v) * ps + pm, 0.0, 1.0).reshape(image_shape)
    else:
        def to_image(v):
            return np.asarray(v).reshape(image_shape)

    def grid_of(n):
        """An n-point grid over the ground-truth factor range."""
        if periodic:
            return np.linspace(0.0, 360.0, n, endpoint=False)
        return np.linspace(factor.min(), factor.max(), n)

    def nearest_observations(grid):
        """Indices of the observations whose true factor is closest to the grid."""
        return [_nearest_index(factor, g, periodic) for g in grid]

    def pick(n):
        """Observations nearest to the grid, each reconstructed at its own t."""
        grid = grid_of(n)
        idx = nearest_observations(grid)
        recon = pipe.reconstruct(spline(np.asarray(t)[idx]))
        return idx, recon, [f"{g:.0f}" for g in grid]

    def decode_at(grid):
        """
        Decoder output at the curve coordinate that aligns to each grid value.

        This is the generative direction: a ground-truth factor value goes in, an
        image comes out, without the corresponding observation ever being seen.
        """
        if periodic:
            t_grid = np.mod(alignment["t_of_theta"](grid), 1.0)
        else:
            t_grid = np.clip(alignment["t_of_factor"](grid), 0.0, 1.0)
        return pipe.reconstruct(np.asarray(spline(t_grid)))

    def make(rows, labels, title):
        return F.plot_image_strip(rows, image_shape, col_labels=labels,
                                  to_image=to_image, title=title)

    truth = X_input if X_target is None else X_target

    # reconstruction / denoising strip
    idx, recon, labels = pick(n_strip)
    if X_target is None:
        rows = [("true", np.stack([truth[i] for i in idx])), ("recon", recon)]
        recon_strip = make(rows, labels, "Reconstruction")
    else:
        rows = [("noisy", np.stack([X_input[i] for i in idx])), ("recon", recon)]
        recon_strip = make(rows, labels, "Denoising")

    # factor-recovery strip: top row is the actual input at each grid value, the
    # bottom row the decoder driven by that same grid value (never by the image)
    grid = grid_of(n_recovery_strip)
    idx = nearest_observations(grid)
    top_label = "noisy" if X_target is not None else "true"
    rows = [(top_label, np.stack([X_input[i] for i in idx])),
            ("model", decode_at(grid))]
    recovery_strip = make(rows, [f"{g:.0f}" for g in grid],
                          "Angle recovery" if periodic else "Factor recovery")

    return {"fig_reconstruction.pdf": recon_strip,
            "fig_recovery_strip.pdf": recovery_strip}

