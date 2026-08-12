"""
Single-run evaluation orchestrator.

'evaluate_run' takes a fitted+detected pipeline, a ground-truth factor and the
data, computes the metrics (M1 topology, M2 angle error, M3 reconstruction decomposition, M4 ARI),
renders the figures and writes everything to one self-contained run directory:

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
                pipe, X_input, X_target, factor, t, periodic, curve_index,
                image_shape, pixel_mean, pixel_std, n_strip, n_recovery_strip))

        for name, fig in figs.items():
            fig.savefig(os.path.join(run_dir, name), bbox_inches="tight")

    return summary, run_dir


def _strip_figures(
    pipe, X_input, X_target, factor, t, periodic, ci,
    image_shape, pixel_mean, pixel_std, n_strip, n_recovery_strip
):
    """
    Two image strips. Both pick the observations nearest to a grid of
    ground-truth factor values and reconstruct each at its own encoded
    coordinate t

    fig_reconstruction : reconstruction / denoising view
        capacity  : true  | recon
        denoising : noisy | recon
    fig_recovery_strip : the visual factor residual
        capacity / denoising : true | recon
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

    def pick(n):
        """Observations nearest to an n-point ground-truth factor grid."""
        if periodic:
            grid = np.linspace(0.0, 360.0, n, endpoint=False)
        else:
            grid = np.linspace(factor.min(), factor.max(), n)
        idx = [_nearest_index(factor, g, periodic) for g in grid]
        recon = pipe.reconstruct(spline(np.asarray(t)[idx]))
        return idx, recon, [f"{g:.0f}" for g in grid]

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

    # visual factor-residual strip: top row is the actual input (noisy under
    # noise), so the denoising by the reconstruction stays visible
    idx, recon, labels = pick(n_recovery_strip)
    top_label = "noisy" if X_target is not None else "true"
    rows = [(top_label, np.stack([X_input[i] for i in idx])), ("recon", recon)]
    recovery_strip = make(rows, labels, "Reconstruction at the estimated angle")

    return {"fig_reconstruction.pdf": recon_strip,
            "fig_recovery_strip.pdf": recovery_strip}

