"""
Motion capture: several motions of one person at once.

The only experiment on data that are not images. An observation is a pose, the 93 coordinates of
31 joints, and a component is one motion. Walking and running return to the pose they started
from and are therefore loops. Waving is an arc, since the arm travels out and back along the same
path, and so is sitting down.

The ground-truth factor is the progress within one repetition, in percent. It is cut from a
physical signal, the foot gap, the hand position or the hip height, never from the fitted model.

    M1  topology        expected against detected components, and the type of each
    M2  progress error  per component, in percent of its repetition
    M4  ARI             how well the detected components agree with the motion labels

Results are written to results/mocap/<tag>/.
"""

import os
import json
import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.mocap import make_multi_motion_dataset, FACTOR_UNIT, motion_spec
from core.pipeline import ManifoldPipeline
from experiments.eval import (
    align_loop, align_arc, topology_report, discrete_ari, component_labels,
    persistence_tag,
)
from experiments.eval import figures as F
from visualization.mocap import draw_vector

# Two periodic motions and two one-way movements, so two loops and two arcs. With the defaults
# below, and `n_neighbors = 4` in particular, the automatic H0 rule separates all four on its own.
# `h0_persistence_factor` is the explicit threshold that can replace that rule.
DEFAULT_SPECS = [
    {"motion": "walk"},
    {"motion": "run"},
    {"motion": "wave"},
    {"motion": "sit_down"},
]

# a pose vector is 31 joints by 3 coordinates, so there is no image shape here
POSE_SHAPE = (31, 3)


def _kind(spec):
    """Whether the motion of this spec is expected to be a loop or an arc."""
    return motion_spec(str(spec["motion"]))["kind"]


def _expected_type(spec):
    """The curve type `topology_report` expects, where an arc is traced as a path."""
    return "loop" if _kind(spec) == "loop" else "path"


def _spec_label(spec):
    return str(spec["motion"])


def _spec_tag(specs):
    """Compact folder name describing the setup, such as 'walk_run_wave'."""
    return "_".join(_spec_label(s) for s in specs)


def _format_value(value):
    return f"{value:.0f} %"


def _progress_error(t, values, kind):
    """
    Residual of the recovered coordinate against the true progress, in percent.

    The progress of a loop is periodic, so it is read as an angle with 100 % standing for 360
    degrees and gets the circular alignment. An arc gets the affine one. In neither case is the
    speed calibrated out.
    """
    if kind == "loop":
        return align_loop(t, np.asarray(values) * 3.6)["error_deg"] / 3.6
    return align_arc(t, values)["error"]


def _match_specs_to_curves(cid, component_gt, curves, specs):
    """
    Assign the expected motions to the detected curves one to one, maximising the total overlap.

    A curve backs at most one motion and a pair with zero overlap is never assigned, so every
    curve that nobody claims comes back as an extra one.
    """
    valid = [j for j, c in enumerate(curves) if c.get("spline") is not None]
    overlap = np.zeros((len(specs), len(valid)), dtype=int)
    for i in range(len(specs)):
        for jj, j in enumerate(valid):
            overlap[i, jj] = int(np.sum((cid == j) & (component_gt == i)))

    assign = {}
    if valid and len(specs):
        rows, cols = linear_sum_assignment(-overlap)
        for i, jj in zip(rows, cols):
            if overlap[i, jj] > 0:
                assign[i] = (valid[jj], int(overlap[i, jj]))

    matches, used = [], set()
    for i, spec in enumerate(specs):
        j, n = assign.get(i, (None, 0))
        matches.append({
            "spec": i, "motion": _spec_label(spec), "kind": _kind(spec),
            "curve": j, "n_overlap": n,
        })
        if j is not None:
            used.add(j)
    return matches, sorted(j for j in valid if j not in used)


def _per_curve_progress_errors(t, cid, values, component_gt, curves, specs):
    """
    Align each detected curve to the true progress of its majority motion, reported in percent.

    A curve that swallowed a second motion shows up as a low purity together with a large error.
    """
    out = []
    for j, curve in enumerate(curves):
        mask = cid == j
        n = int(mask.sum())
        if n == 0:
            continue
        labels, counts = np.unique(component_gt[mask], return_counts=True)
        majority = int(labels[np.argmax(counts)])
        # align against the type the curve was detected as, not the one we expected
        kind = "loop" if curve["type"] == "loop" else "arc"
        err = _progress_error(t[mask], values[mask], kind)
        out.append({
            "curve": j, "type": curve["type"], "n": n,
            "majority_motion": _spec_label(specs[majority]),
            "expected_kind": _kind(specs[majority]),
            "unit": FACTOR_UNIT,
            "purity": float(counts.max() / n),
            "error_mean": float(err.mean()),
            "error_median": float(np.median(err)),
            "error_max": float(err.max()),
        })
    return out


def _ground_truth_strip(pipe, curve, match, cid, values, t, component_gt, X,
                        n_frames=6):
    """
    Build one strip for a matched (motion, curve) pair.

    The top row holds that motion's own input poses, at the ground-truth progress values closest
    to an evenly spaced grid over the repetition. The bottom row is the matched curve,
    reconstructed at the coordinate the alignment maps the same grid to.
    """
    i = match["spec"]
    is_loop = curve["type"] == "loop"
    grid = np.linspace(0.0, 100.0, n_frames, endpoint=not is_loop)

    own = np.where(component_gt == i)[0]
    top = X[[own[int(np.argmin(np.abs(values[own] - g)))] for g in grid]]

    on_curve = (cid == match["curve"]) & (component_gt == i)
    if is_loop:
        t_grid = align_loop(t[on_curve], values[on_curve] * 3.6)["t_of_theta"](grid * 3.6)
    else:
        t_grid = np.clip(
            align_arc(t[on_curve], values[on_curve])["t_of_factor"](grid), 0.0, 1.0)
    recon = pipe.reconstruct(np.asarray(curve["spline"](t_grid)))
    return grid, top, recon


def _reconstruct_along(pipe, curve, n):
    """n reconstructions spread evenly along a curve."""
    t_grid = np.linspace(0.0, 1.0, n, endpoint=curve["type"] != "loop")
    return pipe.reconstruct(np.asarray(curve["spline"](t_grid)))


def run_evaluation(
    *, specs=None, samples_per=100, noise=0.0, n_components=100, pca_dim=50,
    lambda_aniso=30.0, n_neighbors=4, interp_tangent_weight=3.0,
    h0_persistence_factor=0.0, h1_persistence_factor=0.0,
    seed=0, results_root=None, save=True, progress=None,
):
    """
    Run the motion capture evaluation for one configuration and save it.

    The defaults reproduce the stored run in `results/mocap/walk_run_wave_sit_down_seed0/`.

    A positive `h0_persistence_factor` or `h1_persistence_factor` replaces the automatic detection
    rule by an explicit threshold at that multiple of the median H0 merge scale; 0 keeps the
    automatic rule. The H0 threshold matters here as soon as one motion sits much further from the
    others than they do from each other, because the largest-gap rule then only sees that one gap
    and undercounts the components.

    Returns the summary dict and the directory it was written to.
    """
    def say(msg):
        if progress is not None:
            progress(msg)

    specs = [{**s, "samples": int(s.get("samples", samples_per))}
             for s in (specs if specs is not None else DEFAULT_SPECS)]
    tag = (f"{_spec_tag(specs)}"
           f"{persistence_tag(h0_persistence_factor, h1_persistence_factor)}"
           f"_seed{seed}")
    expected_n = len(specs)
    expected_types = [_expected_type(s) for s in specs]

    say("reading recordings ...")
    X, values, component_gt, meta, pose_mean, pose_std = make_multi_motion_dataset(
        specs, samples_per=samples_per, add_noise=noise, random_state=seed,
        progress=lambda done, total: say(f"reading recordings ... {done}/{total}"))

    say(f"fitting MFA ({n_components} components) ...")
    pipe = ManifoldPipeline(
        n_components=n_components, latent_dim=1, cov_type="mfa", shared=False,
        pca_dim=pca_dim if pca_dim and pca_dim > 0 else None,
        lambda_aniso=lambda_aniso, n_neighbors=n_neighbors, detection="tda",
        interp_tangent_weight=interp_tangent_weight, seed=seed,
    )
    pipe.fit(X)

    say("detecting structure (TDA) ...")
    pipe.detect()
    pipe.apply_persistence_thresholds(h0_persistence_factor, h1_persistence_factor)
    t, cid = pipe.transform(X)
    comp_id = component_labels(pipe, X) # the detected component of every observation

    matches, extra_curve_ids = _match_specs_to_curves(
        cid, component_gt, pipe.curves_, specs)

    m1 = topology_report(pipe.structure_, expected_n, expected_types,
                         component_labels=comp_id)
    m4 = discrete_ari(comp_id, component_gt) # detected components against the true motions
    per_curve = _per_curve_progress_errors(t, cid, values, component_gt,
                                           pipe.curves_, specs)

    summary = {
        "experiment": "mocap",
        "tag": tag,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "specs": specs, "samples_per": samples_per, "noise": noise,
            "n_components": n_components, "pca_dim": pca_dim,
            "lambda_aniso": lambda_aniso, "n_neighbors": n_neighbors,
            "interp_tangent_weight": interp_tangent_weight,
            "h0_persistence_factor": h0_persistence_factor,
            "h1_persistence_factor": h1_persistence_factor, "seed": seed,
            "n_frames": [m["n_frames"] for m in meta],
        },
        "metrics": {
            "topology": m1,
            "ari": m4,
            "per_component_factor": per_curve,
            "matching": [
                {"motion": m["motion"], "kind": m["kind"],
                 "curve": m["curve"], "n_overlap": m["n_overlap"]}
                for m in matches
            ],
            "n_extra_curves": len(extra_curve_ids),
        },
    }

    run_dir = None
    if save:
        root = results_root or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
        run_dir = os.path.abspath(os.path.join(root, "mocap", tag))
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        np.savez(os.path.join(run_dir, "arrays.npz"),
                 t=t, curve_id=cid, component_id=comp_id,
                 values=values, component_gt=component_gt,
                 pose_mean=pose_mean, pose_std=pose_std)

        say("rendering figures ...")
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        Z3 = X @ Vt[:3].T
        title = f"{expected_n} motions of one person"
        motion_labels = np.array([_spec_label(specs[i]) for i in component_gt])

        def draw(ax, vector):
            draw_vector(ax, vector, pose_mean, pose_std, lw=1.4)

        # one strip per matched motion: its own poses above the model curve
        correct = []
        for m in matches:
            j = m["curve"]
            if j is None or pipe.curves_[j].get("spline") is None:
                continue
            grid, top, recon = _ground_truth_strip(
                pipe, pipe.curves_[j], m, cid, values, t, component_gt, X)
            correct.append({
                "label": m["motion"],
                "angles": grid,
                "labels": [_format_value(g) for g in grid],
                "true_imgs": top, "recon_imgs": recon,
            })

        # the curves no motion claimed become model-only rows
        extra = [(f"extra {k}", _reconstruct_along(pipe, pipe.curves_[j], 6), None)
                 for k, j in enumerate(extra_curve_ids, start=1)]

        figs = {
            "fig_persistence.pdf": F.plot_persistence(
                pipe.structure_["diagram"],
                title="Persistence diagram (H0 = motions, H1 = loops)"),
            "fig_components_scatter.pdf": F.plot_component_scatter(
                Z3[:, :2], motion_labels, comp_id, title=title,
                label_name="ground-truth motion"),
            "fig_components_scatter_3d.pdf": F.plot_component_scatter_3d(
                Z3, motion_labels, comp_id, title=title,
                label_name="ground-truth motion"),
            "fig_sweep_correct.pdf": F.plot_recovery_strips(
                correct, POSE_SHAPE, draw=draw, cell_size=(1.0, 1.15),
                title="Factor recovery",
                show_captions=False, row_gap=0.0, component_gap=0.2),
            "fig_sweep_extra.pdf": F.plot_reconstruction_strips(
                extra, POSE_SHAPE, draw=draw, cell_size=(1.0, 1.6),
                title="Unmatched components",
                empty_note="No unmatched components detected."),
        }
        for name, fig in figs.items():
            fig.savefig(os.path.join(run_dir, name), bbox_inches="tight")

        say(f"done. results in {run_dir}")

    return summary, run_dir


def _format_summary(summary):
    """Compact readable summary for logging or a status label."""
    m = summary["metrics"]
    topo = m["topology"]
    lines = [
        f"H0 components: {'OK' if topo['components_match'] else 'MISMATCH'} "
        f"(detected {topo['n_components']}, expected {topo['n_components_expected']})",
        f"H1: {'OK' if topo['paths_match'] and topo['loops_match'] else 'MISMATCH'} "
        f"(detected {topo['n_loops']} loop(s) + {topo['n_paths']} arc(s), "
        f"expected {topo['n_loops_expected']} loop(s) + {topo['n_paths_expected']} arc(s))",
        f"ARI (H0 component vs motion): {m['ari']:.3f}",
    ]
    for mm in m["matching"]:
        target = "no curve matched" if mm["curve"] is None else \
            f"curve {mm['curve']} (overlap n={mm['n_overlap']})"
        lines.append(f"  {mm['motion']} [{mm['kind']}] -> {target}")
    for c in m["per_component_factor"]:
        lines.append(
            f"  curve {c['curve']} [{c['type']}] ~{c['majority_motion']} "
            f"(purity {c['purity']:.2f}, n={c['n']})  "
            f"error med={c['error_median']:.1f} max={c['error_max']:.1f} {c['unit']}")
    if m["n_extra_curves"]:
        lines.append(f"unmatched curves: {m['n_extra_curves']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    summary, out = run_evaluation(progress=print)
    print("\n" + _format_summary(summary))
    print("saved to", out)
