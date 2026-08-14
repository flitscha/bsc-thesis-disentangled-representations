"""
Multi-factor experiment: several rotating MNIST digits at once.

Different digits form separate connected components (H0); a digit swept over the
full circle is a loop (H1), a partial sweep an open arc. This is the mixed
multi-structure case of §5.6.

Lighter than the single-digit evaluation: TDA only, one noise regime, and only
the structural results, which is where the multi-factor story lives.

    M1  topology     expected vs. detected components and per-component type
    M2  angle error  per component in degrees (loop / arc alignment)
    M4  ARI          agreement of detected components with the digit labels

Results are written to results/mnist_multi/<tag>/.
"""

import os
import json
import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.mnist_rotation import make_multi_rotation_dataset
from core.pipeline import ManifoldPipeline
from experiments.eval import (
    align_loop, align_arc, topology_report, discrete_ari, component_labels,
)
from experiments.eval import figures as F

IMAGE_SHAPE = (30, 30)


def _is_loop_spec(s):
    """A ground-truth spec is a loop iff its digit is swept over the full circle."""
    return float(s.get("start", 0.0)) == 0.0 and float(s.get("end", 360.0)) == 360.0


def _spec_tag(specs):
    """Compact filename tag like '3loop_7loop_2arc' describing the setup."""
    parts = []
    for s in specs:
        parts.append(f"{int(s['digit'])}{'loop' if _is_loop_spec(s) else 'arc'}")
    return "_".join(parts)


def _per_component_angle_errors(t, cid, angles, digit_id, curves):
    """
    Per detected curve: align its coordinate to the true angle (loop or arc
    depending on the curve type) and report the residual in degrees. The digit
    label is the majority vote over the curve's members, so a curve that wrongly
    merged two digits is visible as a split label / large error.
    """
    out = []
    for j, curve in enumerate(curves):
        mask = cid == j
        n = int(mask.sum())
        if n == 0:
            continue
        t_j, ang_j, dig_j = t[mask], angles[mask], digit_id[mask]
        labels, counts = np.unique(dig_j, return_counts=True)
        majority = int(labels[np.argmax(counts)])
        purity = float(counts.max() / n)

        if curve["type"] == "loop":
            err = align_loop(t_j, ang_j)["error_deg"]
        else:
            err = align_arc(t_j, ang_j)["error"]  # factor is in degrees

        out.append({
            "curve": j, "type": curve["type"], "n": n,
            "majority_digit": majority, "purity": purity,
            "angle_mean_deg": float(err.mean()),
            "angle_median_deg": float(np.median(err)),
            "angle_max_deg": float(err.max()),
        })
    return out


def _loop_entries(t, cid, angles, digit_id, curves):
    """One scalar entry per detected loop: members, majority digit, loop-aligned
    median angle error and H1 persistence."""
    entries = []
    for j, curve in enumerate(curves):
        if curve.get("type") != "loop":
            continue
        mask = cid == j
        n = int(mask.sum())
        if n == 0:
            continue
        vals, counts = np.unique(digit_id[mask], return_counts=True)
        err = align_loop(t[mask], angles[mask])["error_deg"]
        entries.append({
            "curve": j, "n": n, "majority_digit": int(vals[np.argmax(counts)]),
            "persistence": float(curve.get("persistence", 0.0)),
            "angle_median_deg": float(np.median(err)),
        })
    return entries


def _classify_loops(entries, matched_ids):
    """
    Split detected loops into 'correct' and 'extra' by the spec->curve matching:
    a loop is correct if it was picked to represent some expected digit (its id
    is in `matched_ids`), every other detected loop is extra (a shortcut on the
    same digit, or a loop spuriously closed on an arc digit).
    """
    correct = sorted((e for e in entries if e["curve"] in matched_ids),
                     key=lambda e: e["majority_digit"])
    extra = sorted((e for e in entries if e["curve"] not in matched_ids),
                   key=lambda e: -e["persistence"])
    return correct, extra


def _loop_brief(e):
    """JSON-safe scalar summary of a loop entry (drops the raw arrays)."""
    return {"curve": e["curve"], "majority_digit": e["majority_digit"],
            "persistence": e["persistence"], "n": e["n"],
            "angle_median_deg": e["angle_median_deg"]}


def _reconstruct_along(pipe, curve, n):
    """n reconstructions evenly along a curve (loops periodic; arcs incl. ends)."""
    is_loop = curve["type"] == "loop"
    t_grid = np.linspace(0.0, 1.0, n, endpoint=not is_loop)
    imgs = pipe.reconstruct(np.asarray(curve["spline"](t_grid)))
    return t_grid, imgs


def _nearest_member(m_angles, target, periodic):
    """Index (into the members) whose true angle is closest to `target`."""
    if periodic:
        d = np.abs((m_angles - target + 180.0) % 360.0 - 180.0)
    else:
        d = np.abs(m_angles - target)
    return int(np.argmin(d))


def _match_specs_to_curves(cid, digit_id, curves, specs):
    """
    Assign expected digits to detected curves one-to-one, maximising total overlap.

    The score of a (digit, curve) pair is how many of that digit's observations
    are members of that curve; the Hungarian algorithm then maximises the sum, so
    two digits cannot fight over the same loop. Pairs with zero overlap are never
    assigned, so a missing loop is a miss rather than forced onto an unrelated
    curve, and surplus digits stay unmatched (curve=None).

    Returns (matches, extra_curve_ids), where a match is
    {digit, type ('loop'|'arc'), span, curve (int|None), n_overlap} and the extra
    ids are the curves no digit claimed.
    """
    valid = [j for j, c in enumerate(curves) if c.get("spline") is not None]
    overlap = np.zeros((len(specs), len(valid)), dtype=int)
    for i, s in enumerate(specs):
        d = int(s["digit"])
        for jj, j in enumerate(valid):
            overlap[i, jj] = int(np.sum((cid == j) & (digit_id == d)))

    assign = {}  # spec index -> (curve id, overlap)
    if valid and len(specs):
        rows, cols = linear_sum_assignment(-overlap)  # maximise total overlap
        for i, jj in zip(rows, cols):
            if overlap[i, jj] > 0:  # never assign a curve that shares no samples
                assign[i] = (valid[jj], int(overlap[i, jj]))

    matches, used = [], set()
    for i, s in enumerate(specs):
        j, n = assign.get(i, (None, 0))
        matches.append({
            "digit": int(s["digit"]),
            "type": "loop" if _is_loop_spec(s) else "arc",
            "span": (float(s.get("start", 0.0)), float(s.get("end", 360.0))),
            "curve": j, "n_overlap": n,
        })
        if j is not None:
            used.add(j)
    extra = sorted(j for j in valid if j not in used)
    return matches, extra


def _ground_truth_strip(pipe, curve, match, cid, angles, t, digit_id, X):
    """
    Build one 'correct' strip for a matched (digit, curve) pair.

    Top row: the true digit's own frames on a true-angle grid -- sampled purely
    from that digit's ground-truth observations, so it is always a single digit
    regardless of how the detection carved up the space.

    Bottom row: the matched curve reconstructed at the coordinate that aligns to
    each grid angle. The t<->angle alignment (direction + offset for a loop, the
    affine map for an arc) is fitted on this digit's members of the curve, so the
    two rows depict the same true rotation. Returns (grid_angles, top, recon).
    """
    d = match["digit"]
    is_loop = match["type"] == "loop"

    grid = (np.linspace(0.0, 360.0, 8, endpoint=False) if is_loop
            else np.linspace(match["span"][0], match["span"][1], 4))

    # top: nearest true-angle frame among *this digit's* ground-truth samples
    gt = np.where(digit_id == d)[0]
    gt_ang = angles[gt]
    top_imgs = X[[gt[_nearest_member(gt_ang, g, is_loop)] for g in grid]]

    # bottom: reconstruct the matched curve at the aligned coordinate
    on_curve = (cid == match["curve"]) & (digit_id == d)
    if is_loop:
        t_grid = np.mod(align_loop(t[on_curve], angles[on_curve])["t_of_theta"](grid), 1.0)
    else:
        t_grid = np.clip(align_arc(t[on_curve], angles[on_curve])["t_of_factor"](grid), 0.0, 1.0)
    recon_imgs = pipe.reconstruct(np.asarray(curve["spline"](t_grid)))
    return grid, top_imgs, recon_imgs


def run_evaluation(
    *, specs=None, samples_per=360, noise=0.15, n_components=200,
    pca_dim=60, lambda_aniso=30.0, n_neighbors=4, interp_tangent_weight=3.0,
    seed=0, results_root=None, save=True, progress=None,
):
    """
    Run the multi-factor evaluation for one configuration and (optionally) save.

    The defaults are the configuration of the run reported in the thesis
    (`results/mnist_multi/1loop_3loop_6arc_9loop_seed0/`).

    Returns (summary_dict, run_dir).
    """
    def say(msg):
        if progress is not None:
            progress(msg)

    if specs is None:  # three full loops + one half arc (the mixed case)
        specs = [{"digit": 1, "start": 0, "end": 360},
                 {"digit": 3, "start": 0, "end": 360},
                 {"digit": 6, "start": 0, "end": 180},
                 {"digit": 9, "start": 0, "end": 360}]

    # pin the count actually used into every spec, so the saved config says what
    # was generated instead of leaving `samples_per` as a fallback that a
    # per-spec "samples" may have overridden
    specs = [{**s, "samples": int(s.get("samples", samples_per))} for s in specs]

    tag = f"{_spec_tag(specs)}_seed{seed}"
    expected_n = len(specs)
    expected_types = ["loop" if _is_loop_spec(s) else "path" for s in specs]

    say("generating data ...")
    X, angles, digit_id, _, pmean, pstd = make_multi_rotation_dataset(
        specs, samples_per=samples_per, add_noise=noise, random_state=seed)

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
    t, cid = pipe.transform(X)
    comp_id = component_labels(pipe, X)  # H0 component per observation

    # match each expected digit to the detected curve that best covers it; this
    # drives both the 'correct' sweeps (one per expected digit) and the metrics.
    matches, extra_curve_ids = _match_specs_to_curves(cid, digit_id, pipe.curves_, specs)
    matched_ids = {m["curve"] for m in matches if m["curve"] is not None}

    # metrics
    m1 = topology_report(pipe.structure_, expected_n, expected_types,
                         component_labels=comp_id)  # count data-backed H0 components
    m4 = discrete_ari(comp_id, digit_id)  # H0 components vs. ground-truth digits
    per_comp = _per_component_angle_errors(t, cid, angles, digit_id, pipe.curves_)
    loop_entries = _loop_entries(t, cid, angles, digit_id, pipe.curves_)
    correct_loops, extra_loops = _classify_loops(loop_entries, matched_ids)

    summary = {
        "experiment": "mnist_multi",
        "tag": tag,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "specs": specs, "samples_per": samples_per, "noise": noise,
            "n_components": n_components, "pca_dim": pca_dim,
            "lambda_aniso": lambda_aniso, "n_neighbors": n_neighbors,
            "interp_tangent_weight": interp_tangent_weight, "seed": seed,
        },
        "metrics": {
            "topology": m1,
            "ari": m4,
            "per_component_angle": per_comp,
            "matching": [
                {"digit": m["digit"], "type": m["type"],
                 "curve": m["curve"], "n_overlap": m["n_overlap"]}
                for m in matches
            ],
            "loops": {
                "correct": [_loop_brief(e) for e in correct_loops],
                "extra": [_loop_brief(e) for e in extra_loops],
            },
        },
    }

    run_dir = None
    if save:
        root = results_root or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
        run_dir = os.path.abspath(os.path.join(root, "mnist_multi", tag))
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        np.savez(os.path.join(run_dir, "arrays.npz"),
                 t=t, curve_id=cid, component_id=comp_id,
                 angles=angles, digit_id=digit_id)

        say("rendering figures ...")
        # low-dim embedding for the scatters (top principal directions of X)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        Z3 = X @ Vt[:3].T
        title = f"{expected_n} rotating digits"

        # de-standardize a reduced-space reconstruction back to a [0, 1] image
        def to_image(v):
            return np.clip(np.asarray(v) * pstd + pmean, 0.0, 1.0).reshape(IMAGE_SHAPE)

        # one global loop numbering, most persistent = 1. It is the only label
        # for a loop: the same number tags its H1 triangle in the persistence
        # diagram and its row in the sweep figures. Arcs (H0 paths, no H1
        # triangle) get their own "arc k" numbering, sorted after the loops.
        loop_ids = sorted(
            (j for j, c in enumerate(pipe.curves_) if c.get("type") == "loop"),
            key=lambda j: -float(pipe.curves_[j].get("persistence", 0.0)))
        loop_number = {j: k for k, j in enumerate(loop_ids, start=1)}

        def _curve_label(j, arc_counter):
            """(sort key, label) for curve j; loops by number, arcs after."""
            if pipe.curves_[j].get("type") == "loop":
                return loop_number[j], f"loop {loop_number[j]}"
            arc_counter[0] += 1
            return 10_000 + arc_counter[0], f"arc {arc_counter[0]}"

        # correct sweeps: one per expected digit, its own ground-truth frames
        # over the matched model curve, 8 frames for loops / 4 for arcs on a
        # true-angle grid. Rows are ordered by the global loop number.
        correct_entries, arc_counter = [], [0]
        for m in matches:
            j = m["curve"]
            if j is None or pipe.curves_[j].get("spline") is None:
                continue
            grid, top_imgs, recon = _ground_truth_strip(
                pipe, pipe.curves_[j], m, cid, angles, t, digit_id, X)
            key, label = _curve_label(j, arc_counter)
            correct_entries.append((key, {
                "label": label, "angles": grid,
                "true_imgs": top_imgs, "recon_imgs": recon}))
        correct_components = [c for _, c in sorted(correct_entries, key=lambda e: e[0])]

        # leftover curves become model-only strips, same numbering scheme.
        extra_entries, arc_counter = [], [0]
        for j in extra_curve_ids:
            _, imgs = _reconstruct_along(pipe, pipe.curves_[j], 8)
            key, label = _curve_label(j, arc_counter)
            extra_entries.append((key, (label, imgs, None)))
        extra_strips = [s for _, s in sorted(extra_entries, key=lambda e: e[0])]

        # H1-triangle labels: the bare global loop number at each triangle.
        h1_labels = [
            {"birth": pipe.curves_[j]["birth"], "death": pipe.curves_[j]["death"],
             "label": str(loop_number[j])}
            for j in loop_ids
        ]

        figs = {
            "fig_persistence.pdf": F.plot_persistence(
                pipe.structure_["diagram"],
                title="Persistence diagram (H0 = components, H1 = loops)",
                h1_labels=h1_labels),
            "fig_components_scatter.pdf": F.plot_component_scatter(
                Z3[:, :2], digit_id, comp_id, title=title),
            "fig_components_scatter_3d.pdf": F.plot_component_scatter_3d(
                Z3, digit_id, comp_id, title=title),
            "fig_sweep_correct.pdf": F.plot_recovery_strips(
                correct_components, IMAGE_SHAPE, to_image=to_image,
                title="Angle recovery"),
            "fig_sweep_extra.pdf": F.plot_reconstruction_strips(
                extra_strips, IMAGE_SHAPE, to_image=to_image,
                title="Spurious loops",
                empty_note="No spurious loops detected."),
        }
        for name, fig in figs.items():
            fig.savefig(os.path.join(run_dir, name), bbox_inches="tight")

        say(f"done. results in {run_dir}")

    return summary, run_dir


def _format_summary(summary):
    """Compact human-readable one-block summary for logging / a status label."""
    m = summary["metrics"]
    topo = m["topology"]

    def _n_paths_exp(t):
        return t["n_paths_expected"] if t["n_paths_expected"] is not None else 0

    h1_ok = "OK" if (topo["loops_match"] and topo["paths_match"]) else "MISMATCH"
    h1_line = (f"H1 loops: {h1_ok} (detected {topo['n_loops']} loops"
               + (f" + {topo['n_paths']} paths" if topo['n_paths'] else "")
               + f", expected {topo['n_loops_expected']} loops"
               + (f" + {_n_paths_exp(topo)} paths" if _n_paths_exp(topo) else "")
               + ")")
    lines = [
        f"H0 components: {'OK' if topo['components_match'] else 'MISMATCH'} "
        f"(detected {topo['n_components']}, expected {topo['n_components_expected']})",
        h1_line,
        f"ARI (H0 component vs digit): {m['ari']:.3f}",
    ]
    for mm in m.get("matching", []):
        if mm["curve"] is None:
            lines.append(f"  digit {mm['digit']} [{mm['type']}] -> no curve matched")
        else:
            lines.append(
                f"  digit {mm['digit']} [{mm['type']}] -> curve {mm['curve']} "
                f"(overlap n={mm['n_overlap']})")
    for c in m["per_component_angle"]:
        lines.append(
            f"  curve {c['curve']} [{c['type']}] digit~{c['majority_digit']} "
            f"(purity {c['purity']:.2f}, n={c['n']})  "
            f"angle med={c['angle_median_deg']:.1f} deg  max={c['angle_max_deg']:.1f}")
    loops = m.get("loops", {})
    correct, extra = loops.get("correct", []), loops.get("extra", [])
    lines.append(f"loops: {len(correct)} correct, {len(extra)} extra (spurious)")
    for e in extra:
        lines.append(
            f"  extra loop {e['curve']} digit~{e['majority_digit']} "
            f"pers={e['persistence']:.0f} n={e['n']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    summary, out = run_evaluation(progress=print)
    print("\n" + _format_summary(summary))
    print("saved to", out)
