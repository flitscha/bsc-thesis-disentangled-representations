"""
Several rotating MNIST digits at once, the mixed multi-structure case.

Different digits form separate connected components. A digit swept over the full circle is a
loop, a partial sweep an open arc, so both kinds of structure show up in the same dataset.

Lighter than the single-digit evaluation: TDA only, one noise level, and only the structural
results, since that is where the multi-factor story lives.

    M1  topology     expected against detected components, and the type of each
    M2  angle error  per component, in degrees
    M4  ARI          how well the detected components agree with the digit labels

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
    persistence_tag,
)
from experiments.eval import figures as F

IMAGE_SHAPE = (30, 30)


def _is_loop_spec(s):
    """A spec is a loop exactly when its digit is swept over the full circle."""
    return float(s.get("start", 0.0)) == 0.0 and float(s.get("end", 360.0)) == 360.0


def _spec_tag(specs):
    """Compact folder name describing the setup, such as '3loop_7loop_2arc'."""
    parts = []
    for s in specs:
        parts.append(f"{int(s['digit'])}{'loop' if _is_loop_spec(s) else 'arc'}")
    return "_".join(parts)


def _per_component_angle_errors(t, cid, angles, digit_id, curves):
    """
    Align each detected curve to the true angle and report the residual in degrees.

    Whether the loop or the arc alignment is used follows the type the curve was detected as. The
    digit label is a majority vote over the curve's members, so a curve that wrongly merged two
    digits shows up as a low purity together with a large error.
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
            err = align_arc(t_j, ang_j)["error"] # factor is in degrees

        out.append({
            "curve": j, "type": curve["type"], "n": n,
            "majority_digit": majority, "purity": purity,
            "angle_mean_deg": float(err.mean()),
            "angle_median_deg": float(np.median(err)),
            "angle_max_deg": float(err.max()),
        })
    return out


def _loop_entries(t, cid, angles, digit_id, curves):
    """One entry per detected loop: its members, majority digit, median angle error and
    persistence."""
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
    Split the detected loops into correct and extra ones, following the matching.

    A loop counts as correct if it was picked to represent one of the expected digits. Every other
    loop is extra: either a shortcut on a digit that already has its loop, or a loop wrongly
    closed on a digit that should have stayed an arc.
    """
    correct = sorted((e for e in entries if e["curve"] in matched_ids),
                     key=lambda e: e["majority_digit"])
    extra = sorted((e for e in entries if e["curve"] not in matched_ids),
                   key=lambda e: -e["persistence"])
    return correct, extra


def _loop_brief(e):
    """A JSON-safe summary of one loop entry, without the raw arrays."""
    return {"curve": e["curve"], "majority_digit": e["majority_digit"],
            "persistence": e["persistence"], "n": e["n"],
            "angle_median_deg": e["angle_median_deg"]}


def _reconstruct_along(pipe, curve, n):
    """n reconstructions spread evenly along a curve. An arc includes both ends, a loop does not."""
    is_loop = curve["type"] == "loop"
    t_grid = np.linspace(0.0, 1.0, n, endpoint=not is_loop)
    imgs = pipe.reconstruct(np.asarray(curve["spline"](t_grid)))
    return t_grid, imgs


def _nearest_member(m_angles, target, periodic):
    """The member whose true angle comes closest to `target`."""
    if periodic:
        d = np.abs((m_angles - target + 180.0) % 360.0 - 180.0)
    else:
        d = np.abs(m_angles - target)
    return int(np.argmin(d))


def _match_specs_to_curves(cid, digit_id, curves, specs):
    """
    Assign the expected digits to the detected curves one to one, maximising the total overlap.

    A (digit, curve) pair scores how many of that digit's observations are members of that curve,
    and the Hungarian algorithm then maximises the sum, so two digits cannot fight over the same
    loop. A pair with zero overlap is never assigned, so a missing loop stays a miss instead of
    being forced onto an unrelated curve, and a surplus digit stays unmatched.

    Returns the matches and the ids of the curves no digit claimed.
    """
    valid = [j for j, c in enumerate(curves) if c.get("spline") is not None]
    overlap = np.zeros((len(specs), len(valid)), dtype=int)
    for i, s in enumerate(specs):
        d = int(s["digit"])
        for jj, j in enumerate(valid):
            overlap[i, jj] = int(np.sum((cid == j) & (digit_id == d)))

    assign = {} # spec index -> (curve id, size of the overlap)
    if valid and len(specs):
        rows, cols = linear_sum_assignment(-overlap) # negate, since the solver minimises
        for i, jj in zip(rows, cols):
            if overlap[i, jj] > 0: # never assign a curve that shares no samples at all
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
    Build one strip for a matched (digit, curve) pair.

    The top row holds the true digit's own frames on a grid of true angles. They are taken purely
    from that digit's ground-truth observations, so the row always shows a single digit no matter
    how the detection carved up the space.

    The bottom row is the matched curve, reconstructed at the coordinate that aligns to each grid
    angle. That alignment is fitted on this digit's members of the curve, so the two rows depict
    the same true rotation.
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
    h0_persistence_factor=0.0, h1_persistence_factor=0.0,
    seed=0, results_root=None, save=True, progress=None,
):
    """
    Run the multi-factor evaluation for one configuration and optionally save it.

    The defaults reproduce the stored run in `results/mnist_multi/1loop_3loop_6arc_9loop_seed0/`.

    A positive `h0_persistence_factor` or `h1_persistence_factor` replaces the automatic detection
    rule by an explicit threshold at that multiple of the median H0 merge scale; 0 keeps the
    automatic rule.

    Returns the summary dict and the directory it was written to.
    """
    def say(msg):
        if progress is not None:
            progress(msg)

    if specs is None: # three full loops and one half arc, the mixed case
        specs = [{"digit": 1, "start": 0, "end": 360},
                 {"digit": 3, "start": 0, "end": 360},
                 {"digit": 6, "start": 0, "end": 180},
                 {"digit": 9, "start": 0, "end": 360}]

    # Write the count actually used into every spec, so that the saved config says what was
    # really generated. Otherwise `samples_per` stays in there as a fallback that a per-spec
    # "samples" may have overridden.
    specs = [{**s, "samples": int(s.get("samples", samples_per))} for s in specs]

    tag = (f"{_spec_tag(specs)}"
           f"{persistence_tag(h0_persistence_factor, h1_persistence_factor)}"
           f"_seed{seed}")
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
    pipe.apply_persistence_thresholds(h0_persistence_factor, h1_persistence_factor)
    t, cid = pipe.transform(X)
    comp_id = component_labels(pipe, X) # H0 component per observation

    # Match each expected digit to the curve that covers it best. This drives both the sweep
    # figures, one row per expected digit, and the metrics.
    matches, extra_curve_ids = _match_specs_to_curves(cid, digit_id, pipe.curves_, specs)
    matched_ids = {m["curve"] for m in matches if m["curve"] is not None}

    # metrics
    m1 = topology_report(pipe.structure_, expected_n, expected_types,
                         component_labels=comp_id) # only count components that hold data
    m4 = discrete_ari(comp_id, digit_id) # detected components against the true digits
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
            "interp_tangent_weight": interp_tangent_weight,
            "h0_persistence_factor": h0_persistence_factor,
            "h1_persistence_factor": h1_persistence_factor, "seed": seed,
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
        # a low-dimensional embedding for the scatter plots
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        Z3 = X @ Vt[:3].T
        title = f"{expected_n} rotating digits"

        # turn a reconstruction back into a displayable [0, 1] image
        def to_image(v):
            return np.clip(np.asarray(v) * pstd + pmean, 0.0, 1.0).reshape(IMAGE_SHAPE)

        # One global loop numbering, where 1 is the most persistent loop. It is the only name a
        # loop has: the same number marks it in the persistence diagram and in the sweep figures.
        # Arcs have no marker in the diagram, so they get their own numbering after the loops.
        loop_ids = sorted(
            (j for j, c in enumerate(pipe.curves_) if c.get("type") == "loop"),
            key=lambda j: -float(pipe.curves_[j].get("persistence", 0.0)))
        loop_number = {j: k for k, j in enumerate(loop_ids, start=1)}

        def _curve_label(j, arc_counter):
            """The sort key and label of curve j. Loops come first, then the arcs."""
            if pipe.curves_[j].get("type") == "loop":
                return loop_number[j], f"loop {loop_number[j]}"
            arc_counter[0] += 1
            return 10_000 + arc_counter[0], f"arc {arc_counter[0]}"

        # One sweep per expected digit: its own ground-truth frames over the matched model curve,
        # 8 frames for a loop and 4 for an arc, on a grid of true angles.
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

        # every leftover curve becomes a model-only strip, under the same numbering
        extra_entries, arc_counter = [], [0]
        for j in extra_curve_ids:
            _, imgs = _reconstruct_along(pipe, pipe.curves_[j], 8)
            key, label = _curve_label(j, arc_counter)
            extra_entries.append((key, (label, imgs, None)))
        extra_strips = [s for _, s in sorted(extra_entries, key=lambda e: e[0])]

        # label each loop in the persistence diagram with its global number
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
    """Compact readable summary for logging or a status label."""
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

