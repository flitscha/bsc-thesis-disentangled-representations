"""
Face experiment: several faces, each varying a different generative factor.

Every face is a different identity of the ICT morphable face model, sweeping one
factor over its full range: one head turns (yaw in degrees), another smiles,
another opens its jaw. The identities are far apart in pixel space and never
meet, so each face is a separate connected component and, since no factor closes,
an open arc.

    M1  topology      expected vs. detected components and per-component type
    M2  factor error  per component, in its own unit (degrees or units)
    M4  ARI           agreement of detected components with the face labels

Results are written to results/faces/<tag>/.
"""

import os
import json
import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.faces import make_multi_face_dataset, FACES, FACTOR_UNITS
from core.pipeline import ManifoldPipeline
from experiments.eval import align_arc, topology_report, discrete_ari, component_labels
from experiments.eval import figures as F

DEFAULT_SPECS = [
    {"face": 0, "factor": "yaw"},
    {"face": 1, "factor": "smile"},
    {"face": 2, "factor": "jaw_open"},
]


def _unit(spec):
    return FACTOR_UNITS.get(spec.get("factor"), "units")


def _face_name(spec):
    return FACES[int(spec["face"])]["name"]


def _format_value(value, unit):
    """Column caption for a factor value: degrees stay integral, units get 2 dp."""
    return f"{value:.0f}°" if unit == "deg" else f"{value:.2f}"


def _spec_tag(specs):
    """Compact filename tag like 'Ayaw_Bsmile_Cjaw_open'."""
    return "_".join(f"{_face_name(s)}{s['factor']}" for s in specs)


def _spec_label(spec):
    """Human-readable name of a component, e.g. 'face A: yaw'."""
    return f"face {_face_name(spec)}: {spec['factor']}"


def _match_specs_to_curves(cid, component_gt, curves, specs):
    """
    Assign expected faces to detected curves one-to-one, maximising the total
    ground-truth overlap.
    A curve backs at most one face, a pair with zero overlap is never assigned,
    and every curve nobody claims is 'extra' (spurious).

    Returns (matches, extra_curve_ids) with each match
        {spec, face, factor, span, curve (int|None), n_overlap}.
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
            "spec": i, "face": _face_name(spec),
            "factor": str(spec["factor"]),
            "span": (float(spec["start"]), float(spec["end"])),
            "unit": _unit(spec), "curve": j, "n_overlap": n,
        })
        if j is not None:
            used.add(j)
    return matches, sorted(j for j in valid if j not in used)


def _per_curve_factor_errors(t, cid, values, component_gt, curves, specs):
    """
    Per detected curve: align its coordinate to the true factor of its majority
    face (affine, so only offset and scale are quotiented out) and report the residual
    in that face's unit.
    A curve that merged two faces shows up as a low purity together with a large error.
    """
    out = []
    for j, curve in enumerate(curves):
        mask = cid == j
        n = int(mask.sum())
        if n == 0:
            continue
        labels, counts = np.unique(component_gt[mask], return_counts=True)
        majority = int(labels[np.argmax(counts)])
        err = align_arc(t[mask], values[mask])["error"]
        out.append({
            "curve": j, "type": curve["type"], "n": n,
            "majority_face": majority,
            "face": _face_name(specs[majority]),
            "factor": str(specs[majority]["factor"]),
            "unit": _unit(specs[majority]),
            "purity": float(counts.max() / n),
            "error_mean": float(err.mean()),
            "error_median": float(np.median(err)),
            "error_max": float(err.max()),
        })
    return out


def _ground_truth_strip(pipe, curve, match, cid, values, t, component_gt, X, n_frames=5):
    """
    One strip for a matched (face, curve) pair.

    Top row: that face's own input frames, taken at the ground-truth values
    closest to an evenly spaced grid over its factor range.
    Bottom row: the matched curve reconstructed at the coordinate that aligns to
    the same grid with the affine t <-> factor map fitted on this face's members of the curve.
    """
    i = match["spec"]
    grid = np.linspace(match["span"][0], match["span"][1], n_frames)

    own = np.where(component_gt == i)[0]
    top = X[[own[int(np.argmin(np.abs(values[own] - g)))] for g in grid]]

    on_curve = (cid == match["curve"]) & (component_gt == i)
    t_grid = np.clip(align_arc(t[on_curve], values[on_curve])["t_of_factor"](grid),
                     0.0, 1.0)
    recon = pipe.reconstruct(np.asarray(curve["spline"](t_grid)))
    return grid, top, recon


def _reconstruct_along(pipe, curve, n):
    """n reconstructions evenly along a curve."""
    t_grid = np.linspace(0.0, 1.0, n, endpoint=curve["type"] != "loop")
    return pipe.reconstruct(np.asarray(curve["spline"](t_grid)))


def run_evaluation(
    *, specs=None, samples_per=120, image_size=128, noise=0.0, n_components=90,
    pca_dim=40, lambda_aniso=30.0, n_neighbors=4, interp_tangent_weight=3.0,
    seed=0, results_root=None, save=True, progress=None,
):
    """
    Run the face evaluation for one configuration and save it.

    The defaults are the configuration of the run reported in the thesis
    (`results/faces/Ayaw_Bsmile_Cjaw_open_seed0/`).

    Returns (summary_dict, run_dir).
    """
    def say(msg):
        if progress is not None:
            progress(msg)

    specs = [{**s, "samples": int(s.get("samples", samples_per))}
             for s in (specs if specs is not None else DEFAULT_SPECS)]
    image_shape = (int(image_size), int(image_size))
    tag = f"{_spec_tag(specs)}_seed{seed}"
    expected_n = len(specs)
    expected_types = ["path"] * expected_n

    say("rendering faces ...")
    X, values, component_gt, meta, pmean, pstd = make_multi_face_dataset(
        specs, samples_per=samples_per, image_size=image_size,
        add_noise=noise, random_state=seed,
        progress=lambda done, total: say(f"rendering faces ... {100 * done // total}%"))

    specs = [{**s, "start": m["start"], "end": m["end"]} for s, m in zip(specs, meta)]

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
    comp_id = component_labels(pipe, X) # H0 component per observation

    matches, extra_curve_ids = _match_specs_to_curves(
        cid, component_gt, pipe.curves_, specs)

    m1 = topology_report(pipe.structure_, expected_n, expected_types,
                         component_labels=comp_id)
    m4 = discrete_ari(comp_id, component_gt) # H0 components vs. faces
    per_curve = _per_curve_factor_errors(t, cid, values, component_gt,
                                         pipe.curves_, specs)

    summary = {
        "experiment": "faces",
        "tag": tag,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "specs": specs, "samples_per": samples_per, "image_size": image_size,
            "noise": noise, "n_components": n_components, "pca_dim": pca_dim,
            "lambda_aniso": lambda_aniso, "n_neighbors": n_neighbors,
            "interp_tangent_weight": interp_tangent_weight, "seed": seed,
        },
        "metrics": {
            "topology": m1,
            "ari": m4,
            "per_component_factor": per_curve,
            "matching": [
                {"face": m["face"], "factor": m["factor"],
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
        run_dir = os.path.abspath(os.path.join(root, "faces", tag))
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        np.savez(os.path.join(run_dir, "arrays.npz"),
                 t=t, curve_id=cid, component_id=comp_id,
                 values=values, component_gt=component_gt)

        say("rendering figures ...")
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        Z3 = X @ Vt[:3].T
        title = f"{expected_n} faces, one factor each"
        face_labels = np.array([_spec_label(specs[i]) for i in component_gt])

        def to_image(v):
            return np.clip(np.asarray(v) * pstd + pmean, 0.0, 1.0).reshape(image_shape)

        # one strip per matched face: its own frames over the model curve
        correct = []
        for m in matches:
            j = m["curve"]
            if j is None or pipe.curves_[j].get("spline") is None:
                continue
            grid, top, recon = _ground_truth_strip(
                pipe, pipe.curves_[j], m, cid, values, t, component_gt, X)
            correct.append({
                "label": _spec_label(specs[m["spec"]]),
                "angles": grid,
                "labels": [_format_value(g, m["unit"]) for g in grid],
                "true_imgs": top, "recon_imgs": recon,
            })

        # curves no face claimed: model-only rows, numbered
        extra = [(f"extra {k}", _reconstruct_along(pipe, pipe.curves_[j], 5), None)
                 for k, j in enumerate(extra_curve_ids, start=1)]

        figs = {
            "fig_persistence.pdf": F.plot_persistence(
                pipe.structure_["diagram"],
                title="Persistence diagram (H0 = faces, H1 = loops)"),
            "fig_components_scatter.pdf": F.plot_component_scatter(
                Z3[:, :2], face_labels, comp_id, title=title,
                label_name="ground-truth face"),
            "fig_components_scatter_3d.pdf": F.plot_component_scatter_3d(
                Z3, face_labels, comp_id, title=title,
                label_name="ground-truth face"),
            "fig_sweep_correct.pdf": F.plot_recovery_strips(
                correct, image_shape, to_image=to_image,
                title="Factor recovery"),
            "fig_sweep_extra.pdf": F.plot_reconstruction_strips(
                extra, image_shape, to_image=to_image,
                title="Unmatched components",
                empty_note="No unmatched components detected."),
        }
        for name, fig in figs.items():
            fig.savefig(os.path.join(run_dir, name), bbox_inches="tight")

        say(f"done. results in {run_dir}")

    return summary, run_dir


def _format_summary(summary):
    """Compact human-readable one-block summary for logging / a status label."""
    m = summary["metrics"]
    topo = m["topology"]
    lines = [
        f"H0 components: {'OK' if topo['components_match'] else 'MISMATCH'} "
        f"(detected {topo['n_components']}, expected {topo['n_components_expected']})",
        f"H1: {'OK' if topo['paths_match'] and topo['loops_match'] else 'MISMATCH'} "
        f"(detected {topo['n_paths']} arc(s) + {topo['n_loops']} loop(s), "
        f"expected {topo['n_paths_expected']} arc(s))",
        f"ARI (H0 component vs face): {m['ari']:.3f}",
    ]
    for mm in m["matching"]:
        target = "no curve matched" if mm["curve"] is None else \
            f"curve {mm['curve']} (overlap n={mm['n_overlap']})"
        lines.append(f"  face {mm['face']} [{mm['factor']}] -> {target}")
    for c in m["per_component_factor"]:
        lines.append(
            f"  curve {c['curve']} [{c['type']}] face~{c['face']} "
            f"({c['factor']}, purity {c['purity']:.2f}, n={c['n']})  "
            f"error med={c['error_median']:.3f} max={c['error_max']:.3f} {c['unit']}")
    if m["n_extra_curves"]:
        lines.append(f"unmatched curves: {m['n_extra_curves']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    summary, out = run_evaluation(progress=print)
    print("\n" + _format_summary(summary))
    print("saved to", out)

