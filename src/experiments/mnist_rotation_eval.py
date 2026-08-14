"""
Experiment 1: rotating MNIST digit (a single rotation loop).

Runs one comprehensive, single-configuration evaluation and writes everything to
disk. It crosses two axes:

    mode      : "capacity"  (clean input, self-reconstruction)
                "denoising" (noisy input, clean target)
    detection : "tda"       (persistent homology, the method, Sec. 5.6)
                "traversal" (naive MST-diameter baseline, Sec. 5.5)

so a single call produces the 2x2 = 4 runs.
Each run is saved under

    results/mnist_rotation/<tag>/<mode>_<method>/
"""

import os
import json
import datetime

from data.mnist_rotation import make_rotation_dataset
from core.pipeline import ManifoldPipeline
from experiments.eval import evaluate_run

IMAGE_SHAPE = (30, 30) # rotated frames are 30x30 = 900 px (see mnist_rotation)


def _make_pipeline(
    *, n_components, pca_dim, latent_dim, lambda_aniso,
    n_neighbors, interp_tangent_weight, seed
):
    """A pipeline configured for the rotation loop (detection set per run)."""
    return ManifoldPipeline(
        n_components=n_components, latent_dim=latent_dim, cov_type="mfa",
        shared=False, pca_dim=pca_dim if pca_dim and pca_dim > 0 else None,
        lambda_aniso=lambda_aniso, n_neighbors=n_neighbors,
        interp_tangent_weight=interp_tangent_weight, seed=seed,
    )


def _headline(summary):
    """Pull the scalar metrics we compare across runs out of a run summary."""
    m = summary["metrics"]
    return {
        "topology_match": m["topology"]["match"],
        "n_detected": m["topology"]["n_detected"],
        "types_detected": m["topology"]["types_detected"],
        "angle_mean_deg": m["angle_error"]["mean"],
        "angle_median_deg": m["angle_error"]["median"],
        "angle_max_deg": m["angle_error"]["max"],
        "recon_full": m["reconstruction"]["full"],
        "recon_mfa_floor": m["reconstruction"]["mfa_floor"],
        "recon_pca_floor": m["reconstruction"]["pca_floor"],
    }


def run_evaluation(
    *, digit=6, n_angles=360, noise=0.25, n_components=24,
    pca_dim=20, latent_dim=1, lambda_aniso=30.0, n_neighbors=4,
    interp_tangent_weight=3.0, seed=0,
    methods=("tda", "traversal"), results_root=None,
    progress=None
):
    """
    Run the full single-config rotation evaluation and save all results.

    The defaults are the configuration of the run reported in the thesis
    (`results/mnist_rotation/digit6_seed0/`).

    Returns
    -------
    (comparison_dict, tag_dir) : the headline metrics of every run and the
    directory they were written to.
    """
    def say(msg):
        if progress is not None:
            progress(msg)

    tag = f"digit{digit}_seed{seed}"
    kw = dict(n_components=n_components, pca_dim=pca_dim, latent_dim=latent_dim,
              lambda_aniso=lambda_aniso, n_neighbors=n_neighbors,
              interp_tangent_weight=interp_tangent_weight, seed=seed)

    say("generating data ...")
    X_clean, angles, pmean, pstd = make_rotation_dataset(
        digit=digit, n_angles=n_angles, add_noise=0.0, random_state=seed)
    modes = {"capacity": (X_clean, None)}
    if noise and noise > 0:
        X_noisy, _, _, _ = make_rotation_dataset(
            digit=digit, n_angles=n_angles, add_noise=noise, random_state=seed)
        modes["denoising"] = (X_noisy, X_clean)

    total = len(modes) * len(methods)
    comparison, done = {}, 0

    for mode, (X_input, X_target) in modes.items():
        # fit once per mode, re-detect per method (detection is the cheap step)
        say(f"[{mode}] fitting MFA ...")
        pipe = _make_pipeline(**kw)
        pipe.fit(X_input)

        for method in methods:
            done += 1
            key = f"{mode}_{method}"
            say(f"[{done}/{total}] {mode} / {method}: detecting + evaluating ...")
            try:
                pipe.set_params(detection=method)
                pipe.detect()
                summary, run_dir = evaluate_run(
                    pipe, X_input, angles, kind="loop",
                    image_shape=IMAGE_SHAPE, pixel_mean=pmean, pixel_std=pstd,
                    X_target=X_target, expected_n=1, expected_types=["loop"],
                    experiment="mnist_rotation", run_name=f"{tag}/{key}",
                    results_root=results_root,
                    extra={"digit": digit, "seed": seed, "mode": mode,
                           "method": method,
                           "noise": float(noise) if mode == "denoising" else 0.0},
                )
                comparison[key] = _headline(summary)
            except Exception as exc: # keep the other runs going on a failure
                comparison[key] = {"error": repr(exc)}
                say(f"[{done}/{total}] {mode} / {method}: FAILED ({exc})")

    # write the cross-run comparison
    root = results_root or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
    tag_dir = os.path.abspath(os.path.join(root, "mnist_rotation", tag))
    os.makedirs(tag_dir, exist_ok=True)
    payload = {
        "experiment": "mnist_rotation",
        "tag": tag,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {"digit": digit, "n_angles": n_angles, "noise": noise, **kw},
        "runs": comparison,
    }
    with open(os.path.join(tag_dir, "comparison.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    say(f"done. results in {tag_dir}")
    return comparison, tag_dir


def _format_comparison(comparison):
    """Compact one-line-per-run summary for logging / a status label."""
    lines = []
    for key, m in comparison.items():
        if "error" in m:
            lines.append(f"{key:20s} ERROR: {m['error']}")
        else:
            lines.append(
                f"{key:20s} topo={'ok' if m['topology_match'] else 'X'} "
                f"(n={m['n_detected']})  angle med={m['angle_median_deg']:.1f} "
                f"deg  recon full/mfa/pca="
                f"{m['recon_full']:.3f}/{m['recon_mfa_floor']:.3f}/"
                f"{m['recon_pca_floor']:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import argparse

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # the defaults reproduce the run reported in the thesis
    ap = argparse.ArgumentParser(description="Rotating-MNIST evaluation (Exp. 1)")
    ap.add_argument("--digit", type=int, default=6)
    ap.add_argument("--n-angles", type=int, default=360)
    ap.add_argument("--noise", type=float, default=0.25)
    ap.add_argument("--n-components", type=int, default=24)
    ap.add_argument("--pca-dim", type=int, default=20)
    ap.add_argument("--n-neighbors", type=int, default=4)
    ap.add_argument("--lambda-aniso", type=float, default=30.0)
    ap.add_argument("--interp-tangent-weight", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    comp, out = run_evaluation(
        digit=args.digit, n_angles=args.n_angles, noise=args.noise,
        n_components=args.n_components, pca_dim=args.pca_dim,
        n_neighbors=args.n_neighbors, lambda_aniso=args.lambda_aniso,
        interp_tangent_weight=args.interp_tangent_weight, seed=args.seed,
        progress=print)
    print("\n" + _format_comparison(comp))
    print("saved to", out)

