"""
evaluation metrics.

- M1 topology_report : H0 components + H1 loops (kept apart) vs expected.

- M2 angle_error     : residual angular error (deg) after quotienting out the
                       unrecoverable freedoms (see eval.align).

- M3 reconstruction_error : (see reconstruct.py)

- M4 discrete_ari    : agreement of detected components with ground-truth classes
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from experiments.eval.align import align_loop, align_arc


def topology_report(structure, expected_n=None, expected_types=None,
                    component_labels=None):
    """
    Compare the detected topology to the expected one, keeping the two
    homology dimensions apart:

      H0 -- connected components: how many separate structures there are.
      H1 -- loops: cycles that live *within* a component. A single component
            can carry several loops, and a component with no loop is traced as
            an open path instead.

    `expected_n` is the number of expected components (H0). `expected_types`
    lists, per expected component, whether it should be a "loop" or an open
    "path" (H1); their counts give the expected number of loops / paths. The
    overall "match" holds iff the component count, the loop count and the path
    count all agree -- so closing an open arc into a loop shows up as an H1
    mismatch even when H0 is perfect.

    `component_labels`, if given, is the per-observation component assignment.
    The H0 count is then the number of components that actually contain data,
    which is what the component scatter shows. Without it the count falls back
    to the chart-level labels in `structure["components"]`, which can include a
    "phantom" component of charts that no observation is nearest to.

    Returns a dict with the H0 block, the H1 block, and (for backwards
    compatibility) the per-structure `n_detected` / `types_detected` fields.
    """
    curves = structure["curves"]
    det_types = [c["type"] for c in curves]
    n_loops = int(sum(t == "loop" for t in det_types))
    n_paths = int(sum(t == "path" for t in det_types))

    # H0: connected components. Prefer the data-backed count (components that
    # actually contain observations); otherwise the chart-level labels, and as a
    # last resort the distinct components the detected curves live in.
    if component_labels is not None:
        cl = np.asarray(component_labels)
        n_components = int(np.unique(cl[cl >= 0]).size)
    elif structure.get("components") is not None:
        comp = np.asarray(structure["components"])
        n_components = int(np.unique(comp[comp >= 0]).size)
    else:
        ids = {c.get("component") for c in curves if c.get("component") is not None}
        n_components = len(ids) or len(curves)

    exp_types = list(expected_types) if expected_types is not None else None
    n_loops_exp = sum(t == "loop" for t in exp_types) if exp_types is not None else None
    n_paths_exp = sum(t == "path" for t in exp_types) if exp_types is not None else None

    components_match = (expected_n is None) or (n_components == expected_n)
    loops_match = (n_loops_exp is None) or (n_loops == n_loops_exp)
    paths_match = (n_paths_exp is None) or (n_paths == n_paths_exp)
    match = bool(components_match and loops_match and paths_match)

    return {
        # H0 -- connected components (how many separate structures)
        "n_components": n_components,
        "n_components_expected": expected_n,
        "components_match": bool(components_match),
        # H1 -- loops (+ open paths) living *within* the components
        "n_loops": n_loops,
        "n_paths": n_paths,
        "n_loops_expected": n_loops_exp,
        "n_paths_expected": n_paths_exp,
        "loops_match": bool(loops_match),
        "paths_match": bool(paths_match),
        # legacy: one entry per detected structure (loop or path)
        "n_detected": len(curves),
        "types_detected": det_types,
        "n_expected": expected_n,
        "types_expected": expected_types,
        "match": match,
    }


def angle_error(t, factor, kind="loop"):
    """
    Residual factor-recovery error after post-hoc alignment.

    kind="loop": periodic angle in degrees, quotient out direction + offset.
    kind="arc" : interval factor, quotient out the affine reparametrization.

    Returns
    -------
    dict with "error" (N,), scalar "mean"/"median"/"max", and the alignment
    ("s"/"delta_deg" for loops, "a"/"b" for arcs) plus the t<->factor maps.
    """
    if kind == "loop":
        al = align_loop(t, factor)
        err = al["error_deg"]
    elif kind == "arc":
        al = align_arc(t, factor)
        err = al["error"]
    else:
        raise ValueError(f"unknown kind '{kind}' (use 'loop' or 'arc')")

    return {
        "error": err,
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
        "alignment": al,
    }


def discrete_ari(components_pred, labels_true):
    """
    Adjusted Rand Index between detected component labels and ground-truth
    classes. Points labelled -1 (pruned charts) are dropped.
    """
    components_pred = np.asarray(components_pred)
    labels_true = np.asarray(labels_true)
    keep = components_pred >= 0
    return float(adjusted_rand_score(labels_true[keep], components_pred[keep]))

