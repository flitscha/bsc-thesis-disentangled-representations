"""
Evaluation metrics.

M1 topology_report : H0 components and H1 loops, kept apart, vs expected.
M2 angle_error     : residual error after quotienting out the unrecoverable
                     freedoms (see eval.align).
M3                 : reconstruction decomposition, see reconstruct.py.
M4 discrete_ari    : agreement of detected components with ground-truth classes.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from experiments.eval.align import align_loop, align_arc


def component_labels(pipe, X):
    """
    Per-observation H0 connected-component label.

    The component scatter and the ARI are about H0, which charts merge into one
    structure, not about the H1 curve an observation happens to lie on. Each
    observation therefore inherits the component of its nearest kept chart,
    independent of the loop / arc detection.
    """
    Z = pipe.pre.transform(np.asarray(X)) if pipe.pre is not None else np.asarray(X)
    kept = pipe.kept_
    means = pipe.model.means[kept]              # (M, d) kept chart means
    comp = pipe.structure_["components"][kept]  # (M,) H0 component labels
    d2 = ((Z * Z).sum(1)[:, None] + (means * means).sum(1)[None, :]
          - 2.0 * Z @ means.T)
    return comp[np.argmin(d2, axis=1)]


def topology_report(structure, expected_n=None, expected_types=None,
                    component_labels=None):
    """
    Compare the detected topology to the expected one, H0 and H1 kept apart.

    'expected_n' is the expected number of components; 'expected_types' says per
    component whether it should be a "loop" or an open "path". "match" holds only
    if component, loop and path counts all agree, so closing an arc into a loop
    is a mismatch even when H0 is perfect.

    'component_labels' is the per-observation component assignment; given it, H0
    counts only components that contain data. Without it the chart-level labels
    are used, which may include a component no observation is nearest to.
    """
    curves = structure["curves"]
    det_types = [c["type"] for c in curves]
    n_loops = int(sum(t == "loop" for t in det_types))
    n_paths = int(sum(t == "path" for t in det_types))

    # prefer the data-backed count, then the chart-level labels, and finally the
    # distinct components the detected curves live in
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
        # H0: how many separate structures
        "n_components": n_components,
        "n_components_expected": expected_n,
        "components_match": bool(components_match),
        # H1: loops and open paths living within the components
        "n_loops": n_loops,
        "n_paths": n_paths,
        "n_loops_expected": n_loops_exp,
        "n_paths_expected": n_paths_exp,
        "loops_match": bool(loops_match),
        "paths_match": bool(paths_match),
        # flat view: one entry per detected structure, used by the single-digit run
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

