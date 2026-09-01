"""
The evaluation metrics M1, M2 and M4. M3 lives in reconstruct.py.

M1 topology_report : the detected H0 components and H1 loops against the expected ones.
M2 angle_error     : the residual error, after the freedoms the model cannot recover have been
                     fitted out (see eval/align.py).
M4 discrete_ari    : how well the detected components agree with the ground-truth classes.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from experiments.eval.align import align_loop, align_arc


def component_labels(pipe, X):
    """
    The H0 connected-component label of every observation.

    The component scatter and the ARI ask which charts merge into one structure, not which curve
    an observation happens to lie on. So each observation inherits the component of its nearest
    surviving chart, independently of the loop and arc detection.

    Needs a detection that produced an H0 decomposition, which means detection="tda". The
    traversal baseline assumes a single structure and stores no components at all.
    """
    Z = pipe.pre.transform(np.asarray(X)) if pipe.pre is not None else np.asarray(X)
    kept = pipe.kept_
    means = pipe.model.means[kept] # (M, d) kept chart means
    comp = pipe.structure_["components"][kept] # (M,) H0 component labels
    d2 = ((Z * Z).sum(1)[:, None] + (means * means).sum(1)[None, :]
          - 2.0 * Z @ means.T)
    return comp[np.argmin(d2, axis=1)]


def topology_report(structure, expected_n=None, expected_types=None,
                    component_labels=None):
    """
    Compare the detected topology to the expected one, keeping H0 and H1 apart.

    'expected_n' is the expected number of components, 'expected_types' says per component whether
    it should be a "loop" or an open "path". "match" only holds if the component, loop and path
    counts all agree, so closing an arc into a loop counts as a mismatch even when H0 is perfect.

    Passing 'component_labels' makes H0 count only components that actually contain data. Without
    it the chart-level labels are used, which can include a component no observation is nearest to.
    """
    curves = structure["curves"]
    det_types = [c["type"] for c in curves]
    n_loops = int(sum(t == "loop" for t in det_types))
    n_paths = int(sum(t == "path" for t in det_types))

    # prefer the data-backed count, then the chart-level labels, and only then the components the
    # detected curves happen to live in
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
        # H0: how many separate structures there are
        "n_components": n_components,
        "n_components_expected": expected_n,
        "components_match": bool(components_match),
        # H1: the loops and open paths inside those components
        "n_loops": n_loops,
        "n_paths": n_paths,
        "n_loops_expected": n_loops_exp,
        "n_paths_expected": n_paths_exp,
        "loops_match": bool(loops_match),
        "paths_match": bool(paths_match),
        # a flat view, one entry per detected structure, used by the single-digit run
        "n_detected": len(curves),
        "types_detected": det_types,
        "n_expected": expected_n,
        "types_expected": expected_types,
        "match": match,
    }


def angle_error(t, factor, kind="loop"):
    """
    The residual factor-recovery error, measured after the post-hoc alignment.

    kind="loop" treats the factor as a periodic angle in degrees and fits out direction and
    offset. kind="arc" treats it as an interval and fits out the affine reparametrization.

    Returns a dict with the (N,) "error", the scalars "mean", "median" and "max", and the
    alignment itself, which also carries the two maps between t and the factor.
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
    Adjusted Rand Index between the detected component labels and the ground-truth classes.

    Points labelled -1, which belong to a pruned chart, are left out.
    """
    components_pred = np.asarray(components_pred)
    labels_true = np.asarray(labels_true)
    keep = components_pred >= 0
    return float(adjusted_rand_score(labels_true[keep], components_pred[keep]))

