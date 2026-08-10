"""
evaluation metrics.

- M1 topology_report : detected components + loop/arc types vs expected.

- M2 angle_error     : residual angular error (deg) after quotienting out the
                       unrecoverable freedoms (see eval.align).

- M3 reconstruction_error : (see reconstruct.py)

- M4 discrete_ari    : agreement of detected components with ground-truth classes
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from experiments.eval.align import align_loop, align_arc


def topology_report(structure, expected_n=None, expected_types=None):
    """
    Compare the detected structure to the expected topology.

    Parameters
    ----------
    structure : dict
        A pipeline detection result (has "curves", each with a "type").
    expected_n : int or None
        Expected number of structures/components.
    expected_types : list[str] or None
        Expected per-structure types ("loop" / "path"), order-independent
        (compared as sorted multisets).

    Returns
    -------
    dict with detected counts/types, the expectations, and a "match" bool.
    """
    curves = structure["curves"]
    det_types = [c["type"] for c in curves]
    report = {
        "n_detected": len(curves),
        "types_detected": det_types,
        "n_expected": expected_n,
        "types_expected": expected_types,
    }
    match = True
    if expected_n is not None:
        match = match and (len(curves) == expected_n)
    if expected_types is not None:
        match = match and (sorted(det_types) == sorted(expected_types))
    report["match"] = bool(match)
    return report


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

