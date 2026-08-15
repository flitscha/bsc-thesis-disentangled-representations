"""
Reusable evaluation of the manifold pipeline against ground-truth factors.
"""

from experiments.eval.align import align_loop, align_arc
from experiments.eval.metrics import (
    topology_report, angle_error, discrete_ari, component_labels,
)
from experiments.eval.reconstruct import reconstruction_errors
from experiments.eval.runner import evaluate_run, persistence_tag

__all__ = [
    "evaluate_run", "persistence_tag",
    "align_loop", "align_arc",
    "topology_report", "angle_error", "discrete_ari", "component_labels",
    "reconstruction_errors",
]
