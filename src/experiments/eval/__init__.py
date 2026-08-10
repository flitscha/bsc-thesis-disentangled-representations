"""
Reusable evaluation of the manifold pipeline against ground-truth factors.
"""

from experiments.eval.align import align_loop, align_arc
from experiments.eval.metrics import topology_report, angle_error, discrete_ari
from experiments.eval.reconstruct import reconstruction_errors

__all__ = [
    "align_loop", "align_arc",
    "topology_report", "angle_error", "discrete_ari",
    "reconstruction_errors",
]
