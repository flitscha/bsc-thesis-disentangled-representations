"""
Evaluation of the manifold pipeline against ground-truth factors.

The protocol shared by all four experiments: the post-hoc alignment (align.py), the metrics M1 to
M4 (metrics.py, reconstruct.py), the plots (figures.py) and the orchestrator that runs all of it
for one fitted pipeline and writes a result directory (runner.py).
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
