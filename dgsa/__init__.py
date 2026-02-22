"""
DGSA: Decomposable Geometric Stability Analysis

A framework for decomposing cell state stability into additive and
non-additive components via feature ablation in regulon activity space.

Reference
---------
Tjiu J-W. Decomposable Geometric Stability Analysis Reveals Conditional
Non-Additivity Under Feature Ablation. Bioinformatics (submitted 2026).
"""

__version__ = "1.0.0"

from dgsa.classifier import StabilityClassifier
from dgsa.ablation import compute_stability_loss, compute_synergy, full_decomposition
from dgsa.simulation import generate_dataset
from dgsa.evaluation import separability_gate, evaluate_metrics

__all__ = [
    "StabilityClassifier",
    "compute_stability_loss",
    "compute_synergy",
    "full_decomposition",
    "generate_dataset",
    "separability_gate",
    "evaluate_metrics",
]
