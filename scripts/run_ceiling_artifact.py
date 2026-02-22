#!/usr/bin/env python3
"""
Ceiling artifact and separability gating demonstration (Figure 4 / Table 3).

Demonstrates:
  1. Ceiling compression: in-sample AUC ≈ 0.98 vs CV AUC ≈ 0.67
  2. Gating decision: TRM-like data correctly halted below gate

Usage:
    python scripts/run_ceiling_artifact.py
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dgsa.simulation import generate_ceiling_case, generate_trm_like
from dgsa.ablation import compute_synergy, full_decomposition
from dgsa.evaluation import evaluate_metrics


def print_case(name, X, y, info, gate_threshold=0.60):
    """Run both evaluation modes and print Table 3 row."""
    print(f"\n{'='*60}")
    print(f"CASE: {name}")
    print(f"  Params: eff={info['effect']}, anticorr={info['anticorr']}, "
          f"seed={info['seed']}")
    print(f"  Data:   n={info['n_total']}, n_pos={info['n_pos']}, p={info['p']}")
    print(f"{'='*60}")

    # Evaluate both modes
    metrics = evaluate_metrics(X, y, random_state=info["seed"])
    print(f"\n  In-sample AUC:       {metrics['is_auc']:.3f}")
    print(f"  Cross-validated AUC: {metrics['cv_auc']:.3f}")
    print(f"  In-sample depth:     {metrics['is_depth']:.3f}")
    print(f"  Cross-validated depth:{metrics['cv_depth']:.3f}")

    # In-sample synergy (for artifact demonstration)
    is_syn = compute_synergy(
        X, y, 0, 1, metric="auc", random_state=info["seed"], insample=True
    )
    print(f"\n  In-sample ablation:")
    print(f"    ΔA  = {is_syn['delta_a']:.3f}")
    print(f"    ΔB  = {is_syn['delta_b']:.3f}")
    print(f"    ΔAB = {is_syn['delta_ab']:.3f}")
    print(f"    S   = {is_syn['synergy']:+.3f}")

    # Cross-validated decomposition with gate
    result = full_decomposition(
        X, y, idx_a=0, idx_b=1, random_state=info["seed"],
        gate_threshold=gate_threshold,
    )
    print(f"\n  Cross-validated decomposition:")
    print(f"    Baseline CV AUC = {result['baseline_auc']:.3f}")
    print(f"    Gate threshold  = {result['gate_threshold']:.2f}")
    print(f"    Gate decision   = {result['status']}")

    if result["gate_pass"]:
        cv_syn = result["auc_synergy"]
        print(f"    ΔA  = {cv_syn['delta_a']:.3f}")
        print(f"    ΔB  = {cv_syn['delta_b']:.3f}")
        print(f"    ΔAB = {cv_syn['delta_ab']:.3f}")
        print(f"    S   = {cv_syn['synergy']:+.3f}")

    return {
        "name": name,
        "metrics": metrics,
        "is_synergy": is_syn,
        "cv_result": result,
        "info": info,
    }


def main():
    print("="*60)
    print("TABLE 3: In-Sample vs Cross-Validated Evaluation")
    print("="*60)

    # Case 1: Ceiling artifact (well-separated)
    X_ceil, y_ceil, info_ceil = generate_ceiling_case()
    ceil_result = print_case("Ceiling Artifact", X_ceil, y_ceil, info_ceil)

    # Case 2: TRM-like (below gate)
    X_trm, y_trm, info_trm = generate_trm_like()
    trm_result = print_case("TRM-like Gating", X_trm, y_trm, info_trm)

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY: Gating Decisions")
    print(f"{'='*60}")
    print(f"  Ceiling case:")
    print(f"    In-sample would PASS  (AUC={ceil_result['metrics']['is_auc']:.3f})")
    print(f"    CV correctly PASSES   (AUC={ceil_result['metrics']['cv_auc']:.3f})")
    print(f"  TRM-like case:")
    print(f"    In-sample would PASS  (AUC={trm_result['metrics']['is_auc']:.3f})  ← MISLEADING")
    print(f"    CV correctly HALTS    (AUC={trm_result['metrics']['cv_auc']:.3f})  ← CORRECT")

    # Save
    os.makedirs("figures", exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.bool_)):
            return float(obj)
        return obj

    with open("figures/ceiling_artifact.json", "w") as f:
        json.dump({
            "ceiling": {
                "is_auc": ceil_result["metrics"]["is_auc"],
                "cv_auc": ceil_result["metrics"]["cv_auc"],
                "is_synergy": ceil_result["is_synergy"]["synergy"],
                "cv_synergy": (ceil_result["cv_result"]["auc_synergy"]["synergy"]
                               if ceil_result["cv_result"]["gate_pass"] else None),
            },
            "trm_like": {
                "is_auc": trm_result["metrics"]["is_auc"],
                "cv_auc": trm_result["metrics"]["cv_auc"],
                "gate_pass": trm_result["cv_result"]["gate_pass"],
            },
        }, f, indent=2, default=convert)
    print(f"\nResults saved to figures/ceiling_artifact.json")


if __name__ == "__main__":
    main()
