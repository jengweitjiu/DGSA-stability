#!/usr/bin/env python3
"""
Run power and separability analysis (Figure 3 / Table 2).

Evaluates detection reliability across varying positive-class sample
sizes (n_pos ∈ {9, 15, 25}) and effect strengths (0.10–1.00)
in the shared-axis regime.

Usage:
    python scripts/run_power_analysis.py [--seed 42] [--replicates 50]
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dgsa.config import load_config, numpy_converter
from dgsa.evaluation import power_analysis


def main():
    parser = argparse.ArgumentParser(description="Power analysis (Table 2)")
    parser.add_argument("--seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--output", type=str, default="figures/power_analysis.json")
    args = parser.parse_args()

    cfg = load_config()
    pa = cfg["power_analysis"]
    effect_strengths = pa["effect_strengths"]
    n_pos_values = pa["n_pos_values"]

    print("="*70)
    print(f"POWER ANALYSIS (n_total={pa['n_total']}, p={pa['p']}, "
          f"anticorr={pa['anticorr']}, seed={args.seed}, reps={args.replicates})")
    print("="*70)

    table = power_analysis(
        effect_strengths=effect_strengths,
        n_pos_values=n_pos_values,
        n_total=pa["n_total"],
        p=pa["p"],
        n_replicates=args.replicates,
        anticorr=pa["anticorr"],
        master_seed=args.seed,
    )

    # Print Table 2
    print(f"\n{'Effect':<10}", end="")
    for n_pos in n_pos_values:
        print(f"  {'n_pos='+str(n_pos):>12}  {'CV AUC':>8}  {'Det':>6}", end="")
    print()
    print("-" * 70)

    for eff in effect_strengths:
        row = [t for t in table if t["effect"] == eff]
        print(f"{eff:<10.2f}", end="")
        for n_pos in n_pos_values:
            entry = [t for t in row if t["n_pos"] == n_pos][0]
            det_str = f"{entry['detection_rate']:.2f}"
            auc_str = f"{entry['mean_cv_auc']:.3f}"
            bold = "**" if entry["detection_rate"] >= 0.80 else "  "
            print(f"  {' ':>12}  {auc_str:>8}  {bold}{det_str}{bold}", end="")
        print()

    print(f"\nGate threshold: CV AUC ≥ 0.60. Detection at chance = 0.50.")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "table2": table,
            "params": {
                "effect_strengths": effect_strengths,
                "n_pos_values": n_pos_values,
                "n_total": pa["n_total"], "p": pa["p"],
                "anticorr": pa["anticorr"],
                "seed": args.seed, "replicates": args.replicates,
            },
        }, f, indent=2, default=numpy_converter)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
