#!/usr/bin/env python3
"""
Feature-space dimensionality sensitivity analysis (Supplementary Figure S3).

Evaluates synergy sign recovery across p = 5, 10, 20, 40, 80 for
redundancy and shared-axis regimes.

Usage:
    python scripts/run_dimensionality.py [--seed 42] [--replicates 50]
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dgsa.evaluation import dimensionality_analysis


def main():
    parser = argparse.ArgumentParser(description="Dimensionality analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--output", type=str, default="figures/dimensionality.json")
    args = parser.parse_args()

    p_values = [5, 10, 20, 40, 80]

    print("="*60)
    print(f"DIMENSIONALITY SENSITIVITY (n=200, n_pos=50, effect=1.0)")
    print("="*60)

    table = dimensionality_analysis(
        p_values=p_values,
        regimes=("redundancy", "shared_axis"),
        n_replicates=args.replicates,
        n_total=200,
        n_pos=50,
        effect=1.0,
        anticorr=1.0,
        alpha=0.9,
        master_seed=args.seed,
    )

    for regime in ("redundancy", "shared_axis"):
        entries = [t for t in table if t["regime"] == regime]
        print(f"\n  {regime.upper()}:")
        print(f"  {'p':>6}  {'Detection':>12}  {'Mean S':>10}  {'Mean AUC':>10}")
        print(f"  {'-'*42}")
        for e in entries:
            print(f"  {e['p']:>6}  {e['detection_rate']:>12.0%}  "
                  f"{e['mean_synergy']:>+10.4f}  {e['mean_baseline']:>10.3f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(args.output, "w") as f:
        json.dump({"dimensionality": table, "p_values": p_values}, f,
                  indent=2, default=convert)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
