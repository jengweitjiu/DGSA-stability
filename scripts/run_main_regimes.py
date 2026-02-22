#!/usr/bin/env python3
"""
Run main regime validation (Figure 2 / Table 1).

Generates representative simulations and 50-replicate distributions
for independence, redundancy, and shared-axis regimes.

Usage:
    python scripts/run_main_regimes.py [--seed 42] [--replicates 50]
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dgsa.simulation import generate_dataset
from dgsa.ablation import compute_synergy
from dgsa.evaluation import run_replicates


def run_representative(regime, seed, **kwargs):
    """Run a single representative simulation and compute synergy."""
    X, y, info = generate_dataset(regime=regime, seed=seed, **kwargs)

    auc_syn = compute_synergy(X, y, 0, 1, metric="auc", random_state=seed)
    depth_syn = compute_synergy(X, y, 0, 1, metric="depth", random_state=seed)

    print(f"\n{'='*60}")
    print(f"Regime: {regime.upper()} (seed={seed})")
    print(f"{'='*60}")
    print(f"  AUC metric:")
    print(f"    Baseline    = {auc_syn['baseline']:.3f}")
    print(f"    ΔA          = {auc_syn['delta_a']:.3f}")
    print(f"    ΔB          = {auc_syn['delta_b']:.3f}")
    print(f"    ΔA+ΔB (exp) = {auc_syn['additive_expected']:.3f}")
    print(f"    ΔAB  (obs)  = {auc_syn['delta_ab']:.3f}")
    print(f"    Synergy     = {auc_syn['synergy']:+.3f}")
    print(f"  Depth metric:")
    print(f"    Synergy     = {depth_syn['synergy']:+.3f}")

    return {"auc": auc_syn, "depth": depth_syn, "info": info}


def main():
    parser = argparse.ArgumentParser(description="Main regime validation")
    parser.add_argument("--seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument("--replicates", type=int, default=50, help="Replicate count")
    parser.add_argument("--output", type=str, default="figures/main_regimes.json")
    args = parser.parse_args()

    # === Table 1: Representative simulations ===
    print("\n" + "="*60)
    print("TABLE 1: Representative Simulations (n=200, n_pos=50, p=20)")
    print("="*60)

    configs = {
        "independence": {"n_total": 200, "n_pos": 50, "p": 20, "effect": 1.0},
        "redundancy":   {"n_total": 200, "n_pos": 50, "p": 20, "effect": 1.0, "alpha": 0.9},
        "shared_axis":  {"n_total": 200, "n_pos": 50, "p": 20, "effect": 1.0, "anticorr": 1.0},
    }

    representatives = {}
    for regime, kwargs in configs.items():
        representatives[regime] = run_representative(regime, seed=args.seed, **kwargs)

    # === Figure 2 D-F: Replicate distributions ===
    print(f"\n{'='*60}")
    print(f"REPLICATE DISTRIBUTIONS (n={args.replicates}, master_seed={args.seed})")
    print(f"{'='*60}")

    replicate_summaries = {}
    for regime, kwargs in configs.items():
        if regime == "redundancy":
            expected = 1
        elif regime == "shared_axis":
            expected = -1
        else:
            expected = None

        results, summary = run_replicates(
            regime=regime,
            n_replicates=args.replicates,
            metric="auc",
            expected_sign=expected,
            master_seed=args.seed,
            **kwargs,
        )

        replicate_summaries[regime] = summary
        print(f"\n  {regime.upper()}:")
        print(f"    Mean synergy = {summary['mean_synergy']:+.4f} "
              f"± {summary['std_synergy']:.4f}")
        print(f"    Frac S > 0   = {summary['frac_positive']:.0%}")
        print(f"    Frac S < 0   = {summary['frac_negative']:.0%}")
        if "detection_rate" in summary:
            print(f"    Detection    = {summary['detection_rate']:.0%}")

    # Also run depth metric for shared-axis comparison (Section 3.1)
    _, depth_summary = run_replicates(
        regime="shared_axis",
        n_replicates=args.replicates,
        metric="depth",
        expected_sign=-1,
        master_seed=args.seed,
        **configs["shared_axis"],
    )
    print(f"\n  SHARED_AXIS (depth metric):")
    print(f"    Frac S < 0   = {depth_summary['frac_negative']:.0%}")
    print(f"    Frac S > 0   = {depth_summary['frac_positive']:.0%}")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "table1": {k: v["auc"] for k, v in representatives.items()},
        "replicate_summaries": replicate_summaries,
        "depth_shared_axis": depth_summary,
        "params": {"seed": args.seed, "replicates": args.replicates},
    }
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
