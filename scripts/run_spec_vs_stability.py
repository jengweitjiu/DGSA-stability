#!/usr/bin/env python3
"""
Specification versus stability rank analysis (Supplementary Figure S1).

Compares differential enrichment rank with stability loss rank
across independence, redundancy, and shared-axis regimes.

Usage:
    python scripts/run_spec_vs_stability.py [--seed 42]
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import spearmanr
from dgsa.simulation import generate_dataset
from dgsa.ablation import specification_rank, stability_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="figures/spec_vs_stability.json")
    args = parser.parse_args()

    configs = {
        "independence": {"n_total": 200, "n_pos": 50, "p": 20, "effect": 1.0},
        "redundancy":   {"n_total": 200, "n_pos": 50, "p": 20, "effect": 1.0, "alpha": 0.9},
        "shared_axis":  {"n_total": 200, "n_pos": 50, "p": 20, "effect": 1.0, "anticorr": 1.0},
    }

    print("="*60)
    print("SPECIFICATION vs STABILITY ANALYSIS")
    print("="*60)

    results = {}
    for regime, kwargs in configs.items():
        X, y, info = generate_dataset(regime=regime, seed=args.seed, **kwargs)

        # Specification rank
        enrichment, spec_ranks = specification_rank(X, y)

        # Stability rank (by AUC loss)
        losses, stab_ranks = stability_rank(
            X, y, metric="auc", random_state=args.seed
        )

        # Spearman correlation between enrichment and losses
        rho, pval = spearmanr(enrichment, losses)

        print(f"\n  {regime.upper()}:")
        print(f"    Signal features (A=col0, B=col1):")
        print(f"      A: enrichment={enrichment[0]:.3f}, loss={losses[0]:.3f}")
        print(f"      B: enrichment={enrichment[1]:.3f}, loss={losses[1]:.3f}")
        print(f"    Spearman rho = {rho:.3f} (p={pval:.4f})")
        print(f"    Top 3 by enrichment: {spec_ranks[:3]}")
        print(f"    Top 3 by stability:  {stab_ranks[:3]}")

        results[regime] = {
            "enrichment": enrichment.tolist(),
            "losses": losses.tolist(),
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
        }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
