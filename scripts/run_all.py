#!/usr/bin/env python3
"""
Run all analyses to reproduce manuscript results.

This is a convenience wrapper that runs all five analysis scripts
in sequence. Each script can also be run independently.

Usage:
    python scripts/run_all.py [--seed 42] [--replicates 50]

Runtime: approximately 15-30 minutes depending on hardware.
"""

import argparse
import subprocess
import sys
import os
import time


SCRIPTS = [
    ("Main regimes (Table 1, Figure 2)", "scripts/run_main_regimes.py"),
    ("Power analysis (Table 2, Figure 3)", "scripts/run_power_analysis.py"),
    ("Ceiling artifact (Table 3, Figure 4)", "scripts/run_ceiling_artifact.py"),
    ("Spec vs stability (Supp Figure S1)", "scripts/run_spec_vs_stability.py"),
    ("Dimensionality (Supp Figure S3)", "scripts/run_dimensionality.py"),
]


def main():
    parser = argparse.ArgumentParser(description="Run all DGSA analyses")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=50)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    os.makedirs("figures", exist_ok=True)

    total_start = time.time()
    print("=" * 60)
    print("DGSA: Full Reproduction Pipeline")
    print(f"Master seed: {args.seed}, Replicates: {args.replicates}")
    print("=" * 60)

    for i, (desc, script) in enumerate(SCRIPTS, 1):
        print(f"\n[{i}/{len(SCRIPTS)}] {desc}")
        print("-" * 60)
        start = time.time()

        cmd = [sys.executable, script]
        # run_ceiling_artifact.py accepts no CLI args
        if "ceiling" not in script:
            cmd.extend(["--seed", str(args.seed)])
        # Only some scripts accept --replicates
        if "ceiling" not in script and "spec_vs" not in script:
            cmd.extend(["--replicates", str(args.replicates)])

        result = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"  ✗ FAILED (exit code {result.returncode})")
        else:
            print(f"  ✓ Completed in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All analyses complete in {total_elapsed:.1f}s")
    print(f"Results in: figures/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
