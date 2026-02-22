# DGSA-stability

**Decomposable Geometric Stability Analysis Reveals Conditional Non-Additivity Under Feature Ablation**

Jeng-Wei Tjiu  
Department of Dermatology, National Taiwan University Hospital and National Taiwan University College of Medicine, Taipei, Taiwan

---

## Overview

DGSA is a computational framework that decomposes cell state stability into additive and non-additive components through systematic feature ablation in regulon activity space. The framework:

1. Defines **stability as geometric depth** — the margin-based distance of cell states from a classification boundary
2. Decomposes stability loss through **single-feature and pairwise ablation**
3. Measures **pairwise synergy** as non-additivity: S(A,B) = ΔAB − (ΔA + ΔB)
4. Applies a **separability gate** (CV AUC ≥ 0.60) to prevent overinterpretation

### Synergy interpretation

| Sign | Meaning | Biological analogue |
|------|---------|---------------------|
| S > 0 | Masked dependence — individual removals underestimate joint loss | Functional redundancy |
| S ≈ 0 | Additive — features contribute independently | Independent regulation |
| S < 0 | Overlapping damage — individual removals overestimate joint loss | Shared structural axis |

## Repository structure

```
DGSA-stability/
├── dgsa/                      # Core Python package
│   ├── __init__.py
│   ├── classifier.py          # L2-regularized logistic regression + margins
│   ├── ablation.py            # Feature ablation + synergy computation
│   ├── simulation.py          # Three simulation regimes
│   └── evaluation.py          # CV evaluation, separability gate, power analysis
├── scripts/                   # Reproduction scripts
│   ├── run_all.py             # Run full pipeline
│   ├── run_main_regimes.py    # Table 1, Figure 2
│   ├── run_power_analysis.py  # Table 2, Figure 3
│   ├── run_ceiling_artifact.py # Table 3, Figure 4
│   ├── run_spec_vs_stability.py # Supp Figure S1
│   └── run_dimensionality.py  # Supp Figure S3
├── config/
│   └── params.yaml            # All simulation parameters
├── figures/                   # Generated outputs (after running scripts)
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/jengweitjiu/DGSA-stability.git
cd DGSA-stability
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, NumPy, SciPy, scikit-learn, matplotlib

## Reproducing manuscript results

### Full reproduction

```bash
python scripts/run_all.py --seed 42 --replicates 50
```

This runs all analyses sequentially (~15–30 min) and saves results to `figures/`.

### Individual analyses

Each script can be run independently:

```bash
# Table 1 + Figure 2: Main regime validation
python scripts/run_main_regimes.py --seed 42 --replicates 50

# Table 2 + Figure 3: Power and separability analysis
python scripts/run_power_analysis.py --seed 42 --replicates 50

# Table 3 + Figure 4: Ceiling artifact + TRM gating
python scripts/run_ceiling_artifact.py

# Supplementary Figure S1: Specification vs stability
python scripts/run_spec_vs_stability.py --seed 42

# Supplementary Figure S3: Dimensionality sensitivity
python scripts/run_dimensionality.py --seed 42 --replicates 50
```

## Quick start: Using DGSA on your own data

```python
import numpy as np
from dgsa import full_decomposition

# X: regulon activity matrix (n_cells × n_regulons)
# y: binary cell state labels (0 or 1)
# idx_a, idx_b: column indices of the two regulons to test

result = full_decomposition(X, y, idx_a=0, idx_b=1)

if result["gate_pass"]:
    syn = result["auc_synergy"]
    print(f"Synergy S = {syn['synergy']:+.3f}")
    print(f"  ΔA = {syn['delta_a']:.3f}")
    print(f"  ΔB = {syn['delta_b']:.3f}")
    print(f"  ΔAB = {syn['delta_ab']:.3f}")
else:
    print(f"Decomposition halted: CV AUC = {result['baseline_auc']:.3f} "
          f"< gate threshold {result['gate_threshold']}")
```

## Simulation parameters

All simulation parameters are documented in `config/params.yaml` and correspond exactly to the values reported in the Methods section:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `C` | 1.0 | L2 regularization strength |
| `n_splits` | 5 | Stratified k-fold CV |
| `master_seed` | 42 | Master RNG seed |
| `gate_threshold` | 0.60 | Separability gate (CV AUC) |
| `n_total` (main) | 200 | Samples per simulation |
| `n_pos` (main) | 50 | Positive-class samples |
| `p` | 20 | Total features (2 signal + 18 noise) |
| `n_replicates` | 50 | Replicate datasets per condition |

### Key seeds for specific figures

| Figure | Seed | Parameters |
|--------|------|------------|
| Table 3 ceiling | 23 | eff=1.0, anticorr=2.0, n=89, n_pos=12 |
| Table 3 TRM-like | 77 | eff=0.08, anticorr=0.3, n=89, n_pos=12 |

## Manuscript correspondence

| Manuscript section | Script | Output |
|-------------------|--------|--------|
| §3.1 Regime validation | `run_main_regimes.py` | Table 1, Figure 2 |
| §3.2 Spec vs maintenance | `run_spec_vs_stability.py` | Supp Figure S1 |
| §3.3 Power analysis | `run_power_analysis.py` | Table 2, Figure 3 |
| §3.4 Ceiling artifact | `run_ceiling_artifact.py` | Table 3, Figure 4 |
| Supp S3 dimensionality | `run_dimensionality.py` | Supp Figure S3 |

## Software versions

Analyses reported in the manuscript were performed with:
- Python 3.10
- scikit-learn 1.8
- NumPy 2.4
- matplotlib 3.10

## Citation

If you use this framework, please cite:

> Tjiu J-W. Decomposable Geometric Stability Analysis Reveals Conditional Non-Additivity Under Feature Ablation. *Bioinformatics* (2026). [submitted]

## License

MIT License. See [LICENSE](LICENSE).
