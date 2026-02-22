"""
Evaluation utilities: separability gating, detection probability,
and batch replicate analysis.

Implements the separability gate (Section 3.3) and power analysis
(Methods: Power and Separability Analysis).
"""

import numpy as np
from dgsa.classifier import StabilityClassifier
from dgsa.ablation import compute_synergy
from dgsa.simulation import generate_dataset


def separability_gate(X, y, threshold=0.60, C=1.0, n_splits=5, random_state=None):
    """
    Apply the separability gate prior to stability decomposition.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    threshold : float, default=0.60
        Minimum CV AUC for decomposition to proceed.
    C, n_splits, random_state : classifier parameters

    Returns
    -------
    passes : bool
        True if baseline CV AUC >= threshold.
    baseline_auc : float
        Cross-validated ROC AUC.
    baseline_depth : float
        Cross-validated stability depth.
    """
    clf = StabilityClassifier(C=C, n_splits=n_splits, random_state=random_state)
    depth, auc = clf.evaluate_cv(X, y)
    return auc >= threshold, auc, depth


def evaluate_metrics(X, y, C=1.0, n_splits=5, random_state=None):
    """
    Evaluate both CV and in-sample metrics for comparison.

    Returns
    -------
    result : dict
        Keys: cv_auc, cv_depth, is_auc, is_depth.
    """
    clf = StabilityClassifier(C=C, n_splits=n_splits, random_state=random_state)
    cv_depth, cv_auc = clf.evaluate_cv(X, y)
    is_depth, is_auc = clf.evaluate_insample(X, y)
    return {
        "cv_auc": cv_auc,
        "cv_depth": cv_depth,
        "is_auc": is_auc,
        "is_depth": is_depth,
    }


def run_replicates(
    regime,
    n_replicates=50,
    metric="auc",
    expected_sign=None,
    master_seed=42,
    **sim_kwargs,
):
    """
    Run replicate simulations and compute synergy for each.

    Parameters
    ----------
    regime : str
        Simulation regime.
    n_replicates : int, default=50
        Number of replicate datasets.
    metric : str, {"auc", "depth"}
        Stability metric for synergy.
    expected_sign : {1, -1, None}
        Expected synergy sign for detection rate calculation.
        +1 for redundancy, -1 for shared_axis, None for independence.
    master_seed : int, default=42
        Master RNG seed for generating per-replicate seeds.
    **sim_kwargs
        Passed to generate_dataset (n_total, n_pos, p, effect, etc.)

    Returns
    -------
    results : list of dict
        Synergy results for each replicate.
    summary : dict
        Aggregate statistics: mean_synergy, std_synergy, detection_rate,
        mean_baseline, sign_counts.
    """
    master_rng = np.random.RandomState(master_seed)
    seeds = master_rng.randint(0, 2**31 - 1, size=n_replicates)

    results = []
    for i, seed in enumerate(seeds):
        seed = int(seed)
        X, y, info = generate_dataset(regime=regime, seed=seed, **sim_kwargs)

        syn = compute_synergy(
            X, y, idx_a=0, idx_b=1, metric=metric, C=1.0, n_splits=5,
            random_state=seed,
        )
        syn["seed"] = seed
        syn["replicate"] = i
        results.append(syn)

    synergies = np.array([r["synergy"] for r in results])
    baselines = np.array([r["baseline"] for r in results])

    summary = {
        "regime": regime,
        "metric": metric,
        "n_replicates": n_replicates,
        "mean_synergy": float(np.mean(synergies)),
        "std_synergy": float(np.std(synergies)),
        "median_synergy": float(np.median(synergies)),
        "mean_baseline": float(np.mean(baselines)),
        "frac_positive": float(np.mean(synergies > 0)),
        "frac_negative": float(np.mean(synergies < 0)),
    }

    if expected_sign is not None:
        if expected_sign > 0:
            detection = np.mean(synergies > 0)
        else:
            detection = np.mean(synergies < 0)
        summary["detection_rate"] = float(detection)
        summary["expected_sign"] = expected_sign

    return results, summary


def power_analysis(
    effect_strengths,
    n_pos_values,
    n_total=89,
    p=20,
    n_replicates=50,
    anticorr=1.0,
    master_seed=42,
):
    """
    Evaluate detection power across sample sizes and effect strengths.

    Implements the power analysis described in Methods: Power and
    Separability Analysis.

    Parameters
    ----------
    effect_strengths : list of float
        Effect sizes to test.
    n_pos_values : list of int
        Positive-class sample sizes to test.
    n_total : int
        Total sample size (fixed).
    p : int
        Feature dimensionality.
    n_replicates : int
        Replicates per condition.
    anticorr : float
        Shared-axis anti-correlation strength.
    master_seed : int
        Master random seed.

    Returns
    -------
    table : list of dict
        One entry per (effect, n_pos) condition with detection rate
        and mean CV AUC.
    """
    table = []

    for eff in effect_strengths:
        for n_pos in n_pos_values:
            _, summary = run_replicates(
                regime="shared_axis",
                n_replicates=n_replicates,
                metric="auc",
                expected_sign=-1,
                master_seed=master_seed,
                n_total=n_total,
                n_pos=n_pos,
                p=p,
                effect=eff,
                anticorr=anticorr,
            )

            table.append({
                "effect": eff,
                "n_pos": n_pos,
                "mean_cv_auc": summary["mean_baseline"],
                "detection_rate": summary["detection_rate"],
                "mean_synergy": summary["mean_synergy"],
            })

    return table


def dimensionality_analysis(
    p_values,
    regimes=("redundancy", "shared_axis"),
    n_replicates=50,
    n_total=200,
    n_pos=50,
    effect=1.0,
    anticorr=1.0,
    alpha=0.9,
    master_seed=42,
):
    """
    Assess synergy sign recovery across feature-space dimensionalities.

    Implements the dimensionality sensitivity analysis described in
    Methods: Dimensionality Sensitivity.

    Parameters
    ----------
    p_values : list of int
        Total feature counts to test (p = 5, 10, 20, 40, 80).
    regimes : tuple of str
        Regimes to evaluate.
    n_replicates, n_total, n_pos, effect, anticorr, alpha : simulation params
    master_seed : int

    Returns
    -------
    table : list of dict
        One entry per (regime, p) condition.
    """
    table = []

    for regime in regimes:
        expected = 1 if regime == "redundancy" else -1
        for p_val in p_values:
            sim_kwargs = dict(
                n_total=n_total, n_pos=n_pos, p=p_val, effect=effect,
            )
            if regime == "redundancy":
                sim_kwargs["alpha"] = alpha
            else:
                sim_kwargs["anticorr"] = anticorr

            _, summary = run_replicates(
                regime=regime,
                n_replicates=n_replicates,
                metric="auc",
                expected_sign=expected,
                master_seed=master_seed,
                **sim_kwargs,
            )

            table.append({
                "regime": regime,
                "p": p_val,
                "detection_rate": summary["detection_rate"],
                "mean_synergy": summary["mean_synergy"],
                "mean_baseline": summary["mean_baseline"],
            })

    return table
