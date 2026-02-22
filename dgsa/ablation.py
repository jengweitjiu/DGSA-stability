"""
Feature ablation and pairwise synergy computation.

Implements the stability decomposition described in Sections 2.1-2.2:
  - Single-feature ablation: ΔA = D₀ − D₋A
  - Pairwise ablation:       ΔAB = D₀ − D₋AB
  - Synergy:                 S(A,B) = ΔAB − (ΔA + ΔB)
"""

import numpy as np
from dgsa.classifier import StabilityClassifier


def compute_stability_loss(
    X, y, feature_indices, metric="auc", C=1.0, n_splits=5, random_state=None,
    insample=False,
):
    """
    Compute stability loss after removing specified features.

    Features are ablated by column removal (not imputation).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Full feature matrix.
    y : ndarray of shape (n_samples,)
        Binary labels.
    feature_indices : list of int
        Column indices to remove.
    metric : str, {"auc", "depth"}
        Stability metric.
    C : float
        Regularization strength.
    n_splits : int
        Cross-validation folds.
    random_state : int or None
        Random seed.
    insample : bool
        If True, use in-sample evaluation (for artifact demonstration).

    Returns
    -------
    baseline : float
        Baseline stability (full feature set).
    ablated : float
        Stability after removing specified features.
    loss : float
        Stability loss (baseline - ablated).
    """
    clf = StabilityClassifier(C=C, n_splits=n_splits, random_state=random_state)

    # Baseline (full features)
    if insample:
        depth_b, auc_b = clf.evaluate_insample(X, y)
    else:
        depth_b, auc_b = clf.evaluate_cv(X, y)
    baseline = auc_b if metric == "auc" else depth_b

    # Ablated (remove columns)
    keep = [i for i in range(X.shape[1]) if i not in feature_indices]
    X_abl = X[:, keep]

    clf_abl = StabilityClassifier(C=C, n_splits=n_splits, random_state=random_state)
    if insample:
        depth_a, auc_a = clf_abl.evaluate_insample(X_abl, y)
    else:
        depth_a, auc_a = clf_abl.evaluate_cv(X_abl, y)
    ablated = auc_a if metric == "auc" else depth_a

    return baseline, ablated, baseline - ablated


def compute_synergy(
    X, y, idx_a, idx_b, metric="auc", C=1.0, n_splits=5, random_state=None,
    insample=False,
):
    """
    Compute pairwise synergy S(A,B) = ΔAB − (ΔA + ΔB).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    idx_a : int
        Column index of feature A.
    idx_b : int
        Column index of feature B.
    metric : str, {"auc", "depth"}
    C : float
    n_splits : int
    random_state : int or None
    insample : bool

    Returns
    -------
    result : dict
        Keys: baseline, delta_a, delta_b, delta_ab, synergy, metric.
    """
    kwargs = dict(
        metric=metric, C=C, n_splits=n_splits, random_state=random_state,
        insample=insample,
    )

    baseline_a, _, delta_a = compute_stability_loss(X, y, [idx_a], **kwargs)
    baseline_b, _, delta_b = compute_stability_loss(X, y, [idx_b], **kwargs)
    baseline_ab, _, delta_ab = compute_stability_loss(X, y, [idx_a, idx_b], **kwargs)

    # Use mean baseline for consistency
    baseline = np.mean([baseline_a, baseline_b, baseline_ab])

    synergy = delta_ab - (delta_a + delta_b)

    return {
        "baseline": baseline,
        "delta_a": delta_a,
        "delta_b": delta_b,
        "delta_ab": delta_ab,
        "additive_expected": delta_a + delta_b,
        "synergy": synergy,
        "metric": metric,
    }


def full_decomposition(
    X, y, idx_a=0, idx_b=1, C=1.0, n_splits=5, random_state=None,
    gate_threshold=0.60,
):
    """
    Full stability decomposition with separability gating.

    Computes synergy under both AUC and depth metrics, applies the
    separability gate, and returns a comprehensive result dictionary.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    idx_a, idx_b : int
        Feature indices for pairwise analysis.
    C : float
    n_splits : int
    random_state : int or None
    gate_threshold : float, default=0.60
        Minimum CV AUC for decomposition to proceed.

    Returns
    -------
    result : dict
        Comprehensive decomposition result including both metrics,
        gating decision, and all component values.
    """
    # Evaluate baseline separability
    clf = StabilityClassifier(C=C, n_splits=n_splits, random_state=random_state)
    baseline_depth, baseline_auc = clf.evaluate_cv(X, y)

    gate_pass = baseline_auc >= gate_threshold

    result = {
        "baseline_auc": baseline_auc,
        "baseline_depth": baseline_depth,
        "gate_threshold": gate_threshold,
        "gate_pass": gate_pass,
        "idx_a": idx_a,
        "idx_b": idx_b,
    }

    if not gate_pass:
        result["status"] = "HALTED: below separability gate"
        result["auc_synergy"] = None
        result["depth_synergy"] = None
        return result

    # Compute synergy under both metrics
    auc_result = compute_synergy(
        X, y, idx_a, idx_b, metric="auc", C=C, n_splits=n_splits,
        random_state=random_state,
    )
    depth_result = compute_synergy(
        X, y, idx_a, idx_b, metric="depth", C=C, n_splits=n_splits,
        random_state=random_state,
    )

    result["status"] = "COMPLETED"
    result["auc_synergy"] = auc_result
    result["depth_synergy"] = depth_result
    result["sign_concordant"] = (
        np.sign(auc_result["synergy"]) == np.sign(depth_result["synergy"])
    )

    return result


def specification_rank(X, y):
    """
    Rank features by class-conditional differential enrichment.

    Specification rank is the absolute mean difference between
    positive and negative class values for each feature, as
    described in Section 2.3.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)

    Returns
    -------
    enrichment : ndarray of shape (n_features,)
        Absolute mean difference per feature.
    ranks : ndarray of shape (n_features,)
        Rank order (0 = highest enrichment).
    """
    pos_mask = y == 1
    neg_mask = y == 0
    enrichment = np.abs(X[pos_mask].mean(axis=0) - X[neg_mask].mean(axis=0))
    ranks = np.argsort(-enrichment)  # descending
    return enrichment, ranks


def stability_rank(X, y, metric="auc", C=1.0, n_splits=5, random_state=None):
    """
    Rank features by marginal stability loss under single-feature ablation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    metric, C, n_splits, random_state : see compute_stability_loss

    Returns
    -------
    losses : ndarray of shape (n_features,)
        Stability loss per feature.
    ranks : ndarray of shape (n_features,)
        Rank order (0 = highest loss).
    """
    losses = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        _, _, loss = compute_stability_loss(
            X, y, [i], metric=metric, C=C, n_splits=n_splits,
            random_state=random_state,
        )
        losses[i] = loss

    ranks = np.argsort(-losses)
    return losses, ranks
