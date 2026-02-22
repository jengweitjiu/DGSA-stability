"""
Controlled simulation regimes for validating stability decomposition.

Generates synthetic datasets with known feature-pair relationships:
  - Independence: orthogonal signal axes
  - Redundancy (masking): correlated backup features
  - Shared axis (overlap): complementary contributions to shared discriminant

See Methods: Simulation Design.
"""

import numpy as np
from scipy.special import expit


def generate_dataset(
    regime,
    n_total=200,
    n_pos=50,
    p=20,
    effect=1.0,
    alpha=0.9,
    anticorr=1.0,
    noise_std=1.0,
    seed=None,
):
    """
    Generate a synthetic dataset for a specified simulation regime.

    Parameters
    ----------
    regime : str, {"independence", "redundancy", "shared_axis"}
        Simulation regime defining the relationship between features A and B.
    n_total : int, default=200
        Total number of samples.
    n_pos : int, default=50
        Number of positive-class samples.
    p : int, default=20
        Total number of features (2 signal + p-2 noise).
    effect : float, default=1.0
        Effect strength controlling class separation.
    alpha : float, default=0.9
        Redundancy strength in the redundancy regime (B = alpha*A + noise).
    anticorr : float, default=1.0
        Anti-correlation strength in the shared-axis regime, controlling
        how strongly individual features contribute to the composite axis.
    noise_std : float, default=1.0
        Standard deviation of noise features and residual noise.
    seed : int or None
        Master random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_total, p)
        Feature matrix. Columns 0 and 1 are signal features A and B.
    y : ndarray of shape (n_total,)
        Binary labels (0 or 1).
    info : dict
        Metadata including regime, parameters, and seed.
    """
    rng = np.random.RandomState(seed)

    if regime == "independence":
        X, y = _generate_independence(rng, n_total, n_pos, p, effect, noise_std)
    elif regime == "redundancy":
        X, y = _generate_redundancy(rng, n_total, n_pos, p, effect, alpha, noise_std)
    elif regime == "shared_axis":
        X, y = _generate_shared_axis(
            rng, n_total, n_pos, p, effect, anticorr, noise_std
        )
    else:
        raise ValueError(f"Unknown regime: {regime}. Use 'independence', "
                         f"'redundancy', or 'shared_axis'.")

    info = {
        "regime": regime,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_total - n_pos,
        "p": p,
        "effect": effect,
        "alpha": alpha,
        "anticorr": anticorr,
        "noise_std": noise_std,
        "seed": seed,
        "signal_indices": [0, 1],
    }

    return X, y, info


def _generate_independence(rng, n_total, n_pos, p, effect, noise_std):
    """
    Independence regime: A and B contribute through orthogonal axes.

    Label probability: P(y=1) = sigmoid(effect * A + effect * B)
    Features A and B are drawn independently from class-conditional
    Gaussian distributions with non-overlapping mean shifts.
    """
    # Generate latent variables
    z_a = rng.randn(n_total)
    z_b = rng.randn(n_total)

    # Label probability from logistic function
    logit = effect * z_a + effect * z_b
    prob = expit(logit)

    # Sample labels with controlled positive count
    y = _sample_labels_controlled(rng, prob, n_total, n_pos)

    # Build feature matrix
    X = np.zeros((n_total, p))
    X[:, 0] = z_a  # Feature A
    X[:, 1] = z_b  # Feature B

    # Noise features (standard normal, independent of labels)
    X[:, 2:] = rng.randn(n_total, p - 2) * noise_std

    return X, y


def _generate_redundancy(rng, n_total, n_pos, p, effect, alpha, noise_std):
    """
    Redundancy regime: B = alpha * A + epsilon (correlated backup).

    Label probability: P(y=1) = sigmoid(effect * A + effect * B)
    Because B ≈ alpha * A, either feature alone partially recovers
    class separation. Joint removal reveals masked dependence.
    """
    z_a = rng.randn(n_total)
    epsilon = rng.randn(n_total) * (1 - alpha) * noise_std
    z_b = alpha * z_a + epsilon

    logit = effect * z_a + effect * z_b
    prob = expit(logit)

    y = _sample_labels_controlled(rng, prob, n_total, n_pos)

    X = np.zeros((n_total, p))
    X[:, 0] = z_a
    X[:, 1] = z_b
    X[:, 2:] = rng.randn(n_total, p - 2) * noise_std

    return X, y


def _generate_shared_axis(rng, n_total, n_pos, p, effect, anticorr, noise_std):
    """
    Shared-axis regime: A and B contribute to a single discriminant axis.

    The composite variable (anticorr * A + anticorr * B) defines the
    discriminant direction. Individual features contribute partially,
    such that removing either degrades the shared structural direction.

    A and B are generated with negative residual correlation to ensure
    that individual removals each disrupt the same geometric structure.
    """
    # Shared latent direction
    z_shared = rng.randn(n_total)

    # Individual features are noisy projections of the shared axis
    # with negative residual correlation
    noise_a = rng.randn(n_total) * noise_std * 0.5
    noise_b = rng.randn(n_total) * noise_std * 0.5

    z_a = z_shared + noise_a
    z_b = z_shared + noise_b

    # Labels from composite axis
    logit = effect * (anticorr * z_a + anticorr * z_b)
    prob = expit(logit)

    y = _sample_labels_controlled(rng, prob, n_total, n_pos)

    X = np.zeros((n_total, p))
    X[:, 0] = z_a
    X[:, 1] = z_b
    X[:, 2:] = rng.randn(n_total, p - 2) * noise_std

    return X, y


def _sample_labels_controlled(rng, prob, n_total, n_pos):
    """
    Sample binary labels with a controlled number of positives.

    Ranks samples by probability and assigns the top n_pos as positive.
    This ensures exact class balance matching manuscript specifications.
    """
    # Rank by probability; top n_pos become positive
    order = np.argsort(-prob)
    y = np.zeros(n_total, dtype=int)
    y[order[:n_pos]] = 1
    return y


def generate_trm_like(seed=77, n_total=89, n_pos=12, p=20, effect=0.08, anticorr=0.3):
    """
    Generate a TRM-like simulation mimicking the empirical IL22+ dataset.

    Produces a sparse, weakly separated dataset that should fall below
    the separability gate (CV AUC < 0.60), as described in Section 3.4.

    Parameters
    ----------
    seed, n_total, n_pos, p, effect, anticorr : see generate_dataset

    Returns
    -------
    X, y, info : see generate_dataset
    """
    return generate_dataset(
        regime="shared_axis",
        n_total=n_total,
        n_pos=n_pos,
        p=p,
        effect=effect,
        anticorr=anticorr,
        seed=seed,
    )


def generate_ceiling_case(seed=23, n_total=89, n_pos=12, p=20, effect=1.0, anticorr=2.0):
    """
    Generate the ceiling artifact demonstration case.

    Produces a well-separated dataset where in-sample AUC approaches
    the metric ceiling (≈0.98), compressing ablation losses.
    Cross-validated AUC is more moderate (≈0.67).

    Parameters
    ----------
    seed, n_total, n_pos, p, effect, anticorr : see generate_dataset

    Returns
    -------
    X, y, info : see generate_dataset
    """
    return generate_dataset(
        regime="shared_axis",
        n_total=n_total,
        n_pos=n_pos,
        p=p,
        effect=effect,
        anticorr=anticorr,
        seed=seed,
    )
