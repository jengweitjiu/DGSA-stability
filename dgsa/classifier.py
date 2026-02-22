"""
Cell state classification and geometric margin computation.

Implements L2-regularized logistic regression with margin-based
stability depth as described in Section 2.1 and Methods.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class StabilityClassifier:
    """
    Binary cell state classifier with geometric stability metrics.

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularization strength for L2-regularized logistic regression.
    n_splits : int, default=5
        Number of folds for stratified cross-validation.
    random_state : int or None, default=None
        Random seed for cross-validation fold assignment.

    Attributes
    ----------
    baseline_depth_ : float
        Cross-validated mean signed margin over positive-class samples.
    baseline_auc_ : float
        Cross-validated ROC AUC.
    is_baseline_depth_ : float
        In-sample baseline depth (for artifact demonstration only).
    is_baseline_auc_ : float
        In-sample ROC AUC (for artifact demonstration only).
    """

    def __init__(self, C=1.0, n_splits=5, random_state=None):
        self.C = C
        self.n_splits = n_splits
        self.random_state = random_state

    def _make_clf(self):
        """Create a fresh logistic regression instance."""
        return LogisticRegression(
            penalty="l2",
            C=self.C,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )

    def _compute_margins(self, clf, X):
        """
        Compute signed geometric margins for each sample.

        Margin of sample x_i: m_i = (w^T x_i + b) / ||w||
        """
        w = clf.coef_[0]
        b = clf.intercept_[0]
        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            return np.zeros(X.shape[0])
        return (X @ w + b) / w_norm

    def evaluate_cv(self, X, y):
        """
        Compute cross-validated stability metrics.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (e.g., regulon activity scores).
        y : ndarray of shape (n_samples,)
            Binary labels (0 or 1).

        Returns
        -------
        depth : float
            Cross-validated stability depth (mean signed margin
            over positive-class held-out samples).
        auc : float
            Cross-validated ROC AUC.
        """
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        all_margins = np.full(len(y), np.nan)
        all_proba = np.full(len(y), np.nan)

        for train_idx, test_idx in skf.split(X, y):
            clf = self._make_clf()
            clf.fit(X[train_idx], y[train_idx])

            # Held-out margins
            margins = self._compute_margins(clf, X[test_idx])
            all_margins[test_idx] = margins

            # Held-out probabilities for AUC
            proba = clf.predict_proba(X[test_idx])[:, 1]
            all_proba[test_idx] = proba

        # Stability depth: mean signed margin over positive class
        pos_mask = y == 1
        depth = np.nanmean(all_margins[pos_mask])

        # ROC AUC from out-of-fold probabilities
        valid = ~np.isnan(all_proba)
        if valid.sum() > 0 and len(np.unique(y[valid])) == 2:
            auc = roc_auc_score(y[valid], all_proba[valid])
        else:
            auc = 0.5

        self.baseline_depth_ = depth
        self.baseline_auc_ = auc
        return depth, auc

    def evaluate_insample(self, X, y):
        """
        Compute in-sample stability metrics (for artifact demonstration).

        This method trains on all data and evaluates on the same data,
        producing inflated metrics that exhibit ceiling compression
        artifacts as described in Section 3.4.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        depth : float
            In-sample stability depth.
        auc : float
            In-sample ROC AUC.
        """
        clf = self._make_clf()
        clf.fit(X, y)

        margins = self._compute_margins(clf, X)
        pos_mask = y == 1
        depth = np.mean(margins[pos_mask])

        proba = clf.predict_proba(X)[:, 1]
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y, proba)
        else:
            auc = 0.5

        self.is_baseline_depth_ = depth
        self.is_baseline_auc_ = auc
        return depth, auc
