"""Hard-negative mining (experiment F).

Hard negatives = legit TRAIN rows that a model scores high when it has NOT seen them (out-of-fold).
In-sample scores cannot be used: a flexible model memorises its own training rows, so legit rows look easy.
Mining uses train rows only; validation and test are never touched.
"""
from __future__ import annotations
import numpy as np
from .evaluate import threshold_at_fpr


def block_folds(steps: np.ndarray, k: int) -> np.ndarray:
    """Assign each row to one of k contiguous blocks of the step axis (time-ordered folds)."""
    if k < 2:
        raise ValueError("need at least 2 folds")
    steps = np.asarray(steps)
    edges = np.linspace(steps.min(), steps.max() + 1, k + 1)
    return np.clip(np.searchsorted(edges, steps, side="right") - 1, 0, k - 1)


def fit_weighted(model, X, y, sample_weight):
    """Fit a sklearn-style model (or a Pipeline whose last step takes sample_weight) with row weights."""
    if hasattr(model, "steps"):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": sample_weight})
    return model.fit(X, y, sample_weight=sample_weight)


def oof_scores(make_model, X, y, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold fraud scores: each fold is scored by a model trained on the other folds only."""
    scores = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        scores[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return scores


def mine_hard_negatives(oof: np.ndarray, y: np.ndarray, fpr: float) -> np.ndarray:
    """Boolean mask of legit rows whose out-of-fold score is above the threshold giving `fpr` on the legit rows."""
    thr = threshold_at_fpr(y, oof, fpr)
    return (y == 0) & (oof > thr)
