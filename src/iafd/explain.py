"""Helpers for explaining the soft-voting ensemble with SHAP.

The ensemble score is the MEAN of the members' fraud probabilities. We explain it on the log-odds scale
(logit of that mean, clipped), with a model-agnostic permutation explainer. Values are relative to a background
of typical (train) rows, so a SHAP value answers: 'how much did this feature move this transaction's score
away from a typical transaction?'.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def logit(p, eps: float = 1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def ensemble_proba(models, X) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def make_ensemble_logit_fn(models, columns):
    """Function numpy (n, p) -> logit of the ensemble's mean fraud probability, as shap expects."""
    columns = list(columns)

    def f(X):
        return logit(ensemble_proba(models, pd.DataFrame(np.asarray(X, dtype="float64"), columns=columns)))
    return f


def mean_abs(values: np.ndarray) -> np.ndarray:
    return np.abs(values).mean(axis=0)


def rank_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two importance vectors (1.0 = same feature ranking)."""
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def pick_cases(y: np.ndarray, score: np.ndarray, thr: float) -> dict[str, int]:
    """Indices of three illustrative test rows: a typical CAUGHT fraud (median score among caught), the
    lowest-scoring fraud (a MISSED fraud if any fraud is below the threshold), and the highest-scoring legit row."""
    fraud = np.flatnonzero(y == 1)
    legit = np.flatnonzero(y == 0)
    if len(fraud) == 0 or len(legit) == 0:
        raise ValueError("need both fraud and legit rows")
    caught = fraud[score[fraud] > thr]
    out = {"missed_fraud": int(fraud[np.argmin(score[fraud])]),
           "false_positive": int(legit[np.argmax(score[legit])])}
    pool = caught if len(caught) else fraud
    order = pool[np.argsort(score[pool])]
    out["caught_fraud"] = int(order[len(order) // 2])
    return out
