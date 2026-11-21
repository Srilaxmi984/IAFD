"""Evaluation under a false-positive-rate cap.

Why not accuracy: with ~0.1% fraud, flagging nothing is 99.9% accurate.
We pick the decision threshold on VALIDATION so that FPR <= cap, then report
recall / FPR / precision on TEST at that fixed threshold.

Fraud prevalence differs between splits (train/val ~0.07-0.09%, test ~0.34%), so
precision and PR-AUC are not comparable across splits. Recall and FPR are.
`precision_at_prevalence` re-computes precision for a chosen prevalence.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_cap: float) -> float:
    """Return thr such that flagging `score > thr` gives FPR <= fpr_cap on this data.
    Exact even with tied scores (tree models output many identical probabilities)."""
    if not 0 <= fpr_cap < 1:
        raise ValueError("fpr_cap must be in [0, 1)")
    neg = np.sort(scores[y_true == 0])[::-1]          # legit scores, descending
    if len(neg) == 0:
        raise ValueError("No legit rows to calibrate the threshold on")
    k = int(np.floor(fpr_cap * len(neg)))             # number of false positives we may afford
    if k >= len(neg):
        return -np.inf
    return float(neg[k])                              # strictly-greater flags at most k legit rows


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> float:
    tp, fp = recall * prevalence, fpr * (1 - prevalence)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def confusion_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    pred = scores > thr
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "fpr": fpr, "precision": precision}


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, thr: float,
                    reference_prevalence: float | None = None) -> dict:
    out = confusion_at_threshold(y_true, scores, thr)
    out["pr_auc"] = float(average_precision_score(y_true, scores))
    out["roc_auc"] = float(roc_auc_score(y_true, scores))
    out["prevalence"] = float(np.mean(y_true))
    if reference_prevalence is not None:
        out["precision_at_ref_prev"] = precision_at_prevalence(out["recall"], out["fpr"], reference_prevalence)
    return out
