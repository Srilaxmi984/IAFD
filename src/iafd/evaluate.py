"""Evaluation under a false-positive-rate cap.

Why not accuracy: with ~0.1% fraud, flagging nothing is 99.9% accurate.
We pick the decision threshold on VALIDATION so that FPR <= cap, then report
recall / FPR / precision on TEST at that fixed threshold.

Fraud prevalence differs between splits (train/val ~0.07-0.09%, test ~0.34%), so
precision and PR-AUC are not comparable across splits. Recall and FPR are.
`precision_at_prevalence` re-computes precision for a chosen prevalence."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


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


def recall_fpr_table(y_val, p_val, y_test, p_test, fpr_caps) -> list[dict]:
    """For each FPR cap: threshold chosen on VALIDATION, then recall/FPR on val and test.
    Shows how much recall survives as the false-alarm budget shrinks."""
    rows = []
    for cap in fpr_caps:
        thr = threshold_at_fpr(y_val, p_val, cap)
        v, t = confusion_at_threshold(y_val, p_val, thr), confusion_at_threshold(y_test, p_test, thr)
        rows.append({"fpr_cap": cap, "threshold": thr,
                     "val_recall": v["recall"], "val_fpr": v["fpr"],
                     "test_recall": t["recall"], "test_fpr": t["fpr"], "test_fp": t["fp"], "test_fn": t["fn"]})
    return rows


def threshold_at_recall(y_true: np.ndarray, scores: np.ndarray, target: float) -> float:
    """Return thr such that flagging `score > thr` gives recall >= target on this data
    (exact with ties). Use on VALIDATION, then apply the threshold to test."""
    if not 0 < target <= 1:
        raise ValueError("target must be in (0, 1]")
    pos = np.sort(scores[y_true == 1])                 # fraud scores, ascending
    if len(pos) == 0:
        raise ValueError("No fraud rows to calibrate the threshold on")
    m = int(np.floor((1 - target) * len(pos)))         # frauds we may miss
    if m >= len(pos):
        return np.inf
    return float(np.nextafter(pos[m], -np.inf))        # just below the (m+1)-th lowest fraud score


def fp_at_recall_table(y_val, p_val, y_test, p_test, targets) -> list[dict]:
    """For each recall target: threshold chosen on VALIDATION, then FPR / false positives on test."""
    rows = []
    for tgt in targets:
        thr = threshold_at_recall(y_val, p_val, tgt)
        v, t = confusion_at_threshold(y_val, p_val, thr), confusion_at_threshold(y_test, p_test, thr)
        rows.append({"recall_target": tgt, "threshold": thr,
                     "val_recall": v["recall"], "val_fpr": v["fpr"],
                     "test_recall": t["recall"], "test_fpr": t["fpr"], "test_fp": t["fp"], "test_fn": t["fn"]})
    return rows


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n at a FIXED threshold.
    It ignores threshold-selection and model-training variability (use several seeds for that)."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
