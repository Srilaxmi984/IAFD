import numpy as np
import pytest
from iafd.evaluate import (wilson_interval, threshold_at_recall, fp_at_recall_table, recall_fpr_table, threshold_at_fpr, confusion_at_threshold, evaluate_scores,
                           precision_at_prevalence)


def test_threshold_respects_cap_distinct_scores():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(10_000, int), np.ones(100, int)]
    s = np.r_[rng.random(10_000) * 0.8, 0.5 + rng.random(100) * 0.5]
    thr = threshold_at_fpr(y, s, 0.005)
    assert confusion_at_threshold(y, s, thr)["fpr"] <= 0.005


def test_threshold_exact_with_ties():
    # tree models emit many identical probabilities; cap must still hold
    y = np.r_[np.zeros(1000, int), np.ones(10, int)]
    s = np.r_[np.zeros(900), np.full(100, 0.3), np.ones(10)]
    thr = threshold_at_fpr(y, s, 0.05)
    assert confusion_at_threshold(y, s, thr)["fpr"] <= 0.05
    thr0 = threshold_at_fpr(y, s, 0.0)
    assert confusion_at_threshold(y, s, thr0)["fp"] == 0


def test_confusion_counts_and_recall():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    c = confusion_at_threshold(y, s, 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 2)
    assert c["recall"] == 0.5 and c["precision"] == 0.5


def test_precision_at_prevalence():
    assert precision_at_prevalence(1.0, 0.0, 0.01) == 1.0
    assert precision_at_prevalence(0.0, 0.0, 0.01) == 0.0
    # recall 0.8, fpr 0.005, prevalence 0.1% -> 0.0008 / (0.0008 + 0.004995)
    assert precision_at_prevalence(0.8, 0.005, 0.001) == pytest.approx(0.0008 / (0.0008 + 0.004995))


def test_evaluate_scores_keys():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.4, 0.35, 0.8])
    out = evaluate_scores(y, s, 0.5, reference_prevalence=0.001)
    assert {"pr_auc", "roc_auc", "recall", "fpr", "precision_at_ref_prev"} <= set(out)


def test_threshold_validation():
    with pytest.raises(ValueError):
        threshold_at_fpr(np.array([1, 1]), np.array([0.1, 0.2]), 0.01)
    with pytest.raises(ValueError):
        threshold_at_fpr(np.array([0, 1]), np.array([0.1, 0.2]), 1.5)


def test_recall_fpr_table_monotone_and_capped():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(20_000, int), np.ones(200, int)]
    s = np.r_[rng.random(20_000), 0.4 + 0.6 * rng.random(200)]
    rows = recall_fpr_table(y, s, y, s, [0.0001, 0.001, 0.01, 0.05])
    assert all(r["val_fpr"] <= r["fpr_cap"] for r in rows)
    rec = [r["val_recall"] for r in rows]
    assert rec == sorted(rec)                      # looser cap never lowers recall


def test_threshold_at_recall_exact_with_ties():
    y = np.r_[np.zeros(500, int), np.ones(10, int)]
    s = np.r_[np.linspace(0, 0.9, 500), [0.2, 0.2, 0.2, 0.5, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]]
    for tgt in (0.5, 0.8, 0.9, 1.0):
        thr = threshold_at_recall(y, s, tgt)
        assert confusion_at_threshold(y, s, thr)["recall"] >= tgt
    with pytest.raises(ValueError):
        threshold_at_recall(y, s, 0.0)
    with pytest.raises(ValueError):
        threshold_at_recall(np.zeros(5, int), np.arange(5.0), 0.9)


def test_fp_at_recall_table_fp_grows_with_target():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(20_000, int), np.ones(200, int)]
    s = np.r_[rng.random(20_000), 0.3 + 0.7 * rng.random(200)]
    rows = fp_at_recall_table(y, s, y, s, [0.9, 0.99, 0.999])
    assert all(r["val_recall"] >= r["recall_target"] for r in rows)
    fps = [r["test_fp"] for r in rows]
    assert fps == sorted(fps)


def test_wilson_interval():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and 0.39 < lo < 0.41 and 0.59 < hi < 0.61
    lo, hi = wilson_interval(4248, 4250)
    assert 0.998 < lo < 0.9995 and hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo0, hi0 = wilson_interval(0, 1000)
    assert lo0 == 0.0 and 0 < hi0 < 0.005
