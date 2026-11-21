import numpy as np
import pytest
from iafd.evaluate import (threshold_at_fpr, confusion_at_threshold, evaluate_scores,
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
