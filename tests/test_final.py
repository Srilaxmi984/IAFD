import numpy as np
import pytest
from iafd.final import (lfr_score, mcc, f_beta, cost_at_threshold, threshold_min_cost, fraud_subsample_mask,
                        strategy_pos_weight, combine, all_subsets, LOW_FPR_CAPS,
                        selection_profile, positive_simplex, weighted_logit, combine_spec, allowed_types, allowed_mask, predict_gated)
from iafd.evaluate import threshold_at_fpr, threshold_at_recall, confusion_at_threshold
from iafd.explain import logit


def test_lfr_matches_brute_force():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(20_000, int), np.ones(150, int)]
    p = np.r_[rng.random(20_000) * 0.9, 0.3 + 0.7 * rng.random(150)]
    brute = np.mean([confusion_at_threshold(y, p, threshold_at_fpr(y, p, c))["recall"] for c in LOW_FPR_CAPS])
    assert lfr_score(y, p) == pytest.approx(brute)
    perfect = np.r_[np.zeros(20_000), np.ones(150)]
    assert lfr_score(y, perfect) == 1.0
    with pytest.raises(ValueError):
        lfr_score(np.zeros(5, int), np.arange(5.0))


def test_mcc_and_fbeta_known_values():
    assert mcc(10, 0, 0, 90) == pytest.approx(1.0)
    assert mcc(0, 10, 10, 0) == pytest.approx(-1.0)
    assert mcc(0, 0, 0, 0) == 0.0
    assert f_beta(5, 5, 5, beta=1.0) == pytest.approx(0.5)
    assert f_beta(0, 0, 3, beta=2.0) == 0.0


def test_cost_and_min_cost_threshold_brute_force():
    rng = np.random.default_rng(1)
    n = 400
    y = (rng.random(n) < 0.08).astype(int)
    p = np.round(np.where(y == 1, rng.beta(5, 2, n), rng.beta(1, 6, n)), 2)          # many ties on purpose
    amt = rng.uniform(100, 5000, n)
    for alert in (0.0, 50.0, 2000.0, 1e9):
        thr = threshold_min_cost(y, p, amt, alert)
        best = cost_at_threshold(y, p, amt, thr, alert)["total_cost"]
        cands = np.unique(np.r_[p, -np.inf, np.inf])
        brute = min(cost_at_threshold(y, p, amt, np.nextafter(c, -np.inf) if np.isfinite(c) else c, alert)["total_cost"]
                    for c in cands)
        assert best <= brute + 1e-6
    p2 = p.copy()
    p2[np.flatnonzero(y == 0)[0]] = 1.5                                    # the top-scored row is legit
    assert threshold_min_cost(y, p2, amt, 1e12) == float("inf")           # alerts too expensive: flag nothing


def test_cost_breakdown():
    y = np.array([1, 1, 0, 0]); p = np.array([0.9, 0.1, 0.8, 0.2]); a = np.array([1000., 500., 10., 10.])
    c = cost_at_threshold(y, p, a, 0.5, alert_cost=100)
    assert c["missed_value"] == 500 and c["alert_cost_total"] == 100 and c["total_cost"] == 600
    assert c["baseline_cost"] == 1500 and c["saving_vs_no_model"] == 900


def test_fraud_subsample_and_strategy():
    y = np.r_[np.zeros(1000, int), np.ones(100, int)]
    keep = fraud_subsample_mask(y, 0.1, seed=0)
    assert keep[y == 0].all() and keep[y == 1].sum() == 10
    assert (fraud_subsample_mask(y, 0.1, 0) == keep).all()
    assert fraud_subsample_mask(y, 1.0, 0).all()
    with pytest.raises(ValueError):
        fraud_subsample_mask(y, 0.0, 0)
    assert strategy_pos_weight(y, "plain") == 1.0
    assert strategy_pos_weight(y, "sqrtw") == pytest.approx(np.sqrt(10))
    with pytest.raises(ValueError):
        strategy_pos_weight(y, "bogus")


def test_combine_and_subsets():
    a, b = np.array([0.1, 0.9]), np.array([0.3, 0.7])
    assert np.allclose(combine([a, b], "mean"), [0.2, 0.8])
    lm = combine([a, b], "logit_mean")
    assert np.all((lm > 0) & (lm < 1)) and lm[0] < 0.5 < lm[1]
    with pytest.raises(ValueError):
        combine([a], "bogus")
    assert len(list(all_subsets(["a", "b", "c", "d"]))) == 11               # sizes 2, 3, 4


def test_selection_profile_matches_the_individual_functions():
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(30_000, int), np.ones(400, int)]
    p = np.round(np.r_[rng.random(30_000) * 0.8, 0.2 + 0.8 * rng.random(400)], 3)     # ties on purpose
    lfr, f5, f9 = selection_profile(y, p)
    assert lfr == pytest.approx(lfr_score(y, p))
    for target, got in ((0.995, f5), (0.999, f9)):
        thr = threshold_at_recall(y, p, target)
        assert got == confusion_at_threshold(y, p, thr)["fp"]


def test_positive_simplex_counts_and_sums():
    for k, n in ((2, 9), (3, 36), (4, 84)):
        w = positive_simplex(k)
        assert len(w) == n and all(abs(sum(x) - 1) < 1e-9 and min(x) > 0 for x in w)
    assert positive_simplex(1) == [(1.0,)]


def test_weighted_logit_equals_logit_mean_for_equal_weights():
    a, b = np.array([0.1, 0.9, 0.4]), np.array([0.3, 0.7, 0.8])
    assert np.allclose(weighted_logit([a, b], [0.5, 0.5]), combine([a, b], "logit_mean"))
    assert np.allclose(weighted_logit([a, b], [1, 0]), np.clip(a, 1e-6, 1 - 1e-6))
    assert np.allclose(combine_spec([a, b], "weighted_logit", [0.3, 0.7]), weighted_logit([a, b], [0.3, 0.7]))
    assert np.allclose(combine_spec([a, b], "mean"), (a + b) / 2)


def test_type_gating():
    types = np.array(["TRANSFER", "PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER"])
    y = np.array([1, 0, 1, 0, 0])
    allowed = allowed_types(types, y)
    assert allowed == ["CASH_OUT", "TRANSFER"]
    mask = allowed_mask(types, allowed)
    assert mask.tolist() == [True, False, True, False, True]

    class Fake:
        def predict_proba(self, X):
            return np.column_stack([1 - X["a"].to_numpy(), X["a"].to_numpy()])
    import pandas as pd
    X = pd.DataFrame({"a": [0.9, 0.8, 0.7, 0.6, 0.5]})
    assert predict_gated(Fake(), X, mask).tolist() == [0.9, 0.0, 0.7, 0.0, 0.5]
    assert not predict_gated(Fake(), X, np.zeros(5, bool)).any()
