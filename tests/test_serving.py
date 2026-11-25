import numpy as np
import pandas as pd
import pytest
from iafd.data import compact_frame, chronological_split
from iafd.features import build_features
from iafd.models import build_model
from iafd.serving import format_probability, verdict, auto_balances, make_row, score_row, decide, features_for, model_proba, default_background, explain_row
from iafd.synthetic import make_synthetic_paysim


@pytest.fixture(scope="module")
def bundles():
    df = compact_frame(make_synthetic_paysim(40_000, fraud_rate=0.01, seed=5))
    tr, _, _ = chronological_split(df, 240, 355)
    allowed = sorted(set(tr["type"].astype(str)[tr["isFraud"] == 1]))
    tr = tr[tr["type"].astype(str).isin(allowed)]
    X, y = build_features(tr, "plus"), tr["isFraud"].to_numpy()
    cols = list(X.columns)
    rf = build_model("rf", 0, {"n_estimators": 20, "max_depth": 8}).fit(X, y)
    lr = build_model("lr", 0).fit(X, y)
    common = {"feature_set": "plus", "columns": cols, "gate": True, "allowed_types": allowed,
              "thresholds": {"cap": 0.5, "recall": 0.2, "min_cost": float("inf")}}
    return {"single": {**common, "model": "rf", "strategy": "plain", "estimator": rf},
            "ensemble": {**common, "members": [{"model": "rf", "strategy": "plain"}, {"model": "lr", "strategy": "plain"}],
                         "models": [rf, lr], "aggregator": "mean", "weights": None}}


def test_make_row_validation():
    r = make_row("TRANSFER", 100, 100, 0, 0, 0)
    assert r.shape[0] == 1 and r["type"].iloc[0] == "TRANSFER"
    with pytest.raises(ValueError):
        make_row("WIRE", 1, 1, 0, 0, 0)


def test_gated_type_scores_zero_and_is_never_flagged(bundles):
    for b in bundles.values():
        res = score_row(b, make_row("PAYMENT", 64, 64.01, 0.01, 0, 0))
        assert res == {"score": 0.0, "gated": True}
        assert decide(b, res["score"], "cap")[0] is False


def test_fraud_like_transfer_scores_higher_than_ordinary_cash_out(bundles):
    for b in bundles.values():
        hi = score_row(b, make_row("TRANSFER", 250000, 250000, 0, 0, 0))["score"]
        lo = score_row(b, make_row("CASH_OUT", 500, 80000, 79500, 20000, 20500))["score"]
        assert hi > lo


def test_decide_uses_strict_threshold_and_inf_means_never(bundles):
    b = bundles["single"]
    assert decide(b, 0.6, "cap") == (True, 0.5)
    assert decide(b, 0.5, "cap")[0] is False
    assert decide(b, 1.0, "min_cost")[0] is False


def test_explanations_are_additive_for_the_tree_and_the_ensemble(bundles):
    pytest.importorskip("shap")
    row = make_row("TRANSFER", 250000, 250000, 0, 0, 0)
    for key in ("single", "ensemble"):
        b = bundles[key]
        bg = default_background(b, 40)
        sv, base, scale = explain_row(b, row, bg, max_evals=200)
        X = features_for(b, row)
        p = float(model_proba(b, X)[0])
        target = p if scale == "probability" else float(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))))
        assert abs(sv.sum() + base - target) < 0.05
        assert list(sv.index) == b["columns"]


def test_auto_balances_follow_the_transaction_type():
    assert auto_balances("TRANSFER", 100, 250, 40) == (150.0, 140.0)
    assert auto_balances("CASH_OUT", 100, 250, 40) == (150.0, 140.0)
    assert auto_balances("CASH_IN", 100, 250, 40) == (350.0, 40.0)
    assert auto_balances("PAYMENT", 64, 900, 0) == (836.0, 0.0)
    assert auto_balances("DEBIT", 10, 5, 7) == (0.0, 7.0)              # sender balance never goes below 0
    assert auto_balances("TRANSFER", -5, 100, 0) == (100.0, 0.0)       # negative amounts are ignored


def test_format_probability_never_prints_a_misleading_zero():
    assert format_probability(0.97234) == "0.9723"
    assert format_probability(0.0019) == "0.0019"
    assert format_probability(1.5e-6) == "1.50e-06"
    assert format_probability(0.0) == "0"


def test_verdict_texts():
    f = verdict(0.4, 0.1, True)
    assert f["flagged"] and f["headline"].startswith("FLAGGED") and f["ratio_text"] == "4x the threshold"
    n = verdict(1.5e-6, 1.9e-3, False)
    assert not n["flagged"] and n["headline"].startswith("NOT FLAGGED") and "below the flag threshold" in n["explanation"]
    assert n["ratio_text"] == "0.000789x the threshold"
    g = verdict(0.0, 0.5, False, gated=True)
    assert g["headline"].startswith("NOT FLAGGED") and g["ratio_text"] == "scored 0"
    never = verdict(0.9, float("inf"), False)
    assert never["ratio_text"] == "no threshold" and "never flags" in never["explanation"]
