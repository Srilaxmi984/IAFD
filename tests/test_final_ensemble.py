"""Tests for the final ensemble (synthetic PaySim-like data, no real data needed)."""
import numpy as np
import pandas as pd
import pytest

from iafd.features import build_features
from iafd.final_ensemble import FinalEnsemble


def _synth(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    types = rng.choice(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"], n,
                       p=[.35, .2, .3, .1, .05])
    amt = rng.exponential(1e5, n)
    old_o = rng.exponential(2e5, n)
    new_o = np.maximum(old_o - amt, 0)
    old_d = rng.exponential(2e5, n)
    new_d = old_d + amt
    fraud = (np.isin(types, ["CASH_OUT", "TRANSFER"]) & (rng.random(n) < 0.05)).astype(int)
    f = fraud == 1
    amt[f], new_o[f], new_d[f], old_d[f] = old_o[f], 0, 0, 0
    return pd.DataFrame({
        "step": rng.integers(1, 744, n), "type": types, "amount": amt,
        "oldbalanceOrg": old_o, "newbalanceOrig": new_o,
        "oldbalanceDest": old_d, "newbalanceDest": new_d, "isFraud": fraud})


@pytest.fixture(scope="module")
def parts():
    df = _synth()
    tr, va, te = df[df.step <= 400], df[(df.step > 400) & (df.step <= 600)], df[df.step > 600]
    X = [build_features(d, "plus") for d in (tr, va, te)]
    y = [d["isFraud"].to_numpy() for d in (tr, va, te)]
    return X, y


@pytest.fixture(scope="module")
def fitted(parts):
    (Xtr, Xva, _), (ytr, yva, _) = parts
    return FinalEnsemble(n_estimators=40, seed=1).fit(Xtr, ytr, Xva, yva, fpr_cap=1e-3)


def test_gate_learns_only_fraud_types(fitted):
    assert set(fitted.gate_cols) == {"type_CASH_OUT", "type_TRANSFER"}


def test_non_fraud_types_score_exactly_zero(fitted, parts):
    (_, _, Xte), _ = parts
    p = fitted.predict_proba(Xte)
    off = Xte[["type_CASH_OUT", "type_TRANSFER"]].to_numpy().sum(axis=1) == 0
    assert off.any() and np.all(p[off] == 0.0)


def test_weights_valid(fitted):
    w = np.asarray(fitted.weights)
    assert np.all(w >= 0) and abs(w.sum() - 1.0) < 1e-9


def test_scores_are_probabilities(fitted, parts):
    (_, _, Xte), _ = parts
    p = fitted.predict_proba(Xte)
    assert np.all((p >= 0) & (p <= 1))


def test_validation_fpr_respects_cap(fitted, parts):
    (_, Xva, _), (_, yva, _) = parts
    p = fitted.predict_proba(Xva)
    fpr = ((p > fitted.threshold) & (yva == 0)).sum() / (yva == 0).sum()
    assert fpr <= 1e-3


def test_catches_most_fraud(fitted, parts):
    (_, _, Xte), (_, _, yte) = parts
    pred = fitted.predict(Xte)
    assert (pred[yte == 1] == 1).mean() > 0.9


def test_save_load_roundtrip(fitted, parts, tmp_path):
    (_, _, Xte), _ = parts
    fitted.save(tmp_path / "m.joblib")
    again = FinalEnsemble.load(tmp_path / "m.joblib")
    assert np.allclose(fitted.predict_proba(Xte), again.predict_proba(Xte))
    assert again.threshold == fitted.threshold


def test_prob_combine_mode(parts):
    (Xtr, Xva, Xte), (ytr, yva, _) = parts
    m = FinalEnsemble(combine="prob", members=["rf", "lgbm"], n_estimators=30).fit(Xtr, ytr, Xva, yva, 1e-3)
    p = m.predict_proba(Xte)
    assert np.all((p >= 0) & (p <= 1))


def test_bad_combine_raises():
    with pytest.raises(ValueError):
        FinalEnsemble(combine="nope")
