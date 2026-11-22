import numpy as np
import pandas as pd
import pytest
from iafd.explain import logit, ensemble_proba, make_ensemble_logit_fn, mean_abs, rank_agreement, pick_cases
from iafd.models import build_model
from iafd.data import compact_frame, chronological_split
from iafd.features import build_features
from iafd.synthetic import make_synthetic_paysim


@pytest.fixture(scope="module")
def fitted():
    df = compact_frame(make_synthetic_paysim(30_000, fraud_rate=0.01, seed=5))
    tr, va, te = chronological_split(df, 240, 355)
    X, y = build_features(tr), tr["isFraud"].to_numpy()
    models = [build_model("lr", 0).fit(X, y), build_model("xgb", 0, {"n_estimators": 20}).fit(X, y)]
    return models, build_features(te), te["isFraud"].to_numpy()


def test_logit_is_monotone_and_finite():
    p = np.array([0.0, 1e-9, 0.2, 0.9, 1.0])
    z = logit(p)
    assert np.isfinite(z).all() and (np.diff(z) >= 0).all()


def test_ensemble_fn_matches_mean_of_member_probabilities(fitted):
    models, Xte, _ = fitted
    f = make_ensemble_logit_fn(models, Xte.columns)
    X = Xte.iloc[:50]
    assert np.allclose(f(X.to_numpy()), logit(ensemble_proba(models, X)))


def test_rank_agreement_and_mean_abs():
    a = np.array([5.0, 3.0, 1.0, 0.5])
    assert rank_agreement(a, a * 2) == pytest.approx(1.0)
    assert rank_agreement(a, a[::-1]) == pytest.approx(-1.0)
    assert mean_abs(np.array([[1.0, -2.0], [-3.0, 2.0]])).tolist() == [2.0, 2.0]


def test_pick_cases():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.9, 0.2, 0.05, 0.6, 0.7, 0.8])
    c = pick_cases(y, s, thr=0.5)
    assert c["false_positive"] == 1 and c["missed_fraud"] == 3
    assert s[c["caught_fraud"]] > 0.5 and y[c["caught_fraud"]] == 1
    with pytest.raises(ValueError):
        pick_cases(np.zeros(3, int), np.zeros(3), 0.5)


def test_shap_permutation_values_are_additive_for_the_ensemble(fitted):
    shap = pytest.importorskip("shap")
    models, Xte, yte = fitted
    f = make_ensemble_logit_fn(models, Xte.columns)
    bg = Xte.iloc[:50].to_numpy()
    X = Xte.iloc[100:110].to_numpy()
    ex = shap.Explainer(f, shap.maskers.Independent(bg, max_samples=50), algorithm="permutation")(X, max_evals=200, silent=True)
    # efficiency: values + base reproduce the score up to the explainer's small numerical residual
    assert np.abs(ex.values.sum(axis=1) + np.asarray(ex.base_values) - f(X)).max() < 1e-2
