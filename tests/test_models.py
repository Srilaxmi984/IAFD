import numpy as np
import pytest
from iafd.data import compact_frame, chronological_split
from iafd.features import build_features
from iafd.models import ALL_MODELS, build_model, soft_vote
from iafd.evaluate import threshold_at_fpr, confusion_at_threshold
from iafd.synthetic import make_synthetic_paysim


@pytest.fixture(scope="module")
def data():
    df = compact_frame(make_synthetic_paysim(30_000, fraud_rate=0.01, seed=5))
    tr, va, te = chronological_split(df, 240, 355)
    return (build_features(tr), tr["isFraud"].to_numpy(),
            build_features(va), va["isFraud"].to_numpy())


@pytest.mark.parametrize("name", ALL_MODELS)
def test_each_model_trains_and_detects(data, name):
    Xtr, ytr, Xva, yva = data
    p = build_model(name, seed=0, params={"n_estimators": 30} if name in {"rf", "xgb", "lgbm"} else None
                    ).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    assert p.shape == (len(Xva),) and 0 <= p.min() and p.max() <= 1
    thr = threshold_at_fpr(yva, p, 0.005)
    c = confusion_at_threshold(yva, p, thr)
    assert c["fpr"] <= 0.005
    assert c["recall"] > 0.5          # synthetic fraud is easy; guards against a broken pipeline


def test_soft_vote_is_mean():
    out = soft_vote([np.array([0.0, 1.0]), np.array([1.0, 1.0])])
    assert np.allclose(out, [0.5, 1.0])
    with pytest.raises(ValueError):
        soft_vote([])


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("svm")


def test_pos_weight_validation():
    with pytest.raises(ValueError):
        build_model("lr", pos_weight=0)
    with pytest.raises(ValueError):
        build_model("xgb", pos_weight=-1)


@pytest.mark.parametrize("name", ["lr", "xgb"])
def test_pos_weight_raises_scores(data, name):
    Xtr, ytr, Xva, yva = data
    params = {"n_estimators": 30} if name == "xgb" else None
    p1 = build_model(name, seed=0, params=params).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    p50 = build_model(name, seed=0, params=params, pos_weight=50).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    assert p50.mean() >= p1.mean() - 1e-9          # penalising missed fraud pushes scores up


@pytest.mark.parametrize("name", ALL_MODELS)
def test_every_model_accepts_pos_weight(data, name):
    Xtr, ytr, Xva, yva = data
    params = {"n_estimators": 20} if name in {"rf", "xgb", "lgbm"} else None
    p = build_model(name, seed=0, params=params, pos_weight=20).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    assert p.shape == (len(Xva),)
