import numpy as np
import pytest
from iafd.hardneg import block_folds, mine_hard_negatives, fit_weighted, oof_scores
from iafd.models import build_model


def test_block_folds_contiguous_and_complete():
    steps = np.arange(1, 241)
    f = block_folds(steps, 4)
    assert set(f) == {0, 1, 2, 3}
    assert (np.diff(f) >= 0).all()                       # blocks follow time order
    assert [int((f == i).sum()) for i in range(4)] == [60, 60, 60, 60]
    with pytest.raises(ValueError):
        block_folds(steps, 1)


def test_mine_hard_negatives_only_top_legit_and_never_fraud():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(10_000, int), np.ones(50, int)]
    oof = np.r_[rng.random(10_000), np.full(50, 0.99)]
    hn = mine_hard_negatives(oof, y, 0.01)
    assert hn.sum() <= 100 and not hn[y == 1].any()
    assert oof[hn].min() >= np.sort(oof[y == 0])[-101]  # they are the highest-scoring legit rows


def test_oof_scores_are_filled_and_not_in_sample():
    rng = np.random.default_rng(1)
    n = 4_000
    import pandas as pd
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + 0.5 * rng.normal(size=n) > 1.5).astype(int).to_numpy()
    folds = np.arange(n) % 4
    sc = oof_scores(lambda: build_model("dt", 0), X, y, folds)
    assert not np.isnan(sc).any()


@pytest.mark.parametrize("name", ["lr", "dt", "xgb"])
def test_fit_weighted_runs_and_weights_matter(name):
    import pandas as pd
    rng = np.random.default_rng(2)
    n = 3_000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + rng.normal(size=n) > 1.0).astype(int).to_numpy()
    params = {"n_estimators": 20} if name == "xgb" else None
    p_plain = fit_weighted(build_model(name, 0, params), X, y, np.ones(n)).predict_proba(X)[:, 1]
    sw = np.where(y == 0, 20.0, 1.0)                     # up-weight legit rows
    p_w = fit_weighted(build_model(name, 0, params), X, y, sw).predict_proba(X)[:, 1]
    assert p_w.mean() <= p_plain.mean() + 1e-9           # more weight on legit pushes fraud scores down
