import numpy as np
import pandas as pd
import pytest
from iafd.structuring import split_frauds, event_max
from iafd.features import build_features
from iafd.data import compact_frame
from iafd.synthetic import make_synthetic_paysim


@pytest.fixture(scope="module")
def frauds():
    df = compact_frame(make_synthetic_paysim(60_000, fraud_rate=0.01, seed=7))
    return df[df["isFraud"] == 1].reset_index(drop=True)


def cents(s):
    return np.round(s.to_numpy() * 100).astype(np.int64)


@pytest.mark.parametrize("equal", [True, False])
def test_pieces_sum_to_total_and_balances_chain(frauds, equal):
    p = split_frauds(frauds, np.random.default_rng(0), [2, 5, 8], residual=0.0, equal=equal)
    assert len(p) > len(frauds)
    for ev, g in p.groupby("event_id"):
        f = frauds.iloc[ev]
        assert cents(g["amount"]).sum() == round(f["amount"] * 100)                    # pieces add up exactly
        before = np.r_[0, np.cumsum(cents(g["amount"]))[:-1]]
        assert (cents(g["oldbalanceOrg"]) == round(f["oldbalanceOrg"] * 100) - before).all()   # balances chain
        assert (cents(g["amount"]) >= 1).all()
        assert g["newbalanceOrig"].iloc[-1] == 0.0                                      # drain: last piece empties the account


def test_last_piece_drains_remaining_balance_so_feature_still_sees_it(frauds):
    p = split_frauds(frauds, np.random.default_rng(1), [4], residual=0.0, equal=True)
    X = build_features(p, "plus")
    last = p.groupby("event_id").tail(1).index
    first = p.groupby("event_id").head(1).index
    assert (X.loc[last, "drains_full_balance"] == 1).all()
    assert (X.loc[first, "drains_full_balance"] == 0).all()


def test_residual_never_empties_the_account(frauds):
    p = split_frauds(frauds, np.random.default_rng(2), [4, 16], residual=0.2, equal=False)
    assert (p["newbalanceOrig"] > 0).all()
    X = build_features(p, "plus")
    assert X["drains_full_balance"].sum() == 0
    for ev, g in p.groupby("event_id"):
        f = frauds.iloc[ev]
        assert cents(g["amount"]).sum() == round(round(f["amount"] * 100) * 0.8)


def test_dest_balance_follows_the_original_pattern(frauds):
    p = split_frauds(frauds, np.random.default_rng(3), [3], residual=0.0, equal=True)
    for ev, g in p.groupby("event_id"):
        f = frauds.iloc[ev]
        if f["newbalanceDest"] != f["oldbalanceDest"]:
            assert round(g["newbalanceDest"].iloc[-1] * 100) == round(f["newbalanceDest"] * 100)
        else:
            assert (g["newbalanceDest"] == g["oldbalanceDest"]).all()


def test_reproducible_and_k1_is_identity_like(frauds):
    a = split_frauds(frauds, np.random.default_rng(5), [2, 4], 0.0, False)
    b = split_frauds(frauds, np.random.default_rng(5), [2, 4], 0.0, False)
    pd.testing.assert_frame_equal(a, b)
    one = split_frauds(frauds, np.random.default_rng(5), [1], 0.0, True)
    assert len(one) == len(frauds)
    assert np.allclose(one["amount"].to_numpy(), frauds["amount"].to_numpy(), atol=0.011)


def test_event_max_and_validation(frauds):
    ev = np.array([0, 0, 0, 1, 2, 2])
    sc = np.array([0.1, 0.9, 0.3, 0.2, 0.5, 0.4])
    assert event_max(sc, ev).tolist() == [0.9, 0.2, 0.5]
    with pytest.raises(ValueError):
        split_frauds(frauds, np.random.default_rng(0), [0])
    with pytest.raises(ValueError):
        split_frauds(frauds, np.random.default_rng(0), [2], residual=1.0)
    with pytest.raises(ValueError):
        split_frauds(frauds.iloc[0:0], np.random.default_rng(0), [2])


def test_skew_puts_most_of_the_amount_in_the_last_piece(frauds):
    p = split_frauds(frauds, np.random.default_rng(0), [4], residual=0.2, equal=True, skew=0.95)
    for ev, g in p.groupby("event_id"):
        share = g["amount"].iloc[-1] / g["amount"].sum()
        assert share == pytest.approx(0.95, abs=0.01)
        assert g["amount"].iloc[:-1].max() < g["amount"].iloc[-1] / 10
    with pytest.raises(ValueError):
        split_frauds(frauds, np.random.default_rng(0), [2], skew=1.0)


def test_spread_spaces_pieces_over_steps_without_changing_amounts(frauds):
    base = split_frauds(frauds, np.random.default_rng(0), [4], 0.2, True)
    sp = split_frauds(frauds, np.random.default_rng(0), [4], 0.2, True, spread=48)
    assert (sp["amount"].to_numpy() == base["amount"].to_numpy()).all()
    for ev, g in sp.groupby("event_id"):
        f = frauds.iloc[ev]
        assert g["step"].iloc[0] == f["step"] and g["step"].iloc[-1] == f["step"] + 48
        assert g["step"].is_monotonic_increasing
    assert (split_frauds(frauds, np.random.default_rng(0), [4], 0.2, True)["step"].to_numpy()
            == np.repeat(frauds["step"].to_numpy(), 4)).all()


def test_velocity_window_is_blind_to_pieces_spread_beyond_it(frauds):
    X = build_features(split_frauds(frauds, np.random.default_rng(0), [2], 0.2, True, spread=48), "velocity")
    assert (X["orig_prev_count"] == 0).all()                     # two pieces 48 steps apart: outside the 24-step window
    X = build_features(split_frauds(frauds, np.random.default_rng(0), [2], 0.2, True, spread=12), "velocity")
    assert X["orig_prev_count"].max() == 1
