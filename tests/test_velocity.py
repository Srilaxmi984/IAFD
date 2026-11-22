import numpy as np
import pandas as pd
import pytest
from iafd.velocity import sender_history
from iafd.features import build_features, FEATURE_SETS
from iafd.structuring import split_frauds
from iafd.data import compact_frame
from iafd.synthetic import make_synthetic_paysim


def test_sender_history_counts_only_earlier_rows_in_window():
    oid = np.array([1, 1, 2, 1])
    step = np.array([10, 10, 10, 40])
    amt = np.array([5.0, 7.0, 3.0, 1.0])
    cnt, out = sender_history(oid, step, amt, window=24)
    assert cnt.tolist() == [0, 1, 0, 0]          # row 4 is 30 steps after the others: outside the 24-step window
    assert out.tolist() == [0.0, 5.0, 0.0, 0.0]


def test_sender_history_is_input_order_independent_and_never_looks_ahead():
    rng = np.random.default_rng(0)
    n = 500
    oid, step, amt = rng.integers(0, 40, n), np.sort(rng.integers(1, 100, n)), rng.random(n)
    cnt, out = sender_history(oid, step, amt, 24)
    for i in rng.choice(n, 25, replace=False):   # brute force
        prior = [j for j in range(i) if oid[j] == oid[i] and step[j] >= step[i] - 24]
        assert cnt[i] == len(prior) and out[i] == pytest.approx(sum(amt[j] for j in prior))


def test_clean_synthetic_data_has_almost_no_velocity():
    df = compact_frame(make_synthetic_paysim(20_000, seed=4))
    X = build_features(df, "velocity")
    assert list(X.columns) == FEATURE_SETS["velocity"]
    assert (X["orig_prev_count"] > 0).mean() < 0.01


def test_structuring_pieces_show_rising_velocity():
    df = compact_frame(make_synthetic_paysim(60_000, fraud_rate=0.01, seed=7))
    fr = df[df["isFraud"] == 1].reset_index(drop=True)
    p = split_frauds(fr, np.random.default_rng(0), [4], residual=0.2, equal=True)
    X = build_features(p, "velocity")
    for ev in range(min(20, len(fr))):
        g = X[p["event_id"].to_numpy() == ev]
        assert g["orig_prev_count"].tolist() == [0, 1, 2, 3]
        assert g["orig_prev_ratio"].is_monotonic_increasing and g["orig_prev_ratio"].iloc[-1] > 0.5
