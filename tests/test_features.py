import numpy as np
import pytest
from iafd.data import compact_frame, MONEY_COLS, TYPES
from iafd.features import build_features, FEATURE_SETS
from iafd.synthetic import make_synthetic_paysim


@pytest.fixture(scope="module")
def df():
    return compact_frame(make_synthetic_paysim(5_000, seed=2))


def test_paper_set_is_raw_money_cols(df):
    X = build_features(df, "paper")
    assert list(X.columns) == MONEY_COLS and len(X) == len(df)


def test_engineered_columns_order_and_no_nan(df):
    X = build_features(df, "engineered")
    assert list(X.columns) == FEATURE_SETS["engineered"]
    assert not X.isna().any().any() and np.isfinite(X.to_numpy()).all()


def test_balance_error_formulas():
    import pandas as pd
    row = pd.DataFrame({"step": [1], "type": ["TRANSFER"], "orig_id": [1], "dest_id": [2],
                        "amount": [100.0], "oldbalanceOrg": [100.0], "newbalanceOrig": [0.0],
                        "oldbalanceDest": [0.0], "newbalanceDest": [0.0], "isFraud": [1]})
    X = build_features(row, "engineered")
    assert X.loc[0, "error_orig"] == 0.0          # 0 + 100 - 100
    assert X.loc[0, "error_dest"] == 100.0        # 0 + 100 - 0  (receiver balance never grew)
    assert X.loc[0, "orig_emptied"] == 1.0 and X.loc[0, "dest_zero_after"] == 1.0
    assert X.loc[0, "type_TRANSFER"] == 1.0 and X.loc[0, "type_CASH_OUT"] == 0.0


def test_one_hot_rows_sum_to_one(df):
    X = build_features(df, "engineered")
    assert (X[[f"type_{t}" for t in TYPES]].sum(axis=1) == 1).all()


def test_unknown_feature_set(df):
    with pytest.raises(ValueError):
        build_features(df, "nope")


def test_no_dest_has_no_receiver_features(df):
    X = build_features(df, "no_dest")
    assert list(X.columns) == FEATURE_SETS["no_dest"]
    banned = {"oldbalanceDest", "newbalanceDest", "error_dest", "dest_zero_before", "dest_zero_after"}
    assert banned.isdisjoint(X.columns)
    assert not X.isna().any().any()
