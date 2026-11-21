import pandas as pd
import pytest
from iafd.data import (compact_frame, csv_to_parquet, load_dataset, chronological_split,
                       check_split, make_dev_sample, COMPACT_COLS, TYPES)
from iafd.synthetic import make_synthetic_paysim


@pytest.fixture(scope="module")
def raw():
    return make_synthetic_paysim(n_rows=40_000, seed=1)


def test_compact_schema_and_dtypes(raw):
    c = compact_frame(raw)
    assert list(c.columns) == COMPACT_COLS
    assert str(c["step"].dtype) == "int16" and str(c["amount"].dtype) == "float64"
    assert "nameOrig" not in c.columns


def test_compact_rejects_unknown_type_and_missing_cols(raw):
    with pytest.raises(ValueError):
        compact_frame(raw.assign(type="WIRE"))
    with pytest.raises(ValueError):
        compact_frame(raw.drop(columns=["amount"]))


def test_csv_roundtrip_chunked(raw, tmp_path):
    csv, pq = tmp_path / "raw.csv", tmp_path / "out.parquet"
    raw.to_csv(csv, index=False)
    n = csv_to_parquet(csv, pq, chunksize=7_000)          # forces several chunks
    df = load_dataset(pq)
    assert n == len(raw) == len(df)
    assert df["isFraud"].sum() == raw["isFraud"].sum()
    assert set(df["type"].cat.categories) == set(TYPES)


def test_split_is_chronological_and_complete(raw):
    df = compact_frame(raw)
    tr, va, te = chronological_split(df, 240, 355)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr["step"].max() <= 240 < va["step"].min() and va["step"].max() <= 355 < te["step"].min()
    info = check_split(tr, va, te)
    assert all(v["fraud"] > 0 for v in info.values())


def test_check_split_detects_problems(raw):
    df = compact_frame(raw)
    tr, va, te = chronological_split(df, 240, 355)
    with pytest.raises(ValueError):
        check_split(tr, va.iloc[0:0], te)                  # empty part
    with pytest.raises(ValueError):
        check_split(tr, va[va["isFraud"] == 0], te)        # no fraud in part
    with pytest.raises(ValueError):
        check_split(pd.concat([tr, va]), va, te)           # time overlap
    with pytest.raises(ValueError):
        chronological_split(df, 355, 240)


def test_dev_sample_keeps_all_fraud(raw):
    df = compact_frame(raw)
    dev = make_dev_sample(df, 0.1, seed=0)
    assert dev["isFraud"].sum() == df["isFraud"].sum()
    assert len(dev) < len(df) and dev["step"].is_monotonic_increasing
    with pytest.raises(ValueError):
        make_dev_sample(df, 0.0)
