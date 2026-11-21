"""Data loading and chronological splitting for PaySim.

Design notes
------------
* The raw CSV (~6.36M rows) is converted ONCE to a compact parquet file, in chunks,
  so it never has to fit in memory as strings.
* nameOrig / nameDest are hashed to uint64 ids (needed later for velocity features).
* The split is chronological by `step`, not random: a random split lets the model
  see "future" transactions of the same fraud pattern during training.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
MONEY_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
RAW_REQUIRED = ["step", "type", "nameOrig", "nameDest", *MONEY_COLS, "isFraud"]
COMPACT_COLS = ["step", "type", "orig_id", "dest_id", *MONEY_COLS, "isFraud"]


def _hash_ids(s: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(s.astype(str), index=False).to_numpy(dtype="uint64")


def compact_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw PaySim rows -> compact schema (ids hashed, small dtypes, name/flag columns dropped)."""
    missing = [c for c in RAW_REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    bad = set(raw["type"].unique()) - set(TYPES)
    if bad:
        raise ValueError(f"Unknown transaction types: {sorted(bad)}")
    out = pd.DataFrame({
        "step": raw["step"].astype("int16").to_numpy(),
        "type": raw["type"].astype(str).to_numpy(),      # kept as str in parquet; category on load
        "orig_id": _hash_ids(raw["nameOrig"]),
        "dest_id": _hash_ids(raw["nameDest"]),
        **{c: raw[c].astype("float64").to_numpy() for c in MONEY_COLS},  # float64: balance-error features need precision
        "isFraud": raw["isFraud"].astype("int8").to_numpy(),
    })
    return out[COMPACT_COLS]


def csv_to_parquet(csv_path: str | Path, parquet_path: str | Path, chunksize: int = 1_000_000) -> int:
    """Stream the raw CSV into a compact parquet file. Returns the number of rows written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=lambda c: c in RAW_REQUIRED):
            table = pa.Table.from_pandas(compact_frame(chunk), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema)
            writer.write_table(table)
            n += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    if n == 0:
        raise ValueError(f"No rows read from {csv_path}")
    return n


def load_dataset(parquet_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["type"] = pd.Categorical(df["type"], categories=TYPES)
    return df


def chronological_split(df: pd.DataFrame, train_end_step: int, val_end_step: int):
    """train: step <= train_end, val: (train_end, val_end], test: step > val_end."""
    if not train_end_step < val_end_step:
        raise ValueError("train_end_step must be < val_end_step")
    train = df[df["step"] <= train_end_step]
    val = df[(df["step"] > train_end_step) & (df["step"] <= val_end_step)]
    test = df[df["step"] > val_end_step]
    return train, val, test


def check_split(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fail loudly if the split is empty, overlapping in time, or has a fraud-free part."""
    parts = {"train": train, "val": val, "test": test}
    for name, p in parts.items():
        if len(p) == 0:
            raise ValueError(f"{name} split is empty")
        if p["isFraud"].sum() == 0:
            raise ValueError(f"{name} split contains no fraud rows")
    if not (train["step"].max() < val["step"].min() and val["step"].max() < test["step"].min()):
        raise ValueError("Splits overlap in time (leakage)")
    return {
        name: {"rows": int(len(p)), "fraud": int(p["isFraud"].sum()),
               "fraud_rate": float(p["isFraud"].mean()),
               "steps": (int(p["step"].min()), int(p["step"].max()))}
        for name, p in parts.items()
    }


def make_dev_sample(df: pd.DataFrame, legit_fraction: float, seed: int = 42) -> pd.DataFrame:
    """Keep ALL fraud rows and a fraction of legit rows. For fast development ONLY:
    it inflates the fraud rate, so never report final metrics from it."""
    if not 0 < legit_fraction <= 1:
        raise ValueError("legit_fraction must be in (0, 1]")
    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0].sample(frac=legit_fraction, random_state=seed)
    return pd.concat([fraud, legit]).sort_values("step", kind="stable").reset_index(drop=True)
