"""Quick facts about the prepared data. Run once; paste the output when asked.

    python scripts/diagnose_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iafd.config import load_config, resolve
from iafd.data import load_dataset


def main():
    cfg = load_config()
    df = load_dataset(resolve(cfg["paths"]["parquet"]))
    fr = df[df["isFraud"] == 1]
    print(f"rows={len(df):,} fraud={len(fr):,} ({len(fr)/len(df):.4%})")

    print("\n[1] Sender activity (matters for velocity features)")
    counts = df["orig_id"].map(df["orig_id"].value_counts())
    print(f"  unique senders: {df['orig_id'].nunique():,} of {len(df):,} rows")
    print(f"  rows whose sender appears more than once: {(counts > 1).mean():.4%}")
    print(f"  fraud rows whose sender appears more than once: {(counts[df['isFraud'] == 1] > 1).mean():.4%}")

    print("\n[2] Fraud by transaction type")
    print(fr["type"].value_counts().to_string())

    print("\n[3] Simple fraud signatures")
    print(f"  fraud with newbalanceOrig == 0: {(fr['newbalanceOrig'] == 0).mean():.2%}")
    print(f"  fraud with oldbalanceDest == 0: {(fr['oldbalanceDest'] == 0).mean():.2%}")
    legit_te = df[(df['isFraud'] == 0) & df['type'].isin(['TRANSFER', 'CASH_OUT'])]
    print(f"  legit TRANSFER/CASH_OUT with newbalanceOrig == 0: {(legit_te['newbalanceOrig'] == 0).mean():.2%}")

    print("\n[4] Why fraud rate rises late: rows per step bucket")
    b = pd.cut(df["step"], [0, 120, 240, 355, 500, 743])
    print(df.groupby(b, observed=True)["isFraud"].agg(rows="size", fraud="sum").to_string())


if __name__ == "__main__":
    main()
