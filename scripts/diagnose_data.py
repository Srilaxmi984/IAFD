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
    cashy = df["type"].isin(["TRANSFER", "CASH_OUT"])
    legit_c = df[(df["isFraud"] == 0) & cashy]
    print(f"rows={len(df):,} fraud={len(fr):,} ({len(fr)/len(df):.4%})")

    print("\n[1] Sender activity")
    sc = df["orig_id"].map(df["orig_id"].value_counts())
    print(f"  rows whose sender appears more than once: {(sc > 1).mean():.4%}")
    print(f"  fraud rows whose sender appears more than once: {(sc[df['isFraud'] == 1] > 1).mean():.4%}")

    print("\n[2] Destination activity (TRANSFER/CASH_OUT rows only)")
    dc = df["dest_id"].map(df["dest_id"].value_counts())
    fd, ld = dc[(df["isFraud"] == 1)], dc[(df["isFraud"] == 0) & cashy]
    for name, x in (("fraud", fd), ("legit", ld)):
        print(f"  {name}: dest appears more than once {(x > 1).mean():.2%} | median appearances {x.median():.0f} | 95th pct {x.quantile(.95):.0f}")
    print("  (counts here use the whole dataset to test for signal; real features would use past rows only)")

    print("\n[3] Fraud by transaction type")
    print(fr["type"].value_counts().to_string())

    print("\n[4] Balance signatures, TRANSFER/CASH_OUT rows")
    def emptied(d): return ((d["oldbalanceOrg"] > 0) & (d["newbalanceOrig"] == 0)).mean()
    def zero_start(d): return (d["oldbalanceOrg"] == 0).mean()
    print(f"  sender had a balance and was emptied: fraud {emptied(fr):.2%} | legit {emptied(legit_c):.2%}")
    print(f"  sender balance already 0 before:      fraud {zero_start(fr):.2%} | legit {zero_start(legit_c):.2%}")
    print(f"  receiver balance 0 before:            fraud {(fr['oldbalanceDest'] == 0).mean():.2%} | legit {(legit_c['oldbalanceDest'] == 0).mean():.2%}")
    print(f"  receiver balance unchanged after:     fraud {(fr['oldbalanceDest'] == fr['newbalanceDest']).mean():.2%} | legit {(legit_c['oldbalanceDest'] == legit_c['newbalanceDest']).mean():.2%}")

    print("\n[5] Rows and fraud per step bucket")
    b = pd.cut(df["step"], [0, 120, 240, 355, 500, 743])
    g = df.groupby(b, observed=True)["isFraud"].agg(rows="size", fraud="sum")
    g["fraud_rate"] = (g["fraud"] / g["rows"]).map("{:.3%}".format)
    print(g.to_string())


if __name__ == "__main__":
    main()
