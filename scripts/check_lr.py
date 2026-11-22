"""Is logistic regression stopping too early? (validation only; test untouched)

Every earlier LR fit stopped after ~12 iterations even with max_iter=5000, so the stop is by tolerance,
not by the iteration cap. This refits w=1 and w=10 with the default tolerance and with a tight one.

    python scripts/check_lr.py
This runs 4 fits and can take a few minutes. Let it finish, then paste the table.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.metrics import average_precision_score
from iafd.config import load_config, resolve
from iafd.data import load_dataset, chronological_split
from iafd.features import build_features
from iafd.models import build_model
from iafd.evaluate import threshold_at_fpr, confusion_at_threshold


def main():
    cfg = load_config()
    df = load_dataset(resolve(cfg["paths"]["parquet"]))
    s = cfg["split"]
    train, val, _ = chronological_split(df, s["train_end_step"], s["val_end_step"])
    Xtr, Xva = build_features(train, "plus"), build_features(val, "plus")
    ytr, yva = train["isFraud"].to_numpy(), val["isFraud"].to_numpy()
    cap = cfg["constraints"]["fpr_cap"]
    print(f"cap={cap:.4%}  (validation only)\n")
    print(f"{'w':>4s} {'tol':>7s} {'iters':>6s} {'warn':>4s} {'max|coef|':>9s} {'PR-AUC':>7s} {'recall@cap':>10s} {'sec':>5s}", flush=True)
    for w in (1, 10):
        for tol in (1e-4, 1e-8):
            t0 = time.time()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m = build_model("lr", cfg["seed"], {"max_iter": 5000, "tol": tol}, pos_weight=w).fit(Xtr, ytr)
            lr = m[-1]
            p = m.predict_proba(Xva)[:, 1]
            rec = confusion_at_threshold(yva, p, threshold_at_fpr(yva, p, cap))["recall"]
            print(f"{w:>4d} {tol:>7.0e} {int(lr.n_iter_[0]):>6d} {len(caught):>4d} {np.abs(lr.coef_).max():>9.1f} "
                  f"{average_precision_score(yva, p):>7.4f} {rec:>10.4f} {time.time() - t0:>5.0f}", flush=True)


if __name__ == "__main__":
    main()
