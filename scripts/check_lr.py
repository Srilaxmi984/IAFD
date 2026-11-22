"""Why does logistic regression collapse at one class weight (validation recall 0.105 at w=10)?
Validation only; test is untouched.

    python scripts/check_lr.py
Paste the printed table back.
"""
import sys
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
    print(f"{'w':>5s} {'max_iter':>8s} {'iters':>6s} {'warn':>4s} {'max|coef|':>9s} {'scores==1':>9s} "
          f"{'PR-AUC':>7s} {'recall@cap (prob)':>18s} {'recall@cap (logit)':>19s}")
    for w in (1, 5, 10, 20, 50):
        for mi in (1000, 5000):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m = build_model("lr", cfg["seed"], {"max_iter": mi}, pos_weight=w).fit(Xtr, ytr)
            lr = m[-1]
            p, z = m.predict_proba(Xva)[:, 1], m.decision_function(Xva)
            r_p = confusion_at_threshold(yva, p, threshold_at_fpr(yva, p, cap))["recall"]
            r_z = confusion_at_threshold(yva, z, threshold_at_fpr(yva, z, cap))["recall"]
            print(f"{w:>5d} {mi:>8d} {int(lr.n_iter_[0]):>6d} {len(caught):>4d} {np.abs(lr.coef_).max():>9.1f} "
                  f"{np.mean(p >= 1 - 1e-12):>9.4f} {average_precision_score(yva, p):>7.4f} {r_p:>18.4f} {r_z:>19.4f}")


if __name__ == "__main__":
    main()
