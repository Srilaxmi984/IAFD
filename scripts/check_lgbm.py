"""Why does LightGBM collapse (PR-AUC ~ base rate)? Try a few configurations on TRAIN, score on VALIDATION only.

    python scripts/check_lgbm.py
Paste the printed table back.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier
from iafd.config import load_config, resolve
from iafd.data import load_dataset, chronological_split
from iafd.features import build_features

CONFIGS = {
    "default (as in baseline)": dict(n_estimators=300, num_leaves=63, learning_rate=0.1),
    "min_child_samples=5": dict(n_estimators=300, num_leaves=63, learning_rate=0.1, min_child_samples=5),
    "+ min_child_weight=1e-5": dict(n_estimators=300, num_leaves=63, learning_rate=0.1,
                                    min_child_samples=5, min_child_weight=1e-5),
    "smaller leaves, lr 0.05": dict(n_estimators=500, num_leaves=31, learning_rate=0.05,
                                    min_child_samples=5, min_child_weight=1e-5),
}


def main():
    cfg = load_config()
    df = load_dataset(resolve(cfg["paths"]["parquet"]))
    s = cfg["split"]
    train, val, _ = chronological_split(df, s["train_end_step"], s["val_end_step"])
    Xtr, Xva = build_features(train), build_features(val)
    ytr, yva = train["isFraud"].to_numpy(), val["isFraud"].to_numpy()
    print(f"train {Xtr.shape}, fraud {ytr.sum()} | val fraud {yva.sum()}  (validation only; test untouched)\n")
    print(f"{'config':28s} {'val PR-AUC':>10s} {'splits made':>12s} {'sec':>6s}")
    for name, params in CONFIGS.items():
        t0 = time.time()
        m = LGBMClassifier(random_state=cfg["seed"], n_jobs=-1, verbose=-1, **params).fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        n_splits = int(m.booster_.feature_importance("split").sum())
        print(f"{name:28s} {average_precision_score(yva, p):>10.4f} {n_splits:>12d} {time.time() - t0:>6.0f}")


if __name__ == "__main__":
    main()
