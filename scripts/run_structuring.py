"""Experiment E: structuring attack and defence.

ATTACK (independent of the defence): each TEST fraud is replaced by k pieces. An event counts as detected when
any piece scores above the clean-validation threshold. Variants: drain / leave 20% of the balance, equal / uneven
pieces, k = 2..32.
DEFENCE: retrain with extra fraud-labelled pieces made from TRAIN frauds, using a DIFFERENT recipe
(equal pieces, k in {2,3,5,8}, residual 0 or 10%). The test attacks use uneven pieces, residual 20%, and k up to 32,
so a defence that only memorised its own recipe does not look better than it is.
STRESS rows add attacks the defence cannot have seen: a tiny first piece with a 95% last piece, and pieces spread
over 12 or 48 steps (the velocity window is 24 steps, so 48 is expected to evade it).
Thresholds always come from CLEAN validation at the headline FPR cap; the cost of the defence shows up as
false positives and recall on clean test.

    python scripts/run_structuring.py                       # XGBoost and LR on the 'plus' features
    python scripts/run_structuring.py --models xgb lgbm
    python scripts/run_structuring.py --dev                 # quick check (NOT reportable)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iafd.config import load_config, resolve
from iafd.data import load_dataset, chronological_split, check_split
from iafd.features import build_features, FEATURE_SETS
from iafd.models import ALL_MODELS, build_model
from iafd.structuring import split_frauds, event_max
from iafd.evaluate import threshold_at_fpr, confusion_at_threshold

KS = (2, 4, 8, 16, 32)
# (name, residual, equal, skew, spread, ks). The last block are STRESS TESTS the defence was never trained on.
VARIANTS = [("drain, equal", 0.0, True, None, 0, KS), ("drain, uneven", 0.0, False, None, 0, KS),
            ("leave 20%, equal", 0.2, True, None, 0, KS), ("leave 20%, uneven", 0.2, False, None, 0, KS),
            ("STRESS leave 20%, skewed 95%", 0.2, True, 0.95, 0, (2, 4, 8)),
            ("STRESS leave 20%, spread 12 steps", 0.2, True, None, 12, (4, 16)),
            ("STRESS leave 20%, spread 48 steps", 0.2, True, None, 48, (4, 16)),
            ("STRESS skewed 95% + spread 48", 0.2, True, 0.95, 48, (2, 4))]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=["xgb", "lr"], choices=ALL_MODELS)
    ap.add_argument("--dev", action="store_true", help="use the dev sample (NOT reportable)")
    args = ap.parse_args()

    path = resolve(cfg["paths"]["dev_parquet" if args.dev else "parquet"])
    if not path.exists():
        sys.exit(f"{path} not found. Run scripts/prepare_data.py first.")
    df = load_dataset(path)
    s = cfg["split"]
    train, val, test = chronological_split(df, s["train_end_step"], s["val_end_step"])
    print("Split:", check_split(train, val, test))
    fs, cap, seed = args.feature_set, cfg["constraints"]["fpr_cap"], cfg["seed"]
    Xtr, Xva, Xte = (build_features(d, fs) for d in (train, val, test))
    ytr, yva, yte = (d["isFraud"].to_numpy() for d in (train, val, test))

    # defence training data: pieces of TRAIN frauds, recipe that differs from the test attacks
    tr_f = train[train["isFraud"] == 1].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    half = rng.random(len(tr_f)) < 0.5
    aug = pd.concat([split_frauds(tr_f[half], rng, [2, 3, 5, 8], residual=0.0, equal=True),
                     split_frauds(tr_f[~half], rng, [2, 3, 5, 8], residual=0.1, equal=True)], ignore_index=True)
    Xaug = build_features(aug, fs)
    Xtr_def = pd.concat([Xtr, Xaug], ignore_index=True)
    ytr_def = np.r_[ytr, np.ones(len(Xaug), dtype=ytr.dtype)]
    print(f"features={fs} | headline cap={cap:.4%} | defence adds {len(Xaug):,} fraud-labelled pieces to {len(Xtr):,} train rows\n")

    te_f = test[test["isFraud"] == 1].reset_index(drop=True)
    all_rows = []
    for name in args.models:
        fitted = {}
        for label, X, y in (("undefended", Xtr, ytr), ("defended", Xtr_def, ytr_def)):
            t0 = time.time()
            m = build_model(name, seed, cfg.get("models", {}).get(name)).fit(X, y)
            pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
            thr = threshold_at_fpr(yva, pv, cap)
            clean = confusion_at_threshold(yte, pt, thr)
            fitted[label] = (m, thr, clean)
            print(f"  {name:5s} {label:10s} clean test: recall={clean['recall']:.4f} fp={clean['fp']:>5d}  ({time.time() - t0:.0f}s)", flush=True)

        rows = [{"model": name, "attack": "none (clean test)", "k": 1,
                 "undefended": fitted["undefended"][2]["recall"], "defended": fitted["defended"][2]["recall"]}]
        for vi, (vname, resid, equal, skew, spread, ks) in enumerate(VARIANTS):
            for k in ks:
                rec = {}
                for label, (m, thr, _) in fitted.items():
                    pieces = split_frauds(te_f, np.random.default_rng(1000 + 10 * vi + k), [k], resid, equal, skew, spread)
                    sc = m.predict_proba(build_features(pieces, fs))[:, 1]
                    rec[label] = float(np.mean(event_max(sc, pieces["event_id"].to_numpy()) > thr))
                rows.append({"model": name, "attack": vname, "k": k, **rec})
        all_rows.extend(rows)

    res = pd.DataFrame(all_rows)
    out_dir = resolve(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ("dev_" if args.dev else "") + f"E_{fs}"
    res.to_csv(out_dir / f"experiment_{tag}.csv", index=False)

    pd.set_option("display.width", 200)
    for name in args.models:
        r = res[res["model"] == name]
        print(f"\n{name.upper()}: share of test fraud EVENTS detected (any piece flagged), threshold from clean validation")
        print(r[["attack", "k", "undefended", "defended"]].round(4).to_string(index=False))
    print(f"\nSaved reports/experiment_{tag}.csv")


if __name__ == "__main__":
    main()
