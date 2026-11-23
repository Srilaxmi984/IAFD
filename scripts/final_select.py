"""Choose the two FINAL models using VALIDATION data only (test is never touched here).

  FINAL SINGLE MODEL: every candidate model is trained at the natural class imbalance and at more extreme ones
  (only a fraction of the TRAIN frauds kept), with and without a class weight. The winner has the best mean LFR
  (mean recall at low FPR caps, on validation) across the imbalance levels, i.e. it degrades least as imbalance grows.
  FINAL ENSEMBLE: every combination of 2+ candidate models and two ways of averaging is scored on validation;
  near-ties (differences below 0.001) go to the simpler ensemble.

    python scripts/final_select.py
    python scripts/final_select.py --levels 1.0 0.1 --models lr xgb lgbm      # faster
    python scripts/final_select.py --dev                                       # quick check (NOT reportable)
Writes reports/final_selection.json plus reports/final_selection_single.csv and _ensemble.csv.
"""
import argparse
import json
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
from iafd.evaluate import threshold_at_fpr, confusion_at_threshold
from iafd.final import (LOW_FPR_CAPS, lfr_score, fraud_subsample_mask, strategy_pos_weight, combine, all_subsets)

AGG_RANK = {"mean": 0, "logit_mean": 1}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=["lr", "rf", "xgb", "lgbm"], choices=ALL_MODELS)
    ap.add_argument("--levels", nargs="+", type=float, default=[1.0, 0.1, 0.03],
                    help="fraction of TRAIN frauds kept (1.0 = natural imbalance)")
    ap.add_argument("--strategies", nargs="+", default=["plain", "sqrtw"], choices=["plain", "sqrtw"])
    ap.add_argument("--dev", action="store_true", help="use the dev sample (NOT reportable)")
    args = ap.parse_args()
    levels = sorted(set([1.0] + args.levels), reverse=True)
    strategies = sorted(set(["plain"] + args.strategies), key=lambda s: ["plain", "sqrtw"].index(s))

    path = resolve(cfg["paths"]["dev_parquet" if args.dev else "parquet"])
    if not path.exists():
        sys.exit(f"{path} not found. Run scripts/prepare_data.py first.")
    df = load_dataset(path)
    s = cfg["split"]
    train, val, test = chronological_split(df, s["train_end_step"], s["val_end_step"])
    print("Split:", check_split(train, val, test))
    seed = cfg["seed"]
    Xtr, Xva = build_features(train, args.feature_set), build_features(val, args.feature_set)
    ytr, yva = train["isFraud"].to_numpy(), val["isFraud"].to_numpy()
    print(f"features={args.feature_set} | models={args.models} | imbalance levels={levels} | strategies={strategies}\n"
          f"LFR caps={LOW_FPR_CAPS} (validation only)\n", flush=True)

    rows, natural = [], {}
    for level in levels:
        keep = fraud_subsample_mask(ytr, level, seed)
        Xk, yk = Xtr[keep], ytr[keep]
        for strategy in strategies:
            w = strategy_pos_weight(yk, strategy)
            for name in args.models:
                t0 = time.time()
                m = build_model(name, seed, cfg.get("models", {}).get(name), pos_weight=w).fit(Xk, yk)
                pv = m.predict_proba(Xva)[:, 1]
                c = confusion_at_threshold(yva, pv, threshold_at_fpr(yva, pv, 1e-4))
                r = {"model": name, "strategy": strategy, "level": level, "train_frauds": int(yk.sum()),
                     "weight": round(w, 2), "val_lfr": lfr_score(yva, pv), "val_recall_at_0.01%": c["recall"],
                     "val_fp_at_0.01%": c["fp"]}
                rows.append(r)
                if level == 1.0 and strategy == "plain":
                    natural[name] = pv.astype("float64")
                print(f"  level={level:<5g} {strategy:5s} {name:5s} train frauds={r['train_frauds']:>5d}  "
                      f"val LFR={r['val_lfr']:.4f}  recall@0.01%={c['recall']:.4f} fp={c['fp']:>4d}  ({time.time() - t0:.0f}s)", flush=True)
    sdf = pd.DataFrame(rows)

    agg = (sdf.groupby(["model", "strategy"]).agg(mean_lfr=("val_lfr", "mean"), min_lfr=("val_lfr", "min")).reset_index())
    agg["k"] = agg["mean_lfr"].round(3)
    agg["srank"] = agg["strategy"].map({"plain": 0, "sqrtw": 1})
    agg = agg.sort_values(["k", "min_lfr", "srank"], ascending=[False, False, True]).drop(columns=["k", "srank"])
    best_single = agg.iloc[0]
    print("\nSINGLE MODEL ranking (mean / worst validation LFR across imbalance levels):")
    pd.set_option("display.width", 200, "display.float_format", "{:.4f}".format)
    print(agg.to_string(index=False))
    print(f"\n=> FINAL SINGLE MODEL: {best_single['model']} with strategy '{best_single['strategy']}'")

    erows = []
    for members in all_subsets(list(natural)):
        for mode in ("mean", "logit_mean"):
            p = combine([natural[m] for m in members], mode)
            erows.append({"members": "+".join(members), "n": len(members), "aggregator": mode, "val_lfr": lfr_score(yva, p)})
    edf = pd.DataFrame(erows)
    edf["k"] = edf["val_lfr"].round(3)
    edf["arank"] = edf["aggregator"].map(AGG_RANK)
    edf = edf.sort_values(["k", "n", "arank"], ascending=[False, True, True]).drop(columns=["k", "arank"])
    best_ens = edf.iloc[0]
    print("\nENSEMBLE ranking (validation LFR), top 8:")
    print(edf.head(8).to_string(index=False))
    print("\nBest single member at natural imbalance (plain), for reference:")
    print({m: round(lfr_score(yva, p), 4) for m, p in natural.items()})
    print(f"\n=> FINAL ENSEMBLE: {best_ens['members']} averaged by '{best_ens['aggregator']}'")

    out_dir = resolve(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "dev_" if args.dev else ""
    sdf.to_csv(out_dir / f"{tag}final_selection_single.csv", index=False)
    edf.to_csv(out_dir / f"{tag}final_selection_ensemble.csv", index=False)
    sel = {"feature_set": args.feature_set, "selection_seed": seed, "caps": list(LOW_FPR_CAPS), "levels": levels,
           "single": {"model": str(best_single["model"]), "strategy": str(best_single["strategy"]),
                      "mean_lfr": float(best_single["mean_lfr"]), "min_lfr": float(best_single["min_lfr"])},
           "ensemble": {"members": str(best_ens["members"]).split("+"), "aggregator": str(best_ens["aggregator"]),
                        "val_lfr": float(best_ens["val_lfr"])}}
    (out_dir / f"{tag}final_selection.json").write_text(json.dumps(sel, indent=2))
    print(f"\nSaved reports/{tag}final_selection.json (+ _single.csv, _ensemble.csv)")


if __name__ == "__main__":
    main()
