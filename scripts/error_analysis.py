"""Error analysis on VALIDATION (test stays untouched).

Trains a model (default Random Forest) on the train split, sets the threshold at a tight FPR cap on validation, then prints
  1. validation frauds that were MISSED, and
  2. the legit rows the model scores highest (the 'hard negatives' for experiment F).

    python scripts/error_analysis.py                                  # RF on the plus features
    python scripts/error_analysis.py --model xgb --feature-set plus --cap 0.0001
Paste the output back.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iafd.config import load_config, resolve
from iafd.data import load_dataset, chronological_split
from iafd.features import build_features, FEATURE_SETS
from iafd.models import ALL_MODELS, build_model
from iafd.evaluate import threshold_at_fpr, confusion_at_threshold

SHOW = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "score"]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=0.0001)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    ap.add_argument("--model", default="rf", choices=ALL_MODELS)
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()

    df = load_dataset(resolve(cfg["paths"]["dev_parquet" if args.dev else "parquet"]))
    s = cfg["split"]
    train, val, _ = chronological_split(df, s["train_end_step"], s["val_end_step"])
    Xtr, Xva = build_features(train, args.feature_set), build_features(val, args.feature_set)
    ytr, yva = train["isFraud"].to_numpy(), val["isFraud"].to_numpy()

    model = build_model(args.model, cfg["seed"], cfg.get("models", {}).get(args.model)).fit(Xtr, ytr)
    p = model.predict_proba(Xva)[:, 1]
    thr = threshold_at_fpr(yva, p, args.cap)
    c = confusion_at_threshold(yva, p, thr)
    print(f"{args.model.upper()} ({args.feature_set} features) on validation at FPR cap {args.cap:.4%}: threshold={thr:.4f} recall={c['recall']:.4f} "
          f"missed={c['fn']} false_positives={c['fp']}\n")

    v = val.reset_index(drop=True).assign(score=p)
    pd.set_option("display.width", 200, "display.float_format", "{:,.2f}".format)

    missed = v[(v["isFraud"] == 1) & (v["score"] <= thr)]
    print(f"[1] MISSED frauds ({len(missed)}):")
    print(missed[SHOW].sort_values("score").head(30).to_string(index=False))

    legit = v[v["isFraud"] == 0].sort_values("score", ascending=False)
    print(f"\n[2] Highest-scoring LEGIT rows (top {args.top}):")
    print(legit[SHOW].head(args.top).to_string(index=False))

    fp = legit[legit["score"] > thr]
    caught = v[(v["isFraud"] == 1) & (v["score"] > thr)]
    print(f"\n[3] False positives at this threshold ({len(fp)}) vs caught frauds ({len(caught)}):")
    for name, d in (("false positives", fp), ("caught frauds", caught)):
        if len(d) == 0:
            continue
        print(f"  {name:16s} types={d['type'].value_counts().to_dict()} | median amount={d['amount'].median():,.0f} | "
              f"sender emptied={((d['oldbalanceOrg'] > 0) & (d['newbalanceOrig'] == 0)).mean():.0%} | "
              f"receiver old balance 0={(d['oldbalanceDest'] == 0).mean():.0%} | "
              f"receiver unchanged={(d['oldbalanceDest'] == d['newbalanceDest']).mean():.0%}")


if __name__ == "__main__":
    main()
