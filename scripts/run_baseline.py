"""Experiment A: baseline models, no cost-sensitivity, no velocity.

    python scripts/run_baseline.py --feature-set paper        # A0: the paper's features
    python scripts/run_baseline.py --feature-set engineered   # A : + balance-error features
    python scripts/run_baseline.py --dev                      # quick run on the dev sample

Threshold per model is chosen on VALIDATION (FPR <= cap); test is evaluated once.
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
from iafd.features import build_features
from iafd.models import ALL_MODELS, build_model, soft_vote
from iafd.evaluate import threshold_at_fpr, evaluate_scores, recall_fpr_table, fp_at_recall_table


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="engineered", choices=["paper", "engineered"])
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--dev", action="store_true", help="use the small dev sample (metrics NOT reportable)")
    args = ap.parse_args()

    path = resolve(cfg["paths"]["dev_parquet" if args.dev else "parquet"])
    if not path.exists():
        sys.exit(f"{path} not found. Run scripts/prepare_data.py first.")
    df = load_dataset(path)
    s = cfg["split"]
    train, val, test = chronological_split(df, s["train_end_step"], s["val_end_step"])
    print("Split:", check_split(train, val, test))

    Xtr, Xva, Xte = (build_features(d, args.feature_set) for d in (train, val, test))
    ytr, yva, yte = (d["isFraud"].to_numpy() for d in (train, val, test))
    ref_prev, cap, seed = float(ytr.mean()), cfg["constraints"]["fpr_cap"], cfg["seed"]
    print(f"features={args.feature_set} ({Xtr.shape[1]} cols)  fpr_cap={cap:.3%}  train prevalence={ref_prev:.4%}")

    caps = cfg["constraints"].get("fpr_caps", [cap])
    targets = cfg["constraints"].get("recall_targets", [0.99])
    exp = "A0" if args.feature_set == "paper" else "A"
    val_p, test_p, rows, curve, rcurve = {}, {}, [], [], []

    def record(name, pv, pt, secs):
        thr = threshold_at_fpr(yva, pv, cap)
        v = evaluate_scores(yva, pv, thr, ref_prev)
        t = evaluate_scores(yte, pt, thr, ref_prev)
        rows.append({"experiment": "A0" if args.feature_set == "paper" else "A", "features": args.feature_set,
                     "model": name, "threshold": thr, "fit_seconds": round(secs, 1),
                     "val_recall": v["recall"], "val_fpr": v["fpr"],
                     "test_recall": t["recall"], "test_fpr": t["fpr"], "test_precision": t["precision"],
                     "test_precision_at_train_prev": t["precision_at_ref_prev"],
                     "test_pr_auc": t["pr_auc"], "test_roc_auc": t["roc_auc"],
                     "tp": t["tp"], "fp": t["fp"], "fn": t["fn"], "tn": t["tn"]})
        curve.extend({"experiment": exp, "model": name, **r} for r in recall_fpr_table(yva, pv, yte, pt, caps))
        rcurve.extend({"experiment": exp, "model": name, **r} for r in fp_at_recall_table(yva, pv, yte, pt, targets))
        print(f"  {name:9s} val recall={v['recall']:.3f} (fpr {v['fpr']:.4f}) | "
              f"test recall={t['recall']:.3f} fpr={t['fpr']:.4f} PR-AUC={t['pr_auc']:.3f}")

    for name in args.models:
        t0 = time.time()
        model = build_model(name, seed, cfg.get("models", {}).get(name))
        model.fit(Xtr, ytr)
        pv, pt = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
        val_p[name], test_p[name] = pv, pt
        record(name, pv, pt, time.time() - t0)

    if len(args.models) > 1:
        record("ensemble", soft_vote(list(val_p.values())), soft_vote(list(test_p.values())), 0.0)

    out_dir = resolve(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ("dev_" if args.dev else "") + ("A0_paper" if args.feature_set == "paper" else "A_engineered")
    pd.DataFrame(rows).to_csv(out_dir / f"experiment_{tag}.csv", index=False)
    cdf = pd.DataFrame(curve)
    cdf.to_csv(out_dir / f"experiment_{tag}_curve.csv", index=False)
    print("\nTEST recall at each FPR cap (threshold chosen on validation):")
    print(cdf.pivot(index="model", columns="fpr_cap", values="test_recall").round(4).to_string())
    print("\nTEST false positives at each FPR cap:")
    print(cdf.pivot(index="model", columns="fpr_cap", values="test_fp").to_string())
    rdf = pd.DataFrame(rcurve)
    rdf.to_csv(out_dir / f"experiment_{tag}_recall_targets.csv", index=False)
    print("\nTEST false positives needed to reach each recall target (threshold chosen on validation):")
    print(rdf.pivot(index="model", columns="recall_target", values="test_fp").to_string())
    print("\nTEST recall actually achieved at those thresholds:")
    print(rdf.pivot(index="model", columns="recall_target", values="test_recall").round(4).to_string())
    print(f"\nSaved {out_dir / f'experiment_{tag}.csv'} and the _curve / _recall_targets files")


if __name__ == "__main__":
    main()
