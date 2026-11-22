"""Baseline experiments.

    python scripts/run_baseline.py --feature-set paper         # A0 : the paper's 5 raw features
    python scripts/run_baseline.py --feature-set engineered    # A  : + balance-consistency features
    python scripts/run_baseline.py --feature-set plus          # A+ : + full-balance-transfer features
    python scripts/run_baseline.py --feature-set no_dest       # ablation: no receiver-balance features
    python scripts/run_baseline.py --seeds 3                   # repeat with 3 random seeds (mean and spread)
    python scripts/run_baseline.py --dev                       # quick run on the dev sample (NOT reportable)

Thresholds are chosen on VALIDATION; test is evaluated once at those thresholds.
Reports per model: recall / FPR / false positives at several FPR caps and recall targets, plus
95% confidence intervals and results for the early (<= test_slice_step) and late test regimes.
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
from iafd.models import ALL_MODELS, build_model, soft_vote
from iafd.evaluate import (threshold_at_fpr, evaluate_scores, confusion_at_threshold,
                           recall_fpr_table, fp_at_recall_table, wilson_interval)

EXP = {"paper": "A0", "engineered": "A", "no_dest": "A_nodest", "plus": "A_plus"}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="engineered", choices=list(FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--seeds", type=int, default=1, help="number of random seeds to repeat training with")
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
    te_step = test["step"].to_numpy()
    cut = s.get("test_slice_step", 500)
    slices = {"early": te_step <= cut, "late": te_step > cut}

    ref_prev, cap = float(ytr.mean()), cfg["constraints"]["fpr_cap"]
    caps = cfg["constraints"].get("fpr_caps", [cap])
    targets = cfg["constraints"].get("recall_targets", [0.99])
    exp = EXP[args.feature_set]
    seeds = [cfg["seed"] + i for i in range(max(1, args.seeds))]
    print(f"features={args.feature_set} ({Xtr.shape[1]} cols) | headline FPR cap={cap:.4%} | seeds={seeds} | "
          f"train prevalence={ref_prev:.4%} | test slices: steps<={cut} / >{cut}")

    rows, curve, rcurve, primary = [], [], [], []

    def record(seed, name, pv, pt, secs):
        thr = threshold_at_fpr(yva, pv, cap)
        v, t = evaluate_scores(yva, pv, thr, ref_prev), evaluate_scores(yte, pt, thr, ref_prev)
        rows.append({"experiment": exp, "features": args.feature_set, "model": name, "seed": seed,
                     "threshold": thr, "fit_seconds": round(secs, 1),
                     "val_recall": v["recall"], "val_fpr": v["fpr"],
                     "test_recall": t["recall"], "test_fpr": t["fpr"], "test_precision": t["precision"],
                     "test_precision_at_train_prev": t["precision_at_ref_prev"],
                     "test_pr_auc": t["pr_auc"], "test_roc_auc": t["roc_auc"],
                     "tp": t["tp"], "fp": t["fp"], "fn": t["fn"], "tn": t["tn"]})
        curve.extend({"experiment": exp, "model": name, "seed": seed, **r} for r in recall_fpr_table(yva, pv, yte, pt, caps))
        rcurve.extend({"experiment": exp, "model": name, "seed": seed, **r} for r in fp_at_recall_table(yva, pv, yte, pt, targets))
        r_lo, r_hi = wilson_interval(t["tp"], t["tp"] + t["fn"])
        f_lo, f_hi = wilson_interval(t["fp"], t["fp"] + t["tn"])
        row = {"experiment": exp, "model": name, "seed": seed, "cap": cap, "threshold": thr,
               "test_recall": t["recall"], "recall_ci_lo": r_lo, "recall_ci_hi": r_hi,
               "test_fp": t["fp"], "test_fpr": t["fpr"], "fpr_ci_lo": f_lo, "fpr_ci_hi": f_hi,
               "flags_nothing": (t["tp"] + t["fp"]) == 0}
        for sl, m in slices.items():
            c = confusion_at_threshold(yte[m], pt[m], thr)
            row.update({f"{sl}_recall": c["recall"], f"{sl}_fp": c["fp"], f"{sl}_fraud": c["tp"] + c["fn"]})
        primary.append(row)
        note = "  <-- flags nothing (tied scores)" if row["flags_nothing"] else ""
        print(f"  seed {seed} {name:9s} val recall={v['recall']:.4f} (fpr {v['fpr']:.5f}) | "
              f"test recall={t['recall']:.4f} fp={t['fp']:>5d} fpr={t['fpr']:.5f} PR-AUC={t['pr_auc']:.3f}{note}")

    for seed in seeds:
        val_p, test_p = {}, {}
        for name in args.models:
            t0 = time.time()
            model = build_model(name, seed, cfg.get("models", {}).get(name))
            model.fit(Xtr, ytr)
            pv, pt = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
            val_p[name], test_p[name] = pv, pt
            record(seed, name, pv, pt, time.time() - t0)
        if len(args.models) > 1:
            record(seed, "ensemble", soft_vote(list(val_p.values())), soft_vote(list(test_p.values())), 0.0)

    out_dir = resolve(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ("dev_" if args.dev else "") + f"{exp}_{args.feature_set}"
    rdf, cdf, tdf, pdf = (pd.DataFrame(x) for x in (rows, curve, rcurve, primary))
    rdf.to_csv(out_dir / f"experiment_{tag}.csv", index=False)
    cdf.to_csv(out_dir / f"experiment_{tag}_curve.csv", index=False)
    tdf.to_csv(out_dir / f"experiment_{tag}_recall_targets.csv", index=False)
    pdf.to_csv(out_dir / f"experiment_{tag}_primary.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nTEST recall at each FPR cap (mean over {len(seeds)} seed(s); thresholds chosen on validation):")
    print(cdf.pivot_table(index="model", columns="fpr_cap", values="test_recall", aggfunc="mean").round(4).to_string())
    print("\nTEST false positives at each FPR cap:")
    print(cdf.pivot_table(index="model", columns="fpr_cap", values="test_fp", aggfunc="mean").round(1).to_string())
    print("\nTEST false positives needed to reach each recall target:")
    print(tdf.pivot_table(index="model", columns="recall_target", values="test_fp", aggfunc="mean").round(1).to_string())
    print("\nTEST recall actually achieved at those thresholds:")
    print(tdf.pivot_table(index="model", columns="recall_target", values="test_recall", aggfunc="mean").round(4).to_string())

    first = pdf[pdf["seed"] == seeds[0]].set_index("model")
    agg = pdf.groupby("model", sort=False).agg(recall=("test_recall", "mean"), fp=("test_fp", "mean"),
                                               early_recall=("early_recall", "mean"), late_recall=("late_recall", "mean"),
                                               early_fp=("early_fp", "mean"), late_fp=("late_fp", "mean"))
    if len(seeds) > 1:
        sd = pdf.groupby("model", sort=False).agg(recall_sd=("test_recall", "std"), fp_sd=("test_fp", "std"))
        agg = agg.join(sd)
    agg["recall_95ci_seed0"] = [f"[{first.loc[m, 'recall_ci_lo']:.4f}, {first.loc[m, 'recall_ci_hi']:.4f}]" for m in agg.index]
    print(f"\nHEADLINE (FPR cap {cap:.4%}): test recall with 95% CI (first seed, fixed threshold), "
          f"false positives, and the early / late test regimes:")
    print(agg.round(4).to_string())
    n_fraud = {k: int(first.iloc[0][f"{k}_fraud"]) for k in slices}
    print(f"(frauds in test slices: {n_fraud})")
    print(f"\nSaved reports/experiment_{tag}*.csv (main, _curve, _recall_targets, _primary)")


if __name__ == "__main__":
    main()
