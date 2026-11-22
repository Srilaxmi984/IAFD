"""Experiment B: cost-sensitive learning.

Each model is trained with the fraud class weighted w times more than the legit class (class_weight for
LR/DT/RF, scale_pos_weight for XGBoost/LightGBM). The weight is chosen PER MODEL ON VALIDATION ONLY:
highest validation recall at the headline FPR cap, ties broken by fewer validation false positives, then higher
validation PR-AUC, then the smaller weight.
Test numbers are printed only after the weights are chosen. w=1 is the unweighted baseline (same as run_baseline).

    python scripts/run_costsensitive.py                                  # lr, xgb, lgbm on the 'plus' features
    python scripts/run_costsensitive.py --models lr rf xgb lgbm --weights 1 10 50 200 1000
    python scripts/run_costsensitive.py --dev                            # quick check (NOT reportable)
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
from iafd.evaluate import threshold_at_fpr, evaluate_scores, confusion_at_threshold, wilson_interval


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=["lr", "xgb", "lgbm"], choices=ALL_MODELS)
    ap.add_argument("--weights", nargs="+", type=float, default=[1, 10, 50, 200, 1000])
    ap.add_argument("--dev", action="store_true", help="use the dev sample (NOT reportable)")
    args = ap.parse_args()
    if 1.0 not in args.weights:
        args.weights = [1.0] + args.weights                     # always keep the unweighted baseline

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
    cap, seed = cfg["constraints"]["fpr_cap"], cfg["seed"]
    print(f"features={args.feature_set} | headline FPR cap={cap:.4%} | train neg/pos ratio={(ytr == 0).sum() / ytr.sum():.0f} | "
          f"weights={args.weights}\n\nVALIDATION (used to choose the weight):")

    def evaluate(pv, pt):
        thr = threshold_at_fpr(yva, pv, cap)
        v, t = evaluate_scores(yva, pv, thr), evaluate_scores(yte, pt, thr)
        lo, hi = wilson_interval(t["tp"], t["tp"] + t["fn"])
        out = {"threshold": thr, "val_recall": v["recall"], "val_fp": v["fp"], "val_pr_auc": v["pr_auc"],
               "test_recall": t["recall"], "recall_ci_lo": lo, "recall_ci_hi": hi,
               "test_fp": t["fp"], "test_fpr": t["fpr"], "test_pr_auc": t["pr_auc"],
               "flags_nothing": (t["tp"] + t["fp"]) == 0}
        for sl, m in slices.items():
            c = confusion_at_threshold(yte[m], pt[m], thr)
            out.update({f"{sl}_recall": c["recall"], f"{sl}_fp": c["fp"]})
        return out

    grid, vp, tp_ = [], {}, {}
    for name in args.models:
        for w in args.weights:
            t0 = time.time()
            m = build_model(name, seed, cfg.get("models", {}).get(name), pos_weight=w).fit(Xtr, ytr)
            pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
            vp[(name, w)], tp_[(name, w)] = pv.astype("float32"), pt.astype("float32")
            r = {"model": name, "weight": w, **evaluate(pv, pt)}
            grid.append(r)
            print(f"  {name:5s} w={w:>6g}: val recall@cap={r['val_recall']:.4f}  val fp={r['val_fp']:>4d}  "
                  f"val PR-AUC={r['val_pr_auc']:.4f}  ({time.time() - t0:.0f}s)")
    gdf = pd.DataFrame(grid)

    chosen = {}
    for name in args.models:
        g = gdf[gdf["model"] == name].assign(r=lambda d: d["val_recall"].round(4))
        chosen[name] = float(g.sort_values(["r", "val_fp", "val_pr_auc", "weight"], ascending=[False, True, False, True]).iloc[0]["weight"])
    print("\nChosen weight per model (by validation):", chosen)

    rows = []
    for name in args.models:
        for label, w in (("w=1 (baseline)", 1.0), (f"w={chosen[name]:g} (chosen)", chosen[name])):
            r = gdf[(gdf["model"] == name) & (gdf["weight"] == w)].iloc[0].to_dict()
            rows.append({"model": name, "setting": label, **r})
            if chosen[name] == 1.0:
                break
    if len(args.models) > 1:
        for label, key in (("w=1 (baseline)", lambda n: 1.0), ("chosen weights", lambda n: chosen[n])):
            pv = soft_vote([vp[(n, key(n))].astype("float64") for n in args.models])
            pt = soft_vote([tp_[(n, key(n))].astype("float64") for n in args.models])
            rows.append({"model": "ensemble", "setting": label, "weight": np.nan, **evaluate(pv, pt)})
    res = pd.DataFrame(rows)

    out_dir = resolve(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ("dev_" if args.dev else "") + f"B_{args.feature_set}"
    gdf.to_csv(out_dir / f"experiment_{tag}_grid.csv", index=False)
    res.to_csv(out_dir / f"experiment_{tag}_selected.csv", index=False)

    pd.set_option("display.width", 220)
    show = res.assign(recall_95ci=lambda d: [f"[{a:.4f}, {b:.4f}]" for a, b in zip(d["recall_ci_lo"], d["recall_ci_hi"])])
    print(f"\nTEST at FPR cap {cap:.4%} (threshold chosen on validation): baseline vs chosen weight")
    print(show[["model", "setting", "test_recall", "recall_95ci", "test_fp", "early_recall", "late_recall",
                "early_fp", "late_fp"]].round(4).to_string(index=False))
    if res["flags_nothing"].any():
        print("\nNOTE: rows with flags_nothing=True flagged no transactions (tied scores at this cap).")
    print(f"\nSaved reports/experiment_{tag}_grid.csv and _selected.csv")


if __name__ == "__main__":
    main()
