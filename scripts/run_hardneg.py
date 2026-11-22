"""Experiment F: hard-negative tuning.

1. Score the TRAIN rows out-of-fold (4 contiguous step blocks) with the base model.
2. Hard negatives = legit train rows above the --mine-fpr threshold of those out-of-fold scores.
3. Retrain with each hard negative weighted w times (w=1 is the unweighted baseline).
4. Choose w on VALIDATION only (highest recall at the headline FPR cap, then fewer false positives,
   then higher PR-AUC, then the smaller w). Test is printed only after the choice.

    python scripts/run_hardneg.py                                  # XGBoost on the 'plus' features
    python scripts/run_hardneg.py --model lgbm --weights 1 5 20 100 500
    python scripts/run_hardneg.py --dev                            # quick check (NOT reportable)
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
from iafd.hardneg import block_folds, oof_scores, mine_hard_negatives, fit_weighted
from iafd.evaluate import threshold_at_fpr, evaluate_scores, confusion_at_threshold, wilson_interval


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgb", choices=ALL_MODELS)
    ap.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    ap.add_argument("--weights", nargs="+", type=float, default=[1, 5, 20, 100, 500])
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--mine-fpr", type=float, default=0.001, help="out-of-fold FPR defining a hard negative")
    ap.add_argument("--dev", action="store_true", help="use the dev sample (NOT reportable)")
    args = ap.parse_args()
    if 1.0 not in args.weights:
        args.weights = [1.0] + args.weights

    path = resolve(cfg["paths"]["dev_parquet" if args.dev else "parquet"])
    if not path.exists():
        sys.exit(f"{path} not found. Run scripts/prepare_data.py first.")
    df = load_dataset(path)
    s = cfg["split"]
    train, val, test = chronological_split(df, s["train_end_step"], s["val_end_step"])
    print("Split:", check_split(train, val, test))
    Xtr, Xva, Xte = (build_features(d, args.feature_set) for d in (train, val, test))
    ytr, yva, yte = (d["isFraud"].to_numpy() for d in (train, val, test))
    cap, seed = cfg["constraints"]["fpr_cap"], cfg["seed"]
    te_step = test["step"].to_numpy()
    cut = s.get("test_slice_step", 500)
    slices = {"early": te_step <= cut, "late": te_step > cut}
    params = cfg.get("models", {}).get(args.model)
    make = lambda: build_model(args.model, seed, params)

    print(f"model={args.model} features={args.feature_set} | headline cap={cap:.4%} | mining OOF FPR={args.mine_fpr:.3%} | folds={args.folds}")
    t0 = time.time()
    folds = block_folds(train["step"].to_numpy(), args.folds)
    oof = oof_scores(make, Xtr, ytr, folds)
    hn = mine_hard_negatives(oof, ytr, args.mine_fpr)
    tr_t = train["type"].astype(str).to_numpy()
    print(f"\nOut-of-fold scoring took {time.time() - t0:.0f}s. Hard negatives mined: {int(hn.sum()):,} of {int((ytr == 0).sum()):,} legit train rows")
    if hn.sum() == 0:
        sys.exit("No hard negatives found; nothing to weight.")
    xh = Xtr[hn]
    print(f"  types: {pd.Series(tr_t[hn]).value_counts().to_dict()} | median amount={np.median(xh['amount']):,.0f} | "
          f"sender emptied={float(np.mean((xh['oldbalanceOrg'] > 0) & (xh['newbalanceOrig'] == 0))):.0%}")
    print(f"  OOF score range of hard negatives: {oof[hn].min():.4f} to {oof[hn].max():.4f}\n\nVALIDATION (used to choose the weight):")

    def evaluate(pv, pt):
        thr = threshold_at_fpr(yva, pv, cap)
        v, t = evaluate_scores(yva, pv, thr), evaluate_scores(yte, pt, thr)
        lo, hi = wilson_interval(t["tp"], t["tp"] + t["fn"])
        out = {"val_recall": v["recall"], "val_fp": v["fp"], "val_pr_auc": v["pr_auc"],
               "test_recall": t["recall"], "recall_ci_lo": lo, "recall_ci_hi": hi, "test_fp": t["fp"],
               "flags_nothing": (t["tp"] + t["fp"]) == 0}
        for sl, m in slices.items():
            c = confusion_at_threshold(yte[m], pt[m], thr)
            out.update({f"{sl}_recall": c["recall"], f"{sl}_fp": c["fp"]})
        return out

    grid = []
    for w in args.weights:
        t1 = time.time()
        sw = np.ones(len(ytr))
        sw[hn] = w
        m = fit_weighted(make(), Xtr, ytr, sw)
        r = {"model": args.model, "weight": w, **evaluate(m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])}
        grid.append(r)
        print(f"  w={w:>6g}: val recall@cap={r['val_recall']:.4f}  val fp={r['val_fp']:>4d}  val PR-AUC={r['val_pr_auc']:.4f}  ({time.time() - t1:.0f}s)", flush=True)
    gdf = pd.DataFrame(grid)

    g = gdf.assign(r=lambda d: d["val_recall"].round(4))
    best = float(g.sort_values(["r", "val_fp", "val_pr_auc", "weight"], ascending=[False, True, False, True]).iloc[0]["weight"])
    print(f"\nChosen hard-negative weight (by validation): {best:g}")
    rows = [("w=1 (baseline)", 1.0)] + ([] if best == 1.0 else [(f"w={best:g} (chosen)", best)])
    res = pd.DataFrame([{"setting": lab, **gdf[gdf["weight"] == w].iloc[0].to_dict()} for lab, w in rows])

    out_dir = resolve(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ("dev_" if args.dev else "") + f"F_{args.model}_{args.feature_set}"
    gdf.to_csv(out_dir / f"experiment_{tag}_grid.csv", index=False)
    res.to_csv(out_dir / f"experiment_{tag}_selected.csv", index=False)

    pd.set_option("display.width", 220)
    show = res.assign(recall_95ci=lambda d: [f"[{a:.4f}, {b:.4f}]" for a, b in zip(d["recall_ci_lo"], d["recall_ci_hi"])])
    print(f"\nTEST at FPR cap {cap:.4%} (threshold chosen on validation): baseline vs chosen weight")
    print(show[["setting", "test_recall", "recall_95ci", "test_fp", "early_recall", "late_recall", "early_fp", "late_fp"]]
          .round(4).to_string(index=False))
    print(f"\nSaved reports/experiment_{tag}_grid.csv and _selected.csv")


if __name__ == "__main__":
    main()
