"""Choose the two FINAL models using VALIDATION data only (test is never touched here).

  FINAL SINGLE MODEL: every candidate model is trained at the natural class imbalance and at more extreme ones
  (only a fraction of the TRAIN frauds kept), with and without a class weight. The winner has the best mean LFR
  (mean recall at low FPR caps, on validation) across the imbalance levels, i.e. it degrades least as imbalance grows.
  FINAL ENSEMBLE: every combination of 2+ members, averaged by mean, logit-mean, or a validation-fitted WEIGHTED logit
  blend (weights on a 0.1 grid, all strictly positive), is scored on validation AT EVERY IMBALANCE LEVEL, because at the
  natural imbalance validation saturates (many candidates are perfect). Ranking is lexicographic:
     1. mean LFR across levels  2. worst-level LFR  3. false alarms needed for 100% recall  4. ... for 99.5% recall
     5. fewer members          6. weights closer to equal   (the script prints how many candidates still tie)
  TYPE GATING (default on): transaction types with no fraud in TRAIN are excluded from fitting and scored exactly 0.
  They still count in every FPR denominator.

    python scripts/final_select.py
    python scripts/final_select.py --levels 1.0 0.1 --models lr xgb lgbm      # faster
    python scripts/final_select.py --no-gate                                   # without type gating
    python scripts/final_select.py --dev                                       # quick check (NOT reportable)
Writes reports/final_selection.json plus reports/final_selection_single.csv and _ensemble.csv.
"""
import argparse
import itertools
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
from iafd.explain import logit
from iafd.final import (LOW_FPR_CAPS, selection_profile, fraud_subsample_mask, strategy_pos_weight, combine,
                        positive_simplex, allowed_types, allowed_mask, predict_gated)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=["lr", "rf", "xgb", "lgbm"], choices=ALL_MODELS)
    ap.add_argument("--levels", nargs="+", type=float, default=[1.0, 0.1, 0.03],
                    help="fraction of TRAIN frauds kept (1.0 = natural imbalance)")
    ap.add_argument("--strategies", nargs="+", default=["plain", "sqrtw"], choices=["plain", "sqrtw"])
    ap.add_argument("--no-gate", action="store_true", help="do not gate transaction types without training fraud")
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
    allowed = allowed_types(train["type"], ytr)
    gate = not args.no_gate
    m_tr = allowed_mask(train["type"], allowed) if gate else np.ones(len(train), bool)
    m_va = allowed_mask(val["type"], allowed) if gate else np.ones(len(val), bool)
    print(f"features={args.feature_set} | models={args.models} | imbalance levels={levels} | strategies={strategies}\n"
          f"type gating={'ON, allowed types ' + str(allowed) if gate else 'OFF'} | fit rows {int(m_tr.sum()):,} of {len(train):,} "
          f"| validation rows scored by the model {int(m_va.sum()):,} of {len(val):,}\n", flush=True)

    rows, vp = [], {}
    for level in levels:
        keep = fraud_subsample_mask(ytr, level, seed) & m_tr
        Xk, yk = Xtr[keep], ytr[keep]
        for strategy in strategies:
            w = strategy_pos_weight(yk, strategy)
            for name in args.models:
                t0 = time.time()
                m = build_model(name, seed, cfg.get("models", {}).get(name), pos_weight=w).fit(Xk, yk)
                pv = predict_gated(m, Xva, m_va)
                lfr, fp995, fp999 = selection_profile(yva, pv)
                c = confusion_at_threshold(yva, pv, threshold_at_fpr(yva, pv, 1e-4))
                rows.append({"model": name, "strategy": strategy, "level": level, "train_frauds": int(yk.sum()),
                             "weight": round(w, 2), "val_lfr": lfr, "val_fp_rec995": fp995, "val_fp_rec999": fp999,
                             "val_recall_at_0.01%": c["recall"], "val_fp_at_0.01%": c["fp"]})
                vp[(name, strategy, level)] = pv
                print(f"  level={level:<5g} {strategy:5s} {name:5s} train frauds={int(yk.sum()):>5d}  val LFR={lfr:.4f}  "
                      f"fp@rec99.5%={fp995:>6d}  recall@0.01%={c['recall']:.4f} fp={c['fp']:>4d}  ({time.time() - t0:.0f}s)", flush=True)
    sdf = pd.DataFrame(rows)

    # ---------------- single model: best mean LFR across imbalance levels
    agg = sdf.groupby(["model", "strategy"]).agg(mean_lfr=("val_lfr", "mean"), min_lfr=("val_lfr", "min"),
                                                 mean_fp995=("val_fp_rec995", "mean")).reset_index()
    agg["k1"], agg["k2"] = agg["mean_lfr"].round(4), agg["min_lfr"].round(4)
    agg["srank"] = agg["strategy"].map({"plain": 0, "sqrtw": 1})
    agg = agg.sort_values(["k1", "k2", "mean_fp995", "srank"], ascending=[False, False, True, True]).drop(columns=["k1", "k2", "srank"])
    best_single = agg.iloc[0]
    pd.set_option("display.width", 200, "display.float_format", "{:.4f}".format)
    print("\nSINGLE MODEL ranking (validation, across imbalance levels):")
    print(agg.to_string(index=False))
    print(f"\n=> FINAL SINGLE MODEL: {best_single['model']} with strategy '{best_single['strategy']}'")

    # ---------------- ensemble: members = each model's best strategy across the imbalance levels
    member_cfg = {r["model"]: r["strategy"] for _, r in agg.groupby("model").head(1).iterrows()}
    names = list(member_cfg)
    P = {lv: {n: vp[(n, member_cfg[n], lv)] for n in names} for lv in levels}
    Z = {lv: {n: logit(P[lv][n]) for n in names} for lv in levels}
    print("\nEnsemble members (each model's best strategy across imbalance levels):", member_cfg)
    RECALLS = (0.995, 0.999, 1.0)

    def over_levels(score_fn):
        prof = [selection_profile(yva, score_fn(lv), recalls=RECALLS) for lv in levels]
        return {"mean_lfr": float(np.mean([q[0] for q in prof])), "min_lfr": float(np.min([q[0] for q in prof])),
                "mean_fp995": float(np.mean([q[1] for q in prof])), "mean_fp100": float(np.mean([q[3] for q in prof]))}

    erows = []
    for size in range(2, len(names) + 1):
        for sub in itertools.combinations(names, size):
            for mode in ("mean", "logit_mean"):
                erows.append({"members": "+".join(sub), "n": size, "aggregator": mode, "weights": "", "dist": 0.0,
                              **over_levels(lambda lv: combine([P[lv][n] for n in sub], mode))})
            for w in positive_simplex(size):
                erows.append({"members": "+".join(sub), "n": size, "aggregator": "weighted_logit",
                              "weights": "/".join(f"{x:.1f}" for x in w), "dist": float(np.abs(np.array(w) - 1 / size).sum()),
                              **over_levels(lambda lv: sum(wi * Z[lv][n] for wi, n in zip(w, sub)))})   # ranking only: sigmoid is monotone
        print(f"  scored all {size}-member ensembles", flush=True)
    edf = pd.DataFrame(erows)
    edf["k1"], edf["k2"] = edf["mean_lfr"].round(4), edf["min_lfr"].round(4)
    edf = edf.sort_values(["k1", "k2", "mean_fp100", "mean_fp995", "n", "dist"],
                          ascending=[False, False, True, True, True, True]).drop(columns=["k1", "k2", "dist"])
    best = edf.iloc[0]
    tied = int(((edf["mean_lfr"].round(4) == round(best["mean_lfr"], 4)) & (edf["min_lfr"].round(4) == round(best["min_lfr"], 4))
                & (edf["mean_fp100"] == best["mean_fp100"]) & (edf["mean_fp995"] == best["mean_fp995"])).sum())
    print(f"\nENSEMBLE candidates scored: {len(edf)} (each at {len(levels)} imbalance levels). Top 10:")
    print(edf.head(10).to_string(index=False))
    if tied > 1:
        print(f"\nNOTE: {tied} candidates tie with the winner on every criterion; the winner is then the one with the "
              f"fewest members and weights closest to equal, NOT a proven best.")
    print(f"\n=> FINAL ENSEMBLE: {best['members']} | {best['aggregator']} {best['weights']}")

    out_dir = resolve(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "dev_" if args.dev else ""
    sdf.to_csv(out_dir / f"{tag}final_selection_single.csv", index=False)
    edf.to_csv(out_dir / f"{tag}final_selection_ensemble.csv", index=False)
    wts = [float(x) for x in str(best["weights"]).split("/")] if best["weights"] else None
    sel = {"feature_set": args.feature_set, "selection_seed": seed, "caps": list(LOW_FPR_CAPS), "levels": levels,
           "gate": gate, "allowed_types": allowed,
           "single": {"model": str(best_single["model"]), "strategy": str(best_single["strategy"]),
                      "mean_lfr": float(best_single["mean_lfr"]), "min_lfr": float(best_single["min_lfr"])},
           "ensemble": {"members": [{"model": n, "strategy": member_cfg[n]} for n in str(best["members"]).split("+")],
                        "aggregator": str(best["aggregator"]), "weights": wts, "mean_lfr": float(best["mean_lfr"]),
                        "min_lfr": float(best["min_lfr"]), "candidates_tied_with_winner": tied}}
    (out_dir / f"{tag}final_selection.json").write_text(json.dumps(sel, indent=2))
    print(f"\nSaved reports/{tag}final_selection.json (+ _single.csv, _ensemble.csv)")


if __name__ == "__main__":
    main()
