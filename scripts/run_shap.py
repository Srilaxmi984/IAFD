"""SHAP explanations for the soft-voting ensemble (the project's core model).

Trains the final ensemble (default: LR, RF, XGB, LightGBM; DT is left out because its scores are too coarse),
sets the threshold on VALIDATION at the headline FPR cap, then explains TEST rows:
  * GLOBAL: mean |SHAP| of the ensemble on fraud rows and on legit rows (permutation explainer, log-odds scale)
  * CROSS-CHECK: exact TreeSHAP for XGBoost on a larger sample, and the rank agreement between the two
  * LOCAL: waterfall plots for a caught fraud, the lowest-scoring (missed) fraud and the top false positive
Nothing here is used to choose models or thresholds. Also saves the fitted ensemble to models/ for the app.

    python scripts/run_shap.py
    python scripts/run_shap.py --n-fraud 200 --n-legit 200 --max-evals 300      # faster
    python scripts/run_shap.py --dev                                              # quick check (NOT reportable)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iafd.config import load_config, resolve
from iafd.data import load_dataset, chronological_split, check_split
from iafd.features import build_features, FEATURE_SETS
from iafd.models import ALL_MODELS, build_model
from iafd.evaluate import threshold_at_fpr, confusion_at_threshold
from iafd.explain import ensemble_proba, make_ensemble_logit_fn, logit, mean_abs, rank_agreement, pick_cases


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=["lr", "rf", "xgb", "lgbm"], choices=ALL_MODELS)
    ap.add_argument("--n-fraud", type=int, default=300, help="test fraud rows explained with the permutation explainer")
    ap.add_argument("--n-legit", type=int, default=300, help="test legit rows explained with the permutation explainer")
    ap.add_argument("--background", type=int, default=100, help="train rows used as the SHAP background")
    ap.add_argument("--max-evals", type=int, default=400, help="model evaluations per explained row")
    ap.add_argument("--dev", action="store_true", help="use the dev sample (NOT reportable)")
    args = ap.parse_args()
    if "xgb" not in args.models:
        sys.exit("the cross-check needs xgb in --models")

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
    cols = list(Xtr.columns)

    models = {}
    for name in args.models:
        t0 = time.time()
        models[name] = build_model(name, seed, cfg.get("models", {}).get(name)).fit(Xtr, ytr)
        print(f"  trained {name:5s} ({time.time() - t0:.0f}s)", flush=True)
    members = list(models.values())

    pv, pt = ensemble_proba(members, Xva), ensemble_proba(members, Xte)
    thr = threshold_at_fpr(yva, pv, cap)
    c = confusion_at_threshold(yte, pt, thr)
    print(f"\nEnsemble ({'+'.join(args.models)}) on {fs} features at FPR cap {cap:.4%}: "
          f"test recall={c['recall']:.4f} fp={c['fp']} (threshold {thr:.5f} on the mean probability)")

    out_dir, fig_dir = resolve(cfg["paths"]["reports_dir"]), resolve(cfg["paths"]["reports_dir"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mdir = resolve(cfg["paths"]["models_dir"])
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "threshold": float(thr), "columns": cols, "feature_set": fs, "fpr_cap": cap,
                 "members": args.models}, mdir / "ensemble.joblib")

    # ---- which rows to explain
    rng = np.random.default_rng(seed)
    cases = pick_cases(yte, pt, thr)
    fr, lg = np.flatnonzero(yte == 1), np.flatnonzero(yte == 0)
    rows = np.unique(np.r_[rng.choice(fr, min(args.n_fraud, len(fr)), replace=False),
                           rng.choice(lg, min(args.n_legit, len(lg)), replace=False), list(cases.values())])
    Xe = Xte.iloc[rows]
    ye = yte[rows]
    bg = Xtr.iloc[rng.choice(len(Xtr), args.background, replace=False)].to_numpy()

    # ---- ensemble SHAP (log-odds of the mean probability)
    f = make_ensemble_logit_fn(members, cols)
    t0 = time.time()
    explainer = shap.Explainer(f, shap.maskers.Independent(bg, max_samples=args.background), algorithm="permutation")
    ex = explainer(Xe.to_numpy(), max_evals=args.max_evals)
    vals, base = ex.values, np.asarray(ex.base_values)
    err = np.abs(vals.sum(axis=1) + base - f(Xe.to_numpy())).max()
    print(f"Ensemble SHAP on {len(rows)} rows took {time.time() - t0:.0f}s; additivity error (max) = {err:.2e}")

    # ---- XGBoost exact TreeSHAP on a larger sample (cross-check)
    xs = np.unique(np.r_[rng.choice(fr, min(2000, len(fr)), replace=False), rng.choice(lg, min(3000, len(lg)), replace=False)])
    Xx, yx = Xte.iloc[xs], yte[xs]
    xv = shap.TreeExplainer(models["xgb"]).shap_values(Xx)
    xv = xv[1] if isinstance(xv, list) else xv

    tab = pd.DataFrame({
        "feature": cols,
        "ensemble_fraud_rows": mean_abs(vals[ye == 1]),
        "ensemble_legit_rows": mean_abs(vals[ye == 0]),
        "xgb_tree_fraud_rows": mean_abs(xv[yx == 1]),
        "xgb_tree_legit_rows": mean_abs(xv[yx == 0]),
    }).sort_values("ensemble_fraud_rows", ascending=False)
    tab.to_csv(out_dir / "shap_global_importance.csv", index=False)
    rho = rank_agreement(tab["ensemble_fraud_rows"].to_numpy(), tab["xgb_tree_fraud_rows"].to_numpy())
    print(f"\nRank agreement (Spearman) between ensemble SHAP and exact XGBoost TreeSHAP on fraud rows: {rho:.3f}")
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print("\nMean |SHAP| (log-odds), top 12 by the ensemble on fraud rows:")
    print(tab.head(12).to_string(index=False))

    # ---- figures
    top = tab.head(12).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["ensemble_fraud_rows"], color="#1f77b4")
    plt.xlabel("mean |SHAP| on fraud rows (log-odds of the ensemble score)")
    plt.title("Ensemble: what drives the fraud score")
    plt.tight_layout(); plt.savefig(fig_dir / "ensemble_global_bar.png", dpi=150); plt.close()

    plt.figure()
    shap.summary_plot(xv, Xx, show=False, max_display=12)
    plt.tight_layout(); plt.savefig(fig_dir / "xgb_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

    pos = {int(r): i for i, r in enumerate(rows)}
    case_rows = []
    for name, idx in cases.items():
        i = pos[idx]
        expl = shap.Explanation(values=vals[i], base_values=float(base[i]), data=Xe.iloc[i].to_numpy(), feature_names=cols)
        plt.figure()
        shap.plots.waterfall(expl, max_display=10, show=False)
        flagged = pt[idx] > thr
        plt.title(f"{name.replace('_', ' ')} | ensemble score {pt[idx]:.4f} | {'flagged' if flagged else 'not flagged'}", fontsize=10)
        plt.savefig(fig_dir / f"waterfall_{name}.png", dpi=150, bbox_inches="tight"); plt.close()
        case_rows.append({"case": name, "test_row": int(idx), "score": float(pt[idx]), "flagged": bool(flagged),
                          "isFraud": int(yte[idx]), **Xte.iloc[idx].to_dict()})
        top3 = pd.Series(vals[i], index=cols).sort_values(key=np.abs, ascending=False).head(3)
        print(f"\n{name}: score {pt[idx]:.4f} ({'flagged' if flagged else 'not flagged'}); strongest drivers: " +
              ", ".join(f"{k} ({v:+.2f})" for k, v in top3.items()))
    pd.DataFrame(case_rows).to_csv(out_dir / "shap_cases.csv", index=False)
    print(f"\nSaved reports/shap_global_importance.csv, reports/shap_cases.csv, reports/figures/*.png "
          f"and models/ensemble.joblib")


if __name__ == "__main__":
    main()
