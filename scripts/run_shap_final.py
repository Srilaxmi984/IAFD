"""SHAP explanations for the TWO FINAL models (loaded from models/, written by scripts/final_evaluate.py).

For TEST rows of the gated types (the rows the models actually score) it computes
  * permutation SHAP (log-odds) for BOTH final models on the same sample, so they are comparable;
  * exact TreeSHAP for the single model when it is a tree model, as a cross-check;
and saves global importance (mean |SHAP| on fraud and on legit rows), agreement between the two models' feature
rankings, and waterfall plots for a caught fraud, the lowest-scoring fraud and the top false positive of each model.
Explanation only: nothing here is used to choose models or thresholds.

    python scripts/run_shap_final.py
    python scripts/run_shap_final.py --n-fraud 150 --n-legit 150 --max-evals 250     # faster
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
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iafd.config import load_config, resolve
from iafd.data import load_dataset, chronological_split
from iafd.explain import logit, mean_abs, rank_agreement, pick_cases
from iafd.features import build_features
from iafd.serving import load_bundles, model_proba, TREE_MODELS, _positive_class


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fraud", type=int, default=300)
    ap.add_argument("--n-legit", type=int, default=300)
    ap.add_argument("--background", type=int, default=100)
    ap.add_argument("--max-evals", type=int, default=400)
    ap.add_argument("--dev", action="store_true", help="use the dev sample and dev_ models (NOT reportable)")
    args = ap.parse_args()

    tag = "dev_" if args.dev else ""
    mdir = resolve(cfg["paths"]["models_dir"])
    try:
        import joblib
        bundles = {"single": joblib.load(mdir / f"{tag}final_single.joblib"), "ensemble": joblib.load(mdir / f"{tag}final_ensemble.joblib")}
    except FileNotFoundError as e:
        sys.exit(f"{e}\nRun scripts/final_evaluate.py first.")
    fs, cols, seed = bundles["single"]["feature_set"], bundles["single"]["columns"], cfg["seed"]
    allowed = bundles["single"].get("allowed_types") or sorted(set(["TRANSFER", "CASH_OUT"]))

    df = load_dataset(resolve(cfg["paths"]["dev_parquet" if args.dev else "parquet"]))
    s = cfg["split"]
    train, _, test = chronological_split(df, s["train_end_step"], s["val_end_step"])
    test = test[test["type"].astype(str).isin(allowed)]
    train = train[train["type"].astype(str).isin(allowed)]
    Xte, yte = build_features(test, fs)[cols], test["isFraud"].to_numpy()
    Xtr = build_features(train, fs)[cols]
    rng = np.random.default_rng(seed)
    fr, lg = np.flatnonzero(yte == 1), np.flatnonzero(yte == 0)
    bg = Xtr.iloc[rng.choice(len(Xtr), args.background, replace=False)].to_numpy()
    fig_dir = resolve(cfg["paths"]["reports_dir"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print(f"Explaining {len(test):,} test rows of types {allowed}; features={fs}\n")

    # the same sample for both models, plus each model's own three cases
    scores = {k: model_proba(b, Xte) for k, b in bundles.items()}
    cases = {k: pick_cases(yte, scores[k], float(b["thresholds"]["cap"])) for k, b in bundles.items()}
    extra = [i for k in cases for i in cases[k].values()]
    rows = np.unique(np.r_[rng.choice(fr, min(args.n_fraud, len(fr)), replace=False),
                           rng.choice(lg, min(args.n_legit, len(lg)), replace=False), extra])
    Xe, ye = Xte.iloc[rows], yte[rows]
    pos = {int(r): i for i, r in enumerate(rows)}

    table = pd.DataFrame({"feature": cols})
    values = {}
    for key, b in bundles.items():
        f = lambda Xn, b=b: logit(model_proba(b, pd.DataFrame(np.asarray(Xn, dtype="float64"), columns=cols)))
        t0 = time.time()
        ex = shap.Explainer(f, shap.maskers.Independent(bg, max_samples=args.background), algorithm="permutation")(
            Xe.to_numpy(), max_evals=args.max_evals)
        vals, base = ex.values, np.asarray(ex.base_values)
        err = np.abs(vals.sum(axis=1) + base - f(Xe.to_numpy())).max()
        print(f"{key}: permutation SHAP on {len(rows)} rows took {time.time() - t0:.0f}s; additivity error (max) {err:.2e}")
        values[key] = (vals, base)
        table[f"{key}_fraud_rows"] = mean_abs(vals[ye == 1])
        table[f"{key}_legit_rows"] = mean_abs(vals[ye == 0])

    b1 = bundles["single"]
    if b1.get("model") in TREE_MODELS:
        xs = np.unique(np.r_[rng.choice(fr, min(2000, len(fr)), replace=False), rng.choice(lg, min(3000, len(lg)), replace=False)])
        tv = _positive_class(shap.TreeExplainer(b1["estimator"]).shap_values(Xte.iloc[xs]))
        table["single_tree_fraud_rows"] = mean_abs(tv[yte[xs] == 1])
        rho_t = rank_agreement(table["single_fraud_rows"].to_numpy(), table["single_tree_fraud_rows"].to_numpy())
        print(f"Single model: rank agreement between permutation SHAP and exact TreeSHAP = {rho_t:.3f} "
              f"(the two use different output scales: log-odds against probability)")
    table = table.sort_values("single_fraud_rows", ascending=False)
    table.to_csv(resolve(cfg["paths"]["reports_dir"]) / f"{tag}shap_final_importance.csv", index=False)
    rho = rank_agreement(table["single_fraud_rows"].to_numpy(), table["ensemble_fraud_rows"].to_numpy())
    print(f"\nRank agreement between the single model and the ensemble (fraud rows): {rho:.3f}")
    print("\nMean |SHAP| on fraud rows (log-odds), top 12 by the single model:")
    print(table[["feature", "single_fraud_rows", "ensemble_fraud_rows"]].head(12).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    top = table.head(12).iloc[::-1]
    for a, key, color in zip(ax, ("single", "ensemble"), ("#1f77b4", "#2ca02c")):
        a.barh(top["feature"], top[f"{key}_fraud_rows"], color=color)
        a.set_title(f"{key} model: mean |SHAP| on fraud rows")
        a.set_xlabel("log-odds")
    plt.tight_layout(); plt.savefig(fig_dir / f"{tag}shap_final_global.png", dpi=150); plt.close()

    for key, b in bundles.items():
        vals, base = values[key]
        for name, idx in cases[key].items():
            i = pos[int(idx)]
            expl = shap.Explanation(values=vals[i], base_values=float(base[i]), data=Xte.iloc[idx].to_numpy(), feature_names=cols)
            plt.figure()
            shap.plots.waterfall(expl, max_display=10, show=False)
            flagged = scores[key][idx] > float(b["thresholds"]["cap"])
            label = "lowest-scoring fraud" if name == "missed_fraud" else name.replace("_", " ")
            plt.title(f"{key}: {label} | score {scores[key][idx]:.4f} | {'flagged' if flagged else 'not flagged'}", fontsize=9)
            plt.savefig(fig_dir / f"{tag}shap_final_{key}_{name}.png", dpi=150, bbox_inches="tight"); plt.close()
            top3 = pd.Series(vals[i], index=cols).sort_values(key=np.abs, ascending=False).head(3)
            print(f"  {key:8s} {('lowest_fraud' if name == 'missed_fraud' else name):15s} score {scores[key][idx]:.4f} ({'flagged' if flagged else 'not flagged'}): " +
                  ", ".join(f"{k} ({v:+.2f})" for k, v in top3.items()))
    print(f"\nSaved reports/{tag}shap_final_importance.csv and reports/figures/{tag}shap_final_*.png")


if __name__ == "__main__":
    main()
