"""Evaluate the two FINAL models on TEST (once), against the paper's setup, under the same chronological protocol.

Reads reports/final_selection.json (written by final_select.py, validation only). For every seed it retrains
  * FINAL ENSEMBLE  : the selected members and aggregator (features from the selection)
  * FINAL SINGLE    : the selected model and imbalance strategy
  * PAPER BASELINE  : the paper's 5 raw features, LR+DT+RF+XGB+LightGBM, soft vote
and evaluates each at three operating points, all with thresholds chosen on VALIDATION:
  cap      : false-positive rate <= 0.01%
  recall   : recall >= 99.5% with as few false alarms as possible
  min_cost : minimises (value of missed frauds + alert_cost per false alert)
Reports missed frauds (FN), false alarms (FP), recall, FPR, MCC, F2, PR-AUC, LFR and cost, with seed spread,
95% intervals and early/late test regimes. Saves the seed-1 models to models/ for the app.

    python scripts/final_evaluate.py                 # 3 seeds
    python scripts/final_evaluate.py --seeds 1       # faster
    python scripts/final_evaluate.py --alert-cost 1000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iafd.config import load_config, resolve
from iafd.data import load_dataset, chronological_split, check_split
from iafd.features import build_features
from iafd.models import build_model
from iafd.evaluate import (threshold_at_fpr, threshold_at_recall, confusion_at_threshold, recall_fpr_table, wilson_interval)
from iafd.final import (mcc, f_beta, cost_at_threshold, threshold_min_cost, combine_spec, strategy_pos_weight, LOW_FPR_CAPS,
                        allowed_mask, predict_gated)

PAPER_MODELS = ["lr", "dt", "rf", "xgb", "lgbm"]
OPS = ["cap", "recall", "min_cost"]
OP_LABEL = {"cap": "FPR <= 0.01%", "recall": "recall >= 99.5%", "min_cost": "minimum cost"}


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--alert-cost", type=float, default=10000.0, help="cost of one false alert, in the data's currency (an assumption; see the sensitivity table)")
    ap.add_argument("--skip-paper", action="store_true")
    ap.add_argument("--dev", action="store_true", help="use the dev sample (NOT reportable)")
    args = ap.parse_args()

    tag = "dev_" if args.dev else ""
    out_dir = resolve(cfg["paths"]["reports_dir"])
    sel_path = out_dir / f"{tag}final_selection.json"
    if not sel_path.exists():
        sys.exit(f"{sel_path} not found. Run scripts/final_select.py first.")
    sel = json.loads(sel_path.read_text())
    fs = sel["feature_set"]
    members = [m if isinstance(m, dict) else {"model": m, "strategy": "plain"} for m in sel["ensemble"]["members"]]
    agg_mode, weights = sel["ensemble"]["aggregator"], sel["ensemble"].get("weights")
    single_model, single_strategy = sel["single"]["model"], sel["single"]["strategy"]
    gate, allowed = sel.get("gate", False), sel.get("allowed_types", [])

    path = resolve(cfg["paths"]["dev_parquet" if args.dev else "parquet"])
    df = load_dataset(path)
    s = cfg["split"]
    train, val, test = chronological_split(df, s["train_end_step"], s["val_end_step"])
    print("Split:", check_split(train, val, test))
    ytr, yva, yte = (d["isFraud"].to_numpy() for d in (train, val, test))
    amt_va, amt_te = val["amount"].to_numpy(), test["amount"].to_numpy()
    cut = s.get("test_slice_step", 500)
    early = test["step"].to_numpy() <= cut
    ref_prev = float(ytr.mean())
    gm = {k: (allowed_mask(d["type"], allowed) if gate else np.ones(len(d), bool)) for k, d in (("tr", train), ("va", val), ("te", test))}
    F = {name: tuple(build_features(d, name) for d in (train, val, test)) for name in {fs, "paper"} if not (name == "paper" and args.skip_paper)}
    mdesc = "+".join(f"{m['model']}({m['strategy']})" for m in members)
    print(f"Final ensemble: {mdesc} [{agg_mode} {weights or ''}] on '{fs}' | final single: {single_model} ({single_strategy}) | "
          f"type gating={'ON ' + str(allowed) if gate else 'OFF'} | alert cost={args.alert_cost:g} | seeds={args.seeds}\n", flush=True)

    base_seed = cfg["seed"]
    rows, curves, saved, seed0 = [], {}, {}, {}
    for si in range(args.seeds):
        seed = base_seed + si
        cache = {}

        def probs(feat, name, strategy="plain", gated=False):
            key = (feat, name, strategy, gated)
            if key not in cache:
                Xtr, Xva, Xte = F[feat]
                t0 = time.time()
                mt = gm["tr"] if gated else np.ones(len(Xtr), bool)
                w = strategy_pos_weight(ytr[mt], strategy)
                m = build_model(name, seed, cfg.get("models", {}).get(name), pos_weight=w).fit(Xtr[mt], ytr[mt])
                mv = gm["va"] if gated else np.ones(len(Xva), bool)
                me = gm["te"] if gated else np.ones(len(Xte), bool)
                cache[key] = (m, predict_gated(m, Xva, mv), predict_gated(m, Xte, me))
                print(f"  seed {seed}: trained {feat}/{name}/{strategy}{'/gated' if gated else ''} ({time.time() - t0:.0f}s)", flush=True)
            return cache[key]

        systems = {}
        mem = [probs(fs, m["model"], m["strategy"], gate) for m in members]
        systems["final_ensemble"] = (combine_spec([x[1] for x in mem], agg_mode, weights), combine_spec([x[2] for x in mem], agg_mode, weights))
        sm = probs(fs, single_model, single_strategy, gate)
        systems["final_single"] = (sm[1], sm[2])
        if not args.skip_paper:
            pm = [probs("paper", m) for m in PAPER_MODELS]
            systems["paper_ensemble"] = (combine_spec([x[1] for x in pm], "mean"), combine_spec([x[2] for x in pm], "mean"))

        for sysname, (pv, pt) in systems.items():
            thr = {"cap": threshold_at_fpr(yva, pv, 1e-4), "recall": threshold_at_recall(yva, pv, 0.995),
                   "min_cost": threshold_min_cost(yva, pv, amt_va, args.alert_cost)}
            lfr_t = float(np.mean([r["test_recall"] for r in recall_fpr_table(yva, pv, yte, pt, LOW_FPR_CAPS)]))
            pr_auc = float(average_precision_score(yte, pt))
            for op in OPS:
                c = confusion_at_threshold(yte, pt, thr[op])
                ce = confusion_at_threshold(yte[early], pt[early], thr[op])
                cl = confusion_at_threshold(yte[~early], pt[~early], thr[op])
                cost = cost_at_threshold(yte, pt, amt_te, thr[op], args.alert_cost)
                lo, hi = wilson_interval(c["tp"], c["tp"] + c["fn"])
                rows.append({"seed": seed, "system": sysname, "operating_point": op, "threshold": thr[op],
                             "missed_frauds_FN": c["fn"], "false_alarms_FP": c["fp"], "caught_TP": c["tp"],
                             "recall": c["recall"], "recall_ci_lo": lo, "recall_ci_hi": hi, "fpr": c["fpr"],
                             "precision": c["precision"],
                             "precision_at_train_prev": (c["recall"] * ref_prev) / max(c["recall"] * ref_prev + c["fpr"] * (1 - ref_prev), 1e-12),
                             "mcc": mcc(c["tp"], c["fp"], c["fn"], c["tn"]), "f2": f_beta(c["tp"], c["fp"], c["fn"], 2.0),
                             "pr_auc": pr_auc, "lfr_test": lfr_t,
                             "early_recall": ce["recall"], "early_fp": ce["fp"], "late_recall": cl["recall"], "late_fp": cl["fp"],
                             **cost})
            if si == 0:
                seed0[sysname] = (pv, pt)
                caps = np.logspace(-5, -2.3, 12)
                curves[sysname] = recall_fpr_table(yva, pv, yte, pt, caps)
                saved[sysname] = thr

        if si == 0:
            mdir = resolve(cfg["paths"]["models_dir"])
            mdir.mkdir(parents=True, exist_ok=True)
            cols = list(F[fs][0].columns)
            joblib.dump({"members": members, "models": [probs(fs, m["model"], m["strategy"], gate)[0] for m in members],
                         "aggregator": agg_mode, "weights": weights, "gate": gate, "allowed_types": allowed,
                         "thresholds": saved["final_ensemble"], "columns": cols, "feature_set": fs},
                        mdir / f"{tag}final_ensemble.joblib")
            joblib.dump({"model": single_model, "strategy": single_strategy, "estimator": probs(fs, single_model, single_strategy, gate)[0],
                         "gate": gate, "allowed_types": allowed,
                         "thresholds": saved["final_single"], "columns": cols, "feature_set": fs},
                        mdir / f"{tag}final_single.joblib")

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / f"{tag}final_results_by_seed.csv", index=False)

    g = res.groupby(["operating_point", "system"], sort=False)
    mean, sd = g.mean(numeric_only=True), g.std(numeric_only=True).fillna(0.0)
    order = {"final_ensemble": 0, "final_single": 1, "paper_ensemble": 2}
    pd.set_option("display.width", 220)
    md = [f"# Final results (test set, thresholds chosen on validation, {args.seeds} seed(s))\n",
          f"Final ensemble: {mdesc} [{agg_mode} {weights or ''}]; final single model: {single_model} ({single_strategy}); "
          f"features: {fs}; type gating: {'on (' + ', '.join(allowed) + ')' if gate else 'off'}. Paper baseline: raw 5 features, LR+DT+RF+XGB+LightGBM soft vote. Alert cost per false alarm: {args.alert_cost:g}.\n"]
    summary = []
    for op in OPS:
        tab = []
        for sysname in sorted({k[1] for k in mean.index if k[0] == op}, key=order.get):
            m_, s_ = mean.loc[(op, sysname)], sd.loc[(op, sysname)]
            fmt = lambda col, d=0: f"{m_[col]:.{d}f} ± {s_[col]:.{d}f}" if args.seeds > 1 else f"{m_[col]:.{d}f}"
            tab.append({"system": sysname, "missed (FN)": fmt("missed_frauds_FN"), "false alarms (FP)": fmt("false_alarms_FP"),
                        "recall": fmt("recall", 4), "FPR": f"{m_['fpr']:.5f}", "MCC": fmt("mcc", 3), "F2": fmt("f2", 3),
                        "PR-AUC": fmt("pr_auc", 4), "LFR (test)": fmt("lfr_test", 4), "cost saving": f"{m_['saving_vs_no_model']:,.0f}"})
        t = pd.DataFrame(tab)
        summary.append((op, t))
        md += [f"\n## Operating point: {OP_LABEL[op]}\n", md_table(t), ""]
        print(f"\nOPERATING POINT: {OP_LABEL[op]}  (mean ± sd over {args.seeds} seed(s))")
        print(t.to_string(index=False))
    first = res[(res["seed"] == base_seed) & (res["operating_point"] == "cap")].set_index("system")
    md.append("\n## Early and late test regimes at FPR <= 0.01% (first seed)\n")
    reg = pd.DataFrame([{"system": k, "early recall": f"{r.early_recall:.4f}", "early FP": int(r.early_fp),
                         "late recall": f"{r.late_recall:.4f}", "late FP": int(r.late_fp),
                         "recall 95% CI": f"[{r.recall_ci_lo:.4f}, {r.recall_ci_hi:.4f}]"} for k, r in first.iterrows()])
    md.append(md_table(reg))
    print("\nEarly / late regimes at FPR <= 0.01% (first seed):")
    print(reg.to_string(index=False))
    sens = []
    for ac in (100.0, 1000.0, 10000.0, 50000.0):
        for sysname in sorted(seed0, key=order.get):
            pv, pt = seed0[sysname]
            thr_c = threshold_min_cost(yva, pv, amt_va, ac)
            c = confusion_at_threshold(yte, pt, thr_c)
            cost = cost_at_threshold(yte, pt, amt_te, thr_c, ac)
            sens.append({"alert cost": f"{ac:,.0f}", "system": sysname, "missed (FN)": c["fn"], "false alarms (FP)": c["fp"],
                         "recall": f"{c['recall']:.4f}", "cost saving": f"{cost['saving_vs_no_model']:,.0f}"})
    sens = pd.DataFrame(sens)
    md += ["\n## Sensitivity of the minimum-cost operating point to the alert cost (first seed)\n",
           "The cost of one false alert is an assumption; a missed fraud costs its amount. Cheaper alerts push the optimum towards flagging more.\n",
           md_table(sens)]
    print("\nMinimum-cost operating point at different alert costs (first seed):")
    print(sens.to_string(index=False))
    (out_dir / f"{tag}FINAL_RESULTS.md").write_text("\n".join(md))

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for k, rr in curves.items():
        plt.semilogx([r["test_fpr"] + 1e-9 for r in rr], [r["test_recall"] for r in rr], marker="o", label=k)
    plt.xlabel("test false-positive rate (log scale)"); plt.ylabel("test recall")
    plt.title("Recall vs false-positive rate (thresholds from validation)"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(fig_dir / f"{tag}final_recall_vs_fpr.png", dpi=150); plt.close()

    cap = res[(res["operating_point"] == "cap")].groupby("system", sort=False)[["missed_frauds_FN", "false_alarms_FP"]].mean()
    cap = cap.loc[sorted(cap.index, key=order.get)]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    ax[0].bar(cap.index, cap["missed_frauds_FN"], color="#d62728"); ax[0].set_title("Missed frauds (FN) at FPR <= 0.01%")
    ax[1].bar(cap.index, cap["false_alarms_FP"], color="#1f77b4"); ax[1].set_title("False alarms (FP) at FPR <= 0.01%")
    for a in ax: a.tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig(fig_dir / f"{tag}final_fn_fp.png", dpi=150); plt.close()
    print(f"\nSaved reports/{tag}final_results_by_seed.csv, reports/{tag}FINAL_RESULTS.md, "
          f"reports/figures/{tag}final_*.png and models/{tag}final_ensemble.joblib, models/{tag}final_single.joblib")


if __name__ == "__main__":
    main()
