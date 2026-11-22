"""Comprehensive comparison between Approach A (Advanced Ensemble) and Approach B (Imbalance-Aware).

This module provides fair comparison with identical data splits, feature sets, and evaluation metrics.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# Internal
from .config import load_config, resolve
from .data import load_dataset, chronological_split, check_split
from .features import build_features, FEATURE_SETS
from .approach_a_ensemble import AdvancedEnsemble
from .approach_b_imbalance import ImbalanceAwareModel
from .evaluate import (
    threshold_at_fpr,
    confusion_at_threshold,
    evaluate_scores,
    wilson_interval,
    precision_recall_curve,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Comprehensive Metrics Computation
# ============================================================

def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    reference_prevalence: float | None = None,
) -> dict:
    """Compute all requested metrics at a given threshold."""
    c = confusion_at_threshold(y_true, y_scores, threshold)

    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    recall = c["recall"]
    fpr = c["fpr"]
    precision = c["precision"]

    # F1 and F2
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0

    # MCC
    mcc = matthews_corrcoef(y_true, y_scores > threshold)

    # AUCs
    pr_auc = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    # False Negative Rate
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    # Precision at reference prevalence
    precision_at_ref = None
    if reference_prevalence is not None:
        precision_at_ref = precision_at_prevalence(recall, fpr, reference_prevalence)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f2_score": f2,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "fpr": fpr,
        "fnr": fnr,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
        "precision_at_ref_prevalence": precision_at_ref,
    }


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> float:
    """Compute precision at a given prevalence."""
    tp = recall * prevalence
    fp = fpr * (1 - prevalence)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def evaluate_at_multiple_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: dict[str, float],
    reference_prevalence: float | None = None,
) -> dict[str, dict]:
    """Evaluate model at multiple operating thresholds."""
    results = {}
    for name, thr in thresholds.items():
        results[name] = compute_comprehensive_metrics(y_true, y_scores, thr, reference_prevalence)
    return results


def find_optimal_thresholds(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, float]:
    """Find various optimal thresholds."""
    from sklearn.metrics import precision_recall_curve

    thresholds = {}

    # Default 0.5
    thresholds["default_0.5"] = 0.5

    # F1-optimal
    precisions, recalls, thrs = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores[:-1])
    thresholds["f1_optimal"] = float(thrs[best_f1_idx])

    # F2-optimal (favors recall)
    f2_scores = 5 * (precisions * recalls) / (4 * precisions + recalls + 1e-10)
    best_f2_idx = np.argmax(f2_scores[:-1])
    thresholds["f2_optimal"] = float(thrs[best_f2_idx])

    # MCC-optimal
    unique_thrs = np.unique(y_scores)
    best_mcc = -1
    best_mcc_thr = 0.5
    for thr in unique_thrs[::max(1, len(unique_thrs)//1000)]:  # Sample for speed
        pred = y_scores > thr
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_mcc_thr = thr
    thresholds["mcc_optimal"] = float(best_mcc_thr)

    return thresholds


# ============================================================
# Comparison Pipeline
# ============================================================

def compare_approaches(
    approach_a_path: str | Path,
    approach_b_path: str | Path,
    feature_set: str = "plus",
    fpr_cap: float = 0.0001,
    use_dev: bool = False,
) -> dict:
    """Compare Approach A and B on identical test data."""
    cfg = load_config()

    # Load data
    path = resolve(cfg["paths"]["dev_parquet" if use_dev else "parquet"])
    df = load_dataset(path)
    s = cfg["split"]
    train, val, test = chronological_split(df, s["train_end_step"], s["val_end_step"])

    # Build features
    Xte = build_features(test, feature_set)
    yte = test["isFraud"].to_numpy()
    ref_prev = train["isFraud"].mean()

    print(f"Test set: {len(Xte):,} rows, {yte.sum():,} fraud ({yte.mean():.4%})")

    # Load models
    print(f"Loading Approach A from {approach_a_path}...")
    ensemble_a = AdvancedEnsemble.load(approach_a_path)

    print(f"Loading Approach B from {approach_b_path}...")
    model_b = ImbalanceAwareModel.load(approach_b_path)

    # Get predictions
    prob_a = ensemble_a.predict_proba(Xte)
    prob_b = model_b.predict_proba(Xte)

    # Evaluate at each model's validated threshold
    thr_a = ensemble_a.threshold
    thr_b = model_b.threshold

    print(f"\nApproach A threshold: {thr_a:.5f}")
    print(f"Approach B threshold: {thr_b:.5f}")

    # Comprehensive metrics at validated thresholds
    metrics_a = compute_comprehensive_metrics(yte, prob_a, thr_a, ref_prev)
    metrics_b = compute_comprehensive_metrics(yte, prob_b, thr_b, ref_prev)

    # Also evaluate at multiple thresholds
    thresholds_a = find_optimal_thresholds(yte, prob_a)
    thresholds_b = find_optimal_thresholds(yte, prob_b)

    # Add FPR-cap threshold (the validated one)
    thresholds_a["fpr_cap_validated"] = thr_a
    thresholds_b["fpr_cap_validated"] = thr_b

    multi_a = evaluate_at_multiple_thresholds(yte, prob_a, thresholds_a, ref_prev)
    multi_b = evaluate_at_multiple_thresholds(yte, prob_b, thresholds_b, ref_prev)

    # Build comparison table
    comparison = {
        "Approach A (Advanced Ensemble)": {
            "validated_threshold": thr_a,
            "metrics_at_validated": metrics_a,
            "metrics_at_all_thresholds": multi_a,
        },
        "Approach B (Imbalance-Aware)": {
            "validated_threshold": thr_b,
            "metrics_at_validated": metrics_b,
            "metrics_at_all_thresholds": multi_b,
        },
    }

    return comparison


def generate_comparison_table(comparison: dict) -> pd.DataFrame:
    """Generate the comparison table as specified in requirements."""
    rows = []

    for approach_name, data in comparison.items():
        m = data["metrics_at_validated"]
        row = {
            "Approach": approach_name,
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-Score": round(m["f1_score"], 4),
            "F2-Score": round(m["f2_score"], 4),
            "PR-AUC": round(m["pr_auc"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "MCC": round(m["mcc"], 4),
            "False Positive Rate": round(m["fpr"], 6),
            "False Negative Rate": round(m["fnr"], 6),
            "False Positives": m["false_positives"],
            "False Negatives": m["false_negatives"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_comparison_curves(
    comparison: dict,
    y_true: np.ndarray,
    save_dir: Path,
):
    """Generate comparison plots: PR curves, ROC curves, threshold analysis."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract probabilities for each approach
    # We need to re-load models to get probabilities - passing them would be better
    # For now, we'll create placeholder showing structure

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # PR Curves
    ax = axes[0, 0]
    for name, data in comparison.items():
        # We'd need the actual scores to plot curves
        # This is a placeholder - actual implementation needs scores
        ax.text(0.5, 0.5, "PR Curves - requires model scores", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()

    # ROC Curves
    ax = axes[0, 1]
    ax.text(0.5, 0.5, "ROC Curves - requires model scores", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    # Threshold vs Recall/FPR
    ax = axes[1, 0]
    ax.text(0.5, 0.5, "Threshold Analysis", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Threshold vs Metrics")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric Value")

    # Confusion Matrices
    ax = axes[1, 1]
    ax.text(0.5, 0.5, "Confusion Matrices", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Confusion Matrices at Validated Thresholds")

    plt.tight_layout()
    plt.savefig(save_dir / "comparison_plots.png", dpi=150)
    plt.close()


def plot_confusion_matrices(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    thr_a: float,
    thr_b: float,
    save_dir: Path,
):
    """Plot confusion matrices for both approaches."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, name, scores, thr in zip(axes, ["Approach A", "Approach B"], [scores_a, scores_b], [thr_a, thr_b]):
        pred = scores > thr
        cm = confusion_matrix(y_true, pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name} (thr={thr:.4f})")

    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrices.png", dpi=150)
    plt.close()


def plot_threshold_analysis(
    comparison: dict,
    save_dir: Path,
):
    """Plot threshold vs metrics for both approaches."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics_to_plot = ["recall", "precision", "f1_score", "f2_score"]

    for ax, metric in zip(axes.flat, metrics_to_plot):
        for name, data in comparison.items():
            thresholds = []
            values = []
            for thr_name, m in data["metrics_at_all_thresholds"].items():
                thresholds.append(thr_name)
                values.append(m[metric])
            ax.plot(range(len(thresholds)), values, marker="o", label=name)
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels(thresholds, rotation=45, ha="right")
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Threshold")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "threshold_analysis.png", dpi=150)
    plt.close()


def run_full_comparison(
    approach_a_path: str | Path,
    approach_b_path: str | Path,
    feature_set: str = "plus",
    fpr_cap: float = 0.0001,
    use_dev: bool = False,
    save_plots: bool = True,
) -> dict:
    """Run complete comparison and save results."""
    cfg = load_config()

    # Load test data for probabilities
    path = resolve(cfg["paths"]["dev_parquet" if use_dev else "parquet"])
    df = load_dataset(path)
    s = cfg["split"]
    train, val, test = chronological_split(df, s["train_end_step"], s["val_end_step"])
    Xte = build_features(test, feature_set)
    yte = test["isFraud"].to_numpy()
    ref_prev = train["isFraud"].mean()

    # Load models
    ensemble_a = AdvancedEnsemble.load(approach_a_path)
    model_b = ImbalanceAwareModel.load(approach_b_path)

    prob_a = ensemble_a.predict_proba(Xte)
    prob_b = model_b.predict_proba(Xte)

    # Compare
    comparison = compare_approaches(approach_a_path, approach_b_path, feature_set, fpr_cap, use_dev)

    # Generate table
    table = generate_comparison_table(comparison)

    # Save results
    out_dir = resolve(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    table.to_csv(out_dir / "approach_comparison_table.csv", index=False)
    print("\n=== COMPARISON TABLE ===")
    print(table.to_string(index=False))

    # Save detailed results
    with open(out_dir / "approach_comparison_detailed.json", "w") as f:
        # Convert numpy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(comparison), f, indent=2)

    # Save predictions for further analysis
    pred_df = pd.DataFrame({
        "isFraud": yte,
        "approach_a_score": prob_a,
        "approach_b_score": prob_b,
        "approach_a_pred": (prob_a > ensemble_a.threshold).astype(int),
        "approach_b_pred": (prob_b > model_b.threshold).astype(int),
    })
    pred_df.to_csv(out_dir / "approach_comparison_predictions.csv", index=False)

    # Plot if requested
    if save_plots:
        fig_dir = out_dir / "figures"
        plot_confusion_matrices(yte, prob_a, prob_b, ensemble_a.threshold, model_b.threshold, fig_dir)
        plot_threshold_analysis(comparison, fig_dir)
        print(f"\nPlots saved to {fig_dir}")

    # Print summary
    print("\n=== SUMMARY ===")
    m_a = comparison["Approach A (Advanced Ensemble)"]["metrics_at_validated"]
    m_b = comparison["Approach B (Imbalance-Aware)"]["metrics_at_validated"]

    print(f"Approach A: Recall={m_a['recall']:.4f}, Precision={m_a['precision']:.4f}, "
          f"F1={m_a['f1_score']:.4f}, F2={m_a['f2_score']:.4f}, PR-AUC={m_a['pr_auc']:.4f}, "
          f"MCC={m_a['mcc']:.4f}, FP={m_a['false_positives']}, FN={m_a['false_negatives']}")
    print(f"Approach B: Recall={m_b['recall']:.4f}, Precision={m_b['precision']:.4f}, "
          f"F1={m_b['f1_score']:.4f}, F2={m_b['f2_score']:.4f}, PR-AUC={m_b['pr_auc']:.4f}, "
          f"MCC={m_b['mcc']:.4f}, FP={m_b['false_positives']}, FN={m_b['false_negatives']}")

    # Determine winner per metric
    print("\n=== WINNER PER METRIC ===")
    metrics = ["recall", "precision", "f1_score", "f2_score", "pr_auc", "roc_auc", "mcc"]
    for metric in metrics:
        winner = "A" if m_a[metric] > m_b[metric] else "B"
        print(f"  {metric}: Approach {winner} ({m_a[metric]:.4f} vs {m_b[metric]:.4f})")

    # FP/FN comparison
    print(f"\nFalse Positives: A={m_a['false_positives']}, B={m_b['false_positives']}")
    print(f"False Negatives: A={m_a['false_negatives']}, B={m_b['false_negatives']}")

    return {
        "comparison": comparison,
        "table": table,
        "predictions": pred_df,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Approach A and Approach B")
    parser.add_argument("--approach-a", required=True, help="Path to Approach A model")
    parser.add_argument("--approach-b", required=True, help="Path to Approach B model")
    parser.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    parser.add_argument("--fpr-cap", type=float, default=0.0001)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    run_full_comparison(
        approach_a_path=args.approach_a,
        approach_b_path=args.approach_b,
        feature_set=args.feature_set,
        fpr_cap=args.fpr_cap,
        use_dev=args.dev,
        save_plots=not args.no_plots,
    )