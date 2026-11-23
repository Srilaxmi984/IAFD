"""SHAP Explanations for Both Approaches.

This module provides comprehensive explainability for both Approach A (Advanced Ensemble)
and Approach B (Imbalance-Aware Model), including:
- Global feature importance (summary plots, bar plots)
- SHAP dependence plots
- Local explanations for individual predictions
- Comparative feature importance analysis
- Explanations for false positives and false negatives
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

# Internal
from .config import load_config, resolve
from .data import load_dataset, chronological_split, check_split
from .features import build_features, FEATURE_SETS
from .approach_a_ensemble import AdvancedEnsemble
from .approach_b_imbalance import ImbalanceAwareModel
from .evaluate import threshold_at_fpr, confusion_at_threshold

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# SHAP Explanation Core
# ============================================================

def create_ensemble_predict_fn(models: dict, columns: list[str]):
    """Create prediction function for ensemble SHAP."""
    def predict_fn(X_arr):
        X_df = pd.DataFrame(X_arr, columns=columns)
        # Average probabilities from all models
        probs = []
        for model in models.values():
            proba = model.predict_proba(X_df)
            prob = proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba
            probs.append(prob)
        return np.mean(np.column_stack(probs), axis=1)
    return predict_fn


def create_model_predict_fn(model, columns: list[str]):
    """Create prediction function for single model SHAP."""
    def predict_fn(X_arr):
        X_df = pd.DataFrame(X_arr, columns=columns)
        proba = model.predict_proba(X_df)
        return proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba
    return predict_fn


def compute_shap_values(
    predict_fn,
    X_explain: pd.DataFrame,
    background: pd.DataFrame,
    max_evals: int = 400,
    algorithm: str = "permutation",
) -> shap.Explanation:
    """Compute SHAP values using permutation explainer."""
    explainer = shap.Explainer(
        predict_fn,
        shap.maskers.Independent(background, max_samples=len(background)),
        algorithm=algorithm,
    )
    return explainer(X_explain.to_numpy(), max_evals=max_evals)


def compute_tree_shap(model, X_explain: pd.DataFrame) -> np.ndarray:
    """Compute exact TreeSHAP values for tree-based models."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    # Handle different return formats
    if isinstance(shap_values, list):
        # For classifiers, shap_values is list of [class_0, class_1]
        return shap_values[1]
    return shap_values


# ============================================================
# Global Explanations
# ============================================================

def generate_global_explanations(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float,
    columns: list[str],
    n_fraud: int = 300,
    n_legit: int = 300,
    background_size: int = 100,
    max_evals: int = 400,
    seed: int = 42,
) -> dict:
    """Generate global SHAP explanations."""
    rng = np.random.default_rng(seed)

    # Sample background
    bg_idx = rng.choice(len(X_train), min(background_size, len(X_train)), replace=False)
    background = X_train.iloc[bg_idx]

    # Select rows to explain
    fraud_idx = np.flatnonzero(y_test == 1)
    legit_idx = np.flatnonzero(y_test == 0)

    explain_fraud = rng.choice(fraud_idx, min(n_fraud, len(fraud_idx)), replace=False)
    explain_legit = rng.choice(legit_idx, min(n_legit, len(legit_idx)), replace=False)
    explain_idx = np.unique(np.concatenate([explain_fraud, explain_legit]))

    X_explain = X_test.iloc[explain_idx]
    y_explain = y_test[explain_idx]

    # Create predict function
    if hasattr(model, 'models'):  # Ensemble
        predict_fn = create_ensemble_predict_fn(model.models, columns)
    else:  # Single model
        predict_fn = create_model_predict_fn(model, columns)

    # Compute SHAP
    print(f"  Computing SHAP for {model_name} on {len(explain_idx)} rows...")
    shap_exp = compute_shap_values(predict_fn, X_explain, background, max_evals=max_evals)

    # Mean absolute SHAP by class
    fraud_mask = y_explain == 1
    legit_mask = y_explain == 0

    mean_abs_fraud = np.abs(shap_exp.values[fraud_mask]).mean(axis=0)
    mean_abs_legit = np.abs(shap_exp.values[legit_mask]).mean(axis=0)

    # Feature importance ranking
    importance_fraud = pd.DataFrame({
        "feature": columns,
        "mean_abs_shap": mean_abs_fraud,
    }).sort_values("mean_abs_shap", ascending=False)

    importance_legit = pd.DataFrame({
        "feature": columns,
        "mean_abs_shap": mean_abs_legit,
    }).sort_values("mean_abs_shap", ascending=False)

    return {
        "shap_explanation": shap_exp,
        "X_explain": X_explain,
        "y_explain": y_explain,
        "importance_fraud": importance_fraud,
        "importance_legit": importance_legit,
        "mean_abs_fraud": mean_abs_fraud,
        "mean_abs_legit": mean_abs_legit,
    }


def plot_global_importance(
    importance_df: pd.DataFrame,
    title: str,
    save_path: Path,
    top_n: int = 15,
):
    """Plot global feature importance bar chart."""
    top = importance_df.head(top_n).iloc[::-1]  # Reverse for horizontal bar

    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#1f77b4")
    plt.xlabel("Mean |SHAP Value|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_summary(
    shap_exp: shap.Explanation,
    X_explain: pd.DataFrame,
    title: str,
    save_path: Path,
    max_display: int = 15,
):
    """Plot SHAP beeswarm summary plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_exp.values,
        X_explain,
        feature_names=list(X_explain.columns),
        show=False,
        max_display=max_display,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dependence_plots(
    shap_exp: shap.Explanation,
    X_explain: pd.DataFrame,
    top_features: list[str],
    save_dir: Path,
):
    """Plot SHAP dependence plots for top features."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for feature in top_features:
        if feature not in X_explain.columns:
            continue

        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature,
            shap_exp.values,
            X_explain,
            feature_names=list(X_explain.columns),
            show=False,
        )
        plt.title(f"SHAP Dependence: {feature}")
        plt.tight_layout()
        plt.savefig(save_dir / f"dependence_{feature}.png", dpi=150, bbox_inches="tight")
        plt.close()


# ============================================================
# Local Explanations
# ============================================================

def find_explanation_cases(
    y_test: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    seed: int = 42,
) -> dict[str, int]:
    """Find interesting cases for local explanation."""
    rng = np.random.default_rng(seed)
    fraud_idx = np.flatnonzero(y_test == 1)
    legit_idx = np.flatnonzero(y_test == 0)

    # Caught fraud (score > threshold)
    caught_fraud = fraud_idx[scores[fraud_idx] > threshold]
    # Missed fraud (score <= threshold)
    missed_fraud = fraud_idx[scores[fraud_idx] <= threshold]
    # False positives (legit with score > threshold)
    false_positives = legit_idx[scores[legit_idx] > threshold]

    cases = {}

    if len(caught_fraud) > 0:
        # Median-scoring caught fraud
        caught_scores = scores[caught_fraud]
        cases["caught_fraud"] = int(caught_fraud[np.argsort(caught_scores)[len(caught_scores) // 2]])

    if len(missed_fraud) > 0:
        # Lowest-scoring missed fraud
        cases["missed_fraud"] = int(missed_fraud[np.argmin(scores[missed_fraud])])

    if len(false_positives) > 0:
        # Highest-scoring false positive
        cases["false_positive"] = int(false_positives[np.argmax(scores[false_positives])])

    return cases


def generate_local_explanations(
    model,
    model_name: str,
    shap_exp: shap.Explanation,
    X_explain: pd.DataFrame,
    y_explain: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    cases: dict[str, int],
    columns: list[str],
    save_dir: Path,
) -> dict:
    """Generate local waterfall explanations for specific cases."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Map explanation index to original test index
    idx_to_pos = {int(idx): i for i, idx in enumerate(X_explain.index)}

    local_results = {}

    for case_name, test_idx in cases.items():
        if test_idx not in idx_to_pos:
            continue

        pos = idx_to_pos[test_idx]
        case_score = scores[test_idx]
        case_flagged = case_score > threshold
        case_label = int(y_explain[pos])

        # Create SHAP Explanation object for waterfall
        expl = shap.Explanation(
            values=shap_exp.values[pos],
            base_values=shap_exp.base_values[pos],
            data=X_explain.iloc[pos].to_numpy(),
            feature_names=columns,
        )

        # Waterfall plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(expl, max_display=10, show=False)
        plt.title(f"{model_name} | {case_name.replace('_', ' ').title()} | "
                  f"Score: {case_score:.4f} | {'Flagged' if case_flagged else 'Not Flagged'} | "
                  f"True: {'Fraud' if case_label else 'Legit'}")
        plt.tight_layout()
        plt.savefig(save_dir / f"waterfall_{model_name}_{case_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Top drivers
        top_drivers = pd.Series(shap_exp.values[pos], index=columns).sort_values(key=np.abs, ascending=False).head(5)
        local_results[case_name] = {
            "test_index": test_idx,
            "score": float(case_score),
            "flagged": bool(case_flagged),
            "true_label": case_label,
            "top_drivers": top_drivers.to_dict(),
        }

        print(f"  {case_name}: score={case_score:.4f} ({'flagged' if case_flagged else 'not flagged'})")
        for feat, val in top_drivers.items():
            print(f"    {feat}: {val:+.3f}")

    return local_results


# ============================================================
# Comparative Analysis
# ============================================================

def compare_feature_importance(
    importance_a: pd.DataFrame,
    importance_b: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Compare feature importance between two approaches."""
    df_a = importance_a.set_index("feature")["mean_abs_shap"]
    df_b = importance_b.set_index("feature")["mean_abs_shap"]

    comparison = pd.DataFrame({
        "feature": columns,
        "approach_a_importance": df_a.reindex(columns, fill_value=0).values,
        "approach_b_importance": df_b.reindex(columns, fill_value=0).values,
    })

    # Normalize to sum to 1 for fair comparison
    comparison["approach_a_norm"] = comparison["approach_a_importance"] / comparison["approach_a_importance"].sum()
    comparison["approach_b_norm"] = comparison["approach_b_importance"] / comparison["approach_b_importance"].sum()
    comparison["difference"] = comparison["approach_a_norm"] - comparison["approach_b_norm"]
    comparison["abs_difference"] = comparison["difference"].abs()

    return comparison.sort_values("abs_difference", ascending=False)


def plot_importance_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path,
    top_n: int = 15,
):
    """Plot comparative feature importance."""
    top = comparison_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top))
    width = 0.35

    plt.barh(y_pos - width/2, top["approach_a_norm"], width, label="Approach A (Ensemble)", color="#1f77b4")
    plt.barh(y_pos + width/2, top["approach_b_norm"], width, label="Approach B (Imbalance-Aware)", color="#ff7f0e")

    plt.yticks(y_pos, top["feature"])
    plt.xlabel("Normalized Mean |SHAP|")
    plt.title("Feature Importance Comparison: Approach A vs Approach B")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Complete SHAP Pipeline
# ============================================================

def run_shap_analysis(
    approach_a_path: str | Path,
    approach_b_path: str | Path,
    feature_set: str = "plus",
    fpr_cap: float = 0.0001,
    n_fraud: int = 300,
    n_legit: int = 300,
    background_size: int = 100,
    max_evals: int = 400,
    use_dev: bool = False,
    seed: int = 42,
) -> dict:
    """Run complete SHAP analysis for both approaches."""
    cfg = load_config()

    # Load data
    path = resolve(cfg["paths"]["dev_parquet" if use_dev else "parquet"])
    df = load_dataset(path)
    s = cfg["split"]
    train, val, test = chronological_split(df, s["train_end_step"], s["val_end_step"])

    # Build features
    Xtr, Xva, Xte = (build_features(d, feature_set) for d in (train, val, test))
    ytr, yva, yte = (d["isFraud"].to_numpy() for d in (train, val, test))
    columns = list(Xtr.columns)

    # Load models
    print("Loading models...")
    model_a = AdvancedEnsemble.load(approach_a_path)
    model_b = ImbalanceAwareModel.load(approach_b_path)

    # Get test scores
    scores_a = model_a.predict_proba(Xte)
    scores_b = model_b.predict_proba(Xte)

    # Find cases
    cases_a = find_explanation_cases(yte, scores_a, model_a.threshold, seed)
    cases_b = find_explanation_cases(yte, scores_b, model_b.threshold, seed)

    print(f"Approach A cases: {cases_a}")
    print(f"Approach B cases: {cases_b}")

    # Global explanations
    print("\n=== Global SHAP: Approach A ===")
    global_a = generate_global_explanations(
        model_a, "Approach A", Xtr, Xte, yte, model_a.threshold, columns,
        n_fraud=n_fraud, n_legit=n_legit, background_size=background_size,
        max_evals=max_evals, seed=seed,
    )

    print("\n=== Global SHAP: Approach B ===")
    global_b = generate_global_explanations(
        model_b, "Approach B", Xtr, Xte, yte, model_b.threshold, columns,
        n_fraud=n_fraud, n_legit=n_legit, background_size=background_size,
        max_evals=max_evals, seed=seed,
    )

    # Comparative analysis
    print("\n=== Comparative Feature Importance ===")
    comparison = compare_feature_importance(
        global_a["importance_fraud"],
        global_b["importance_fraud"],
        columns,
    )

    # Save results
    out_dir = resolve(cfg["paths"]["reports_dir"])
    fig_dir = out_dir / "figures" / "shap"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Global importance plots
    plot_global_importance(
        global_a["importance_fraud"],
        "Approach A: Feature Importance on Fraud Rows",
        fig_dir / "global_importance_approach_a_fraud.png",
    )
    plot_global_importance(
        global_a["importance_legit"],
        "Approach A: Feature Importance on Legit Rows",
        fig_dir / "global_importance_approach_a_legit.png",
    )
    plot_global_importance(
        global_b["importance_fraud"],
        "Approach B: Feature Importance on Fraud Rows",
        fig_dir / "global_importance_approach_b_fraud.png",
    )
    plot_global_importance(
        global_b["importance_legit"],
        "Approach B: Feature Importance on Legit Rows",
        fig_dir / "global_importance_approach_b_legit.png",
    )

    # SHAP summary plots (beeswarm)
    plot_shap_summary(
        global_a["shap_explanation"],
        global_a["X_explain"],
        "Approach A: SHAP Summary (Fraud + Legit)",
        fig_dir / "shap_summary_approach_a.png",
    )
    plot_shap_summary(
        global_b["shap_explanation"],
        global_b["X_explain"],
        "Approach B: SHAP Summary (Fraud + Legit)",
        fig_dir / "shap_summary_approach_b.png",
    )

    # Dependence plots for top features
    top_features_a = global_a["importance_fraud"].head(5)["feature"].tolist()
    top_features_b = global_b["importance_fraud"].head(5)["feature"].tolist()
    all_top = list(set(top_features_a + top_features_b))

    print(f"\nGenerating dependence plots for: {all_top}")
    plot_dependence_plots(
        global_a["shap_explanation"],
        global_a["X_explain"],
        all_top,
        fig_dir / "dependence_approach_a",
    )
    plot_dependence_plots(
        global_b["shap_explanation"],
        global_b["X_explain"],
        all_top,
        fig_dir / "dependence_approach_b",
    )

    # Local explanations
    print("\n=== Local Explanations: Approach A ===")
    local_a = generate_local_explanations(
        model_a, "Approach_A", global_a["shap_explanation"],
        global_a["X_explain"], global_a["y_explain"],
        scores_a, model_a.threshold, cases_a, columns, fig_dir,
    )

    print("\n=== Local Explanations: Approach B ===")
    local_b = generate_local_explanations(
        model_b, "Approach_B", global_b["shap_explanation"],
        global_b["X_explain"], global_b["y_explain"],
        scores_b, model_b.threshold, cases_b, columns, fig_dir,
    )

    # Comparative importance plot
    plot_importance_comparison(comparison, fig_dir / "importance_comparison.png")

    # Save importance DataFrames
    global_a["importance_fraud"].to_csv(out_dir / "shap_importance_approach_a_fraud.csv", index=False)
    global_a["importance_legit"].to_csv(out_dir / "shap_importance_approach_a_legit.csv", index=False)
    global_b["importance_fraud"].to_csv(out_dir / "shap_importance_approach_b_fraud.csv", index=False)
    global_b["importance_legit"].to_csv(out_dir / "shap_importance_approach_b_legit.csv", index=False)
    comparison.to_csv(out_dir / "shap_importance_comparison.csv", index=False)

    # Save local explanations
    import json
    with open(out_dir / "shap_local_explanations.json", "w") as f:
        json.dump({
            "Approach_A": local_a,
            "Approach_B": local_b,
        }, f, indent=2, default=str)

    print(f"\nSHAP analysis complete. Results saved to {out_dir} and {fig_dir}")

    return {
        "global_a": global_a,
        "global_b": global_b,
        "comparison": comparison,
        "local_a": local_a,
        "local_b": local_b,
        "cases_a": cases_a,
        "cases_b": cases_b,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SHAP explanations for both approaches")
    parser.add_argument("--approach-a", required=True, help="Path to Approach A model")
    parser.add_argument("--approach-b", required=True, help="Path to Approach B model")
    parser.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    parser.add_argument("--fpr-cap", type=float, default=0.0001)
    parser.add_argument("--n-fraud", type=int, default=300)
    parser.add_argument("--n-legit", type=int, default=300)
    parser.add_argument("--background", type=int, default=100)
    parser.add_argument("--max-evals", type=int, default=400)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_shap_analysis(
        approach_a_path=args.approach_a,
        approach_b_path=args.approach_b,
        feature_set=args.feature_set,
        fpr_cap=args.fpr_cap,
        n_fraud=args.n_fraud,
        n_legit=args.n_legit,
        background_size=args.background,
        max_evals=args.max_evals,
        use_dev=args.dev,
        seed=args.seed,
    )