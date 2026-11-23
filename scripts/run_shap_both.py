"""Run SHAP Explanations for Both Approaches.

This script generates comprehensive SHAP explanations for both models.

Usage:
    python scripts/run_shap_both.py \
        --approach-a models/approach_a_stacking_plus.joblib \
        --approach-b models/approach_b_balanced_rf_plus.joblib
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iafd.shap_explain import run_shap_analysis


def main():
    ap = argparse.ArgumentParser(description="Run SHAP explanations for both approaches")
    ap.add_argument("--approach-a", default="models/approach_a_stacking_plus.joblib", help="Path to Approach A model")
    ap.add_argument("--approach-b", default="models/approach_b_balanced_rf_plus.joblib", help="Path to Approach B model")
    ap.add_argument("--feature-set", default="plus", choices=["paper", "engineered", "plus", "velocity", "no_dest"])
    ap.add_argument("--fpr-cap", type=float, default=0.0001)
    ap.add_argument("--n-fraud", type=int, default=300)
    ap.add_argument("--n-legit", type=int, default=300)
    ap.add_argument("--background", type=int, default=100)
    ap.add_argument("--max-evals", type=int, default=400)
    ap.add_argument("--dev", action="store_true", help="Use dev sample (NOT reportable)")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Verify model files exist
    for path_str in [args.approach_a, args.approach_b]:
        if not Path(path_str).exists():
            sys.exit(f"Model not found: {path_str}")

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


if __name__ == "__main__":
    main()