"""Run Robustness Testing on Both Approaches.

This script evaluates both models against realistic fraud evasion patterns.

Usage:
    python scripts/run_robustness.py \
        --approach-a models/approach_a_stacking_plus.joblib \
        --approach-b models/approach_b_balanced_rf_plus.joblib
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iafd.config import load_config, resolve
from iafd.approach_a_ensemble import AdvancedEnsemble
from iafd.approach_b_imbalance import ImbalanceAwareModel
from iafd.robustness import run_robustness_evaluation, generate_robustness_report


def main():
    ap = argparse.ArgumentParser(description="Run robustness testing on both approaches")
    ap.add_argument("--approach-a", default="models/approach_a_stacking_plus.joblib", help="Path to Approach A model")
    ap.add_argument("--approach-b", default="models/approach_b_balanced_rf_plus.joblib", help="Path to Approach B model")
    ap.add_argument("--feature-set", default="plus", choices=["paper", "engineered", "plus", "velocity", "no_dest"])
    ap.add_argument("--fpr-cap", type=float, default=0.0001)
    ap.add_argument("--dev", action="store_true", help="Use dev sample (NOT reportable)")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Verify model files exist
    for path_str in [args.approach_a, args.approach_b]:
        if not Path(path_str).exists():
            sys.exit(f"Model not found: {path_str}")

    # Load models
    print("Loading models...")
    model_a = AdvancedEnsemble.load(args.approach_a)
    model_b = ImbalanceAwareModel.load(args.approach_b)

    # Run evaluation
    print("Running robustness evaluation...")
    results = run_robustness_evaluation(
        model_a, model_b,
        feature_set=args.feature_set,
        fpr_cap=args.fpr_cap,
        use_dev=args.dev,
        seed=args.seed,
    )

    # Save results
    cfg = load_config()
    out_dir = resolve(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_robustness_report(results, out_dir)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()