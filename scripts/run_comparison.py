"""Compare Approach A and Approach B.

This script runs a comprehensive comparison between both approaches
on identical test data with identical feature sets and temporal splits.

Usage:
    python scripts/run_comparison.py \
        --approach-a models/approach_a_stacking_plus.joblib \
        --approach-b models/approach_b_balanced_rf_plus.joblib
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iafd.compare_approaches import run_full_comparison


def main():
    ap = argparse.ArgumentParser(description="Compare Approach A and Approach B")
    ap.add_argument("--approach-a", required=True, help="Path to Approach A model")
    ap.add_argument("--approach-b", required=True, help="Path to Approach B model")
    ap.add_argument("--feature-set", default="plus", choices=["paper", "engineered", "plus", "velocity", "no_dest"])
    ap.add_argument("--fpr-cap", type=float, default=0.0001)
    ap.add_argument("--dev", action="store_true", help="Use dev sample (NOT reportable)")
    ap.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = ap.parse_args()

    # Verify model files exist
    for path_str in [args.approach_a, args.approach_b]:
        if not Path(path_str).exists():
            sys.exit(f"Model not found: {path_str}")

    run_full_comparison(
        approach_a_path=args.approach_a,
        approach_b_path=args.approach_b,
        feature_set=args.feature_set,
        fpr_cap=args.fpr_cap,
        use_dev=args.dev,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()