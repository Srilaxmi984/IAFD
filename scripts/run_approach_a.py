"""Train Approach A: Advanced Optimized Ensemble.

This script trains the advanced ensemble model with multiple architectures,
hyperparameter optimization, calibration, and threshold optimization.

Usage:
    python scripts/run_approach_a.py                              # Default: stacking ensemble
    python scripts/run_approach_a.py --architecture weighted_voting
    python scripts/run_approach_a.py --architecture soft_voting
    python scripts/run_approach_a.py --models rf xgb lgbm et catboost
    python scripts/run_approach_a.py --optimize-hyperparams --n-trials 50
    python scripts/run_approach_a.py --pos-weight 10 --fpr-cap 0.0001
    python scripts/run_approach_a.py --dev                        # Quick run on dev sample
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
from iafd.approach_a_ensemble import train_approach_a, ENSEMBLE_BASE_MODELS


def main():
    cfg = load_config()

    ap = argparse.ArgumentParser(description="Train Approach A: Advanced Ensemble")
    ap.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=["rf", "xgb", "lgbm", "et"],
                    choices=list(ENSEMBLE_BASE_MODELS.keys()))
    ap.add_argument("--architecture", default="stacking",
                    choices=["soft_voting", "weighted_voting", "stacking"])
    ap.add_argument("--calibrate", action="store_true", default=True)
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    ap.add_argument("--optimize-weights", action="store_true", default=True)
    ap.add_argument("--pos-weight", type=float, default=1.0,
                    help="Class weight for fraud class (1.0 = no weighting)")
    ap.add_argument("--optimize-hyperparams", action="store_true",
                    help="Run Optuna hyperparameter optimization")
    ap.add_argument("--n-trials", type=int, default=30,
                    help="Number of Optuna trials per model")
    ap.add_argument("--fpr-cap", type=float, default=0.0001,
                    help="FPR cap for threshold selection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev", action="store_true", help="Use dev sample (NOT reportable)")

    args = ap.parse_args()

    # Validate models
    for m in args.models:
        if m not in ENSEMBLE_BASE_MODELS:
            sys.exit(f"Unknown model: {m}. Available: {list(ENSEMBLE_BASE_MODELS.keys())}")

    print(f"=== Approach A: Advanced Ensemble ===")
    print(f"Feature set: {args.feature_set}")
    print(f"Models: {args.models}")
    print(f"Architecture: {args.architecture}")
    print(f"Calibrate: {args.calibrate}")
    print(f"Optimize weights: {args.optimize_weights}")
    print(f"Pos weight: {args.pos_weight}")
    print(f"Optimize hyperparams: {args.optimize_hyperparams}")
    print(f"FPR cap: {args.fpr_cap:.4%}")
    print(f"Seed: {args.seed}")
    print(f"Dev mode: {args.dev}")
    print()

    t0 = time.time()
    ensemble, metadata = train_approach_a(
        feature_set=args.feature_set,
        base_models=args.models,
        architecture=args.architecture,
        calibrate=args.calibrate,
        optimize_weights=args.optimize_weights,
        pos_weight=args.pos_weight,
        optimize_hyperparams=args.optimize_hyperparams,
        n_trials=args.n_trials,
        seed=args.seed,
        fpr_cap=args.fpr_cap,
        use_dev=args.dev,
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Model saved to: {metadata['model_path']}")


if __name__ == "__main__":
    main()