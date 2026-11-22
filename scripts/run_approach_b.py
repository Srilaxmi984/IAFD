"""Train Approach B: Specialized Imbalance-Aware Model.

This script trains models specifically designed for highly imbalanced fraud detection,
including EasyEnsemble, BalancedRandomForest, RUSBoost, BalancedBagging,
cost-sensitive gradient boosting, and various resampling techniques.

Usage:
    python scripts/run_approach_b.py                              # Default: balanced_rf
    python scripts/run_approach_b.py --model cost_xgb --pos-weight 50
    python scripts/run_approach_b.py --model easy_ensemble
    python scripts/run_approach_b.py --model balanced_rf
    python scripts/run_approach_b.py --model smote_xgb --resampling-strategy 0.5
    python scripts/run_approach_b.py --auto-select                # Auto-select best model
    python scripts/run_approach_b.py --dev                        # Quick run on dev sample
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
from iafd.approach_b_imbalance import train_approach_b, IMBALANCE_MODELS


def main():
    cfg = load_config()

    ap = argparse.ArgumentParser(description="Train Approach B: Imbalance-Aware Model")
    ap.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    ap.add_argument("--model", default="balanced_rf", choices=list(IMBALANCE_MODELS.keys()))
    ap.add_argument("--calibrate", action="store_true", default=True)
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    ap.add_argument("--pos-weight", type=float, default=1.0,
                    help="Class weight for fraud class (for cost-sensitive models)")
    ap.add_argument("--resampling-strategy", type=float, default=0.5,
                    help="Sampling strategy for resampling-based models (0.5 = 1:2 fraud:legit)")
    ap.add_argument("--auto-select", action="store_true",
                    help="Automatically select best model and hyperparameters on validation")
    ap.add_argument("--fpr-cap", type=float, default=0.0001,
                    help="FPR cap for threshold selection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev", action="store_true", help="Use dev sample (NOT reportable)")

    args = ap.parse_args()

    print(f"=== Approach B: Imbalance-Aware Model ===")
    print(f"Feature set: {args.feature_set}")
    print(f"Model: {args.model}")
    print(f"Calibrate: {args.calibrate}")
    print(f"Pos weight: {args.pos_weight}")
    print(f"Resampling strategy: {args.resampling_strategy}")
    print(f"Auto-select: {args.auto_select}")
    print(f"FPR cap: {args.fpr_cap:.4%}")
    print(f"Seed: {args.seed}")
    print(f"Dev mode: {args.dev}")
    print()

    t0 = time.time()
    model, metadata = train_approach_b(
        feature_set=args.feature_set,
        model_name=args.model,
        calibrate=args.calibrate,
        pos_weight=args.pos_weight,
        resampling_strategy=args.resampling_strategy,
        auto_select=args.auto_select,
        fpr_cap=args.fpr_cap,
        seed=args.seed,
        use_dev=args.dev,
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Model saved to: {metadata['model_path']}")


if __name__ == "__main__":
    main()