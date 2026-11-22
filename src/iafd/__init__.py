"""IAFD: Imbalance-Aware Fraud Detection on PaySim.

This package provides:
- Data loading and chronological splitting for PaySim
- Feature engineering for fraud detection
- Baseline models (LR, DT, RF, XGB, LightGBM)
- Approach A: Advanced Optimized Ensemble (stacking, weighted voting, calibration)
- Approach B: Specialized Imbalance-Aware Models (EasyEnsemble, BalancedRF, RUSBoost, cost-sensitive GBM)
- Evaluation under FPR caps
- SHAP explanations
- Robustness testing against fraud evasion patterns
"""
__version__ = "0.2.0"

# Core modules
from . import config
from . import data
from . import features
from . import models
from . import evaluate
from . import explain
from . import velocity
from . import structuring
from . import hardneg

# New approach modules
from . import approach_a_ensemble
from . import approach_b_imbalance
from . import compare_approaches
from . import robustness
from . import shap_explain

__all__ = [
    "config",
    "data",
    "features",
    "models",
    "evaluate",
    "explain",
    "velocity",
    "structuring",
    "hardneg",
    "approach_a_ensemble",
    "approach_b_imbalance",
    "compare_approaches",
    "robustness",
    "shap_explain",
]
