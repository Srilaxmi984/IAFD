"""Model zoo (matches the paper's tree/linear models) and soft voting."""
from __future__ import annotations
import numpy as np

ALL_MODELS = ["lr", "dt", "rf", "xgb", "lgbm"]


def _signed_log1p(x):
    """sign(x) * log1p(|x|): tames the heavy-tailed money columns (up to ~1e7) so the LR sigmoid does not saturate."""
    return np.sign(x) * np.log1p(np.abs(x))

DEFAULTS = {
    "lr": {"max_iter": 2000, "tol": 1e-8},   # tight tol: the default stopped after ~12 iterations on 3M rows
    "dt": {"max_depth": 12, "min_samples_leaf": 5},
    "rf": {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 5},
    "xgb": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
    "lgbm": {"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.1, "min_child_weight": 1.0, "max_delta_step": 1.0},
}


def build_model(name: str, seed: int = 42, params: dict | None = None, pos_weight: float = 1.0):
    if name not in DEFAULTS:
        raise ValueError(f"Unknown model {name!r}; choose from {ALL_MODELS}")
    if pos_weight <= 0:
        raise ValueError("pos_weight must be > 0")
    p = {**DEFAULTS[name], **(params or {})}
    cw = {0: 1.0, 1: float(pos_weight)} if pos_weight != 1.0 else None   # cost-sensitive learning (experiment B)
    if name in ("xgb", "lgbm") and pos_weight != 1.0:
        p["scale_pos_weight"] = float(pos_weight)
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler
        # LR only: signed log, then standardise (the paper standardises only). Trees are scale-free and use raw values.
        return make_pipeline(FunctionTransformer(_signed_log1p), StandardScaler(),
                             LogisticRegression(random_state=seed, class_weight=cw, **p))
    if name == "dt":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=seed, class_weight=cw, **p)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight=cw, **p)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, tree_method="hist", n_jobs=-1, eval_metric="logloss", **p)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **p)
    raise AssertionError("unreachable")


def soft_vote(prob_list: list[np.ndarray]) -> np.ndarray:
    """Average predicted fraud probabilities (soft voting)."""
    if not prob_list:
        raise ValueError("Nothing to vote on")
    return np.mean(np.vstack(prob_list), axis=0)
