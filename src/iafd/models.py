"""Model zoo (matches the paper's tree/linear models) and soft voting."""
from __future__ import annotations
import numpy as np

ALL_MODELS = ["lr", "dt", "rf", "xgb", "lgbm"]

DEFAULTS = {
    "lr": {"max_iter": 1000},
    "dt": {"max_depth": 12, "min_samples_leaf": 5},
    "rf": {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 5},
    "xgb": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
    "lgbm": {"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.1, "min_child_weight": 1.0, "max_delta_step": 1.0},
}


def build_model(name: str, seed: int = 42, params: dict | None = None):
    if name not in DEFAULTS:
        raise ValueError(f"Unknown model {name!r}; choose from {ALL_MODELS}")
    p = {**DEFAULTS[name], **(params or {})}
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **p))
    if name == "dt":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=seed, **p)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **p)
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
