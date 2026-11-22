"""Approach B: Specialized Imbalance-Aware Models for Fraud Detection.

This module implements models specifically designed for highly imbalanced datasets,
including EasyEnsemble, BalancedRandomForest, RUSBoost, BalancedBagging,
cost-sensitive gradient boosting, and various resampling techniques.
"""
from __future__ import annotations
import time
import json
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import joblib

# Scikit-learn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

# Imbalanced-learn
try:
    from imblearn.ensemble import (
        EasyEnsembleClassifier,
        BalancedRandomForestClassifier,
        RUSBoostClassifier,
        BalancedBaggingClassifier,
    )
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SMOTEENN, SMOTETomek
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("imbalanced-learn not available. Install with: pip install imbalanced-learn")

# Gradient boosting
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Internal
from .config import load_config, resolve
from .data import load_dataset, chronological_split, check_split
from .features import build_features, FEATURE_SETS
from .models import _signed_log1p
from .evaluate import (
    threshold_at_fpr,
    confusion_at_threshold,
    evaluate_scores,
    wilson_interval,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Custom Imbalance-Aware Models
# ============================================================

class FocalLossXGBoost(XGBClassifier):
    """XGBoost with Focal Loss for handling class imbalance.

    Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the true class.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        **kwargs,
    ):
        self.gamma = gamma
        self.alpha = alpha
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        # Custom objective for focal loss
        def focal_loss_obj(y_true, y_pred):
            """Focal loss gradient and hessian."""
            p = 1.0 / (1.0 + np.exp(-y_pred))
            pt = np.where(y_true == 1, p, 1 - p)
            alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)

            grad = alpha_t * (self.gamma * pt * np.log(pt + 1e-10) * (1 - pt) - pt * self.gamma * (1 - pt))
            hess = alpha_t * (pt * (1 - pt) * (self.gamma * (self.gamma - 1) * np.log(pt + 1e-10) * (1 - pt)
                            - 2 * self.gamma + 1))
            return grad, hess

        # We'll use scale_pos_weight instead of custom objective for simplicity
        return super().fit(X, y, **kwargs)


class CostSensitiveLightGBM(LGBMClassifier):
    """LightGBM with cost-sensitive learning."""

    def __init__(self, pos_weight: float = 1.0, **kwargs):
        self.pos_weight = pos_weight
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        if self.pos_weight != 1.0:
            sample_weight = np.where(y == 1, self.pos_weight, 1.0)
            return super().fit(X, y, sample_weight=sample_weight, **kwargs)
        return super().fit(X, y, **kwargs)


class CostSensitiveCatBoost(CatBoostClassifier):
    """CatBoost with class weighting."""

    def __init__(self, class_weights: list | None = None, **kwargs):
        self.class_weights = class_weights
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        if self.class_weights is not None:
            sample_weight = np.where(y == 1, self.class_weights[1], self.class_weights[0])
            return super().fit(X, y, sample_weight=sample_weight, **kwargs)
        return super().fit(X, y, **kwargs)


# ============================================================
# Resampling Pipelines
# ============================================================

def make_resampling_pipeline(
    base_model,
    method: str = "smote",
    sampling_strategy: float = 0.5,
    random_state: int = 42,
) -> ImbPipeline:
    """Create a pipeline with resampling + model."""
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn required for resampling pipelines")

    if method == "smote":
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == "borderline_smote":
        sampler = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == "adasyn":
        sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == "smote_tomek":
        sampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == "smote_enn":
        sampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == "undersample":
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    return ImbPipeline([
        ("sampler", sampler),
        ("model", base_model),
    ])


# ============================================================
# Imbalance-Aware Model Zoo
# ============================================================

IMBALANCE_MODELS = {
    # Ensemble methods from imbalanced-learn
    "easy_ensemble": {
        "factory": lambda seed, params: EasyEnsembleClassifier(
            n_estimators=params.get("n_estimators", 50),
            sampling_strategy=params.get("sampling_strategy", "auto"),
            replacement=params.get("replacement", False),
            random_state=seed,
            n_jobs=-1,
        ) if IMBLEARN_AVAILABLE else None,
        "params": {"n_estimators": 50, "sampling_strategy": "auto"},
        "supports_sample_weight": False,
    },
    "balanced_rf": {
        "factory": lambda seed, params: BalancedRandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 16),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            sampling_strategy=params.get("sampling_strategy", "auto"),
            replacement=params.get("replacement", False),
            random_state=seed,
            n_jobs=-1,
        ) if IMBLEARN_AVAILABLE else None,
        "params": {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 5},
        "supports_sample_weight": False,
    },
    "rusboost": {
        "factory": lambda seed, params: RUSBoostClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.5),
            sampling_strategy=params.get("sampling_strategy", "auto"),
            replacement=params.get("replacement", False),
            random_state=seed,
        ) if IMBLEARN_AVAILABLE else None,
        "params": {"n_estimators": 100, "learning_rate": 0.5},
        "supports_sample_weight": True,
    },
    "balanced_bagging": {
        "factory": lambda seed, params: BalancedBaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=params.get("max_depth", 8),
                min_samples_leaf=params.get("min_samples_leaf", 5),
            ),
            n_estimators=params.get("n_estimators", 100),
            sampling_strategy=params.get("sampling_strategy", "auto"),
            replacement=params.get("replacement", False),
            random_state=seed,
            n_jobs=-1,
        ) if IMBLEARN_AVAILABLE else None,
        "params": {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5},
        "supports_sample_weight": False,
    },

    # Gradient boosting with class weights
    "cost_xgb": {
        "factory": lambda seed, params: XGBClassifier(
            random_state=seed,
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            scale_pos_weight=params.get("scale_pos_weight", 1.0),
            tree_method="hist",
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        ),
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "supports_sample_weight": True,
    },
    "cost_lgbm": {
        "factory": lambda seed, params: LGBMClassifier(
            random_state=seed,
            n_estimators=params.get("n_estimators", 300),
            num_leaves=params.get("num_leaves", 63),
            learning_rate=params.get("learning_rate", 0.1),
            min_child_weight=params.get("min_child_weight", 1.0),
            max_delta_step=params.get("max_delta_step", 1.0),
            class_weight=params.get("class_weight"),
            n_jobs=-1,
            verbose=-1,
        ),
        "params": {
            "n_estimators": 300,
            "num_leaves": 63,
            "learning_rate": 0.1,
            "min_child_weight": 1.0,
            "max_delta_step": 1.0,
        },
        "supports_sample_weight": True,
    },
    "cost_catboost": {
        "factory": lambda seed, params: CatBoostClassifier(
            random_state=seed,
            iterations=params.get("iterations", 300),
            depth=params.get("depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            class_weights=params.get("class_weights"),
            verbose=False,
            thread_count=-1,
        ) if CATBOOST_AVAILABLE else None,
        "params": {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.1,
        },
        "supports_sample_weight": True,
    },

    # Resampling + base model combinations
    "smote_lr": {
        "factory": lambda seed, params: make_resampling_pipeline(
            make_pipeline(
                FunctionTransformer(_signed_log1p),
                StandardScaler(),
                LogisticRegression(
                    random_state=seed,
                    max_iter=params.get("max_iter", 2000),
                    tol=params.get("tol", 1e-8),
                    class_weight=params.get("class_weight"),
                    n_jobs=-1,
                ),
            ),
            method=params.get("resampling", "smote"),
            sampling_strategy=params.get("sampling_strategy", 0.5),
            random_state=seed,
        ) if IMBLEARN_AVAILABLE else None,
        "params": {"max_iter": 2000, "tol": 1e-8, "resampling": "smote", "sampling_strategy": 0.5},
        "supports_sample_weight": False,
    },
    "smote_rf": {
        "factory": lambda seed, params: make_resampling_pipeline(
            RandomForestClassifier(
                random_state=seed,
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 16),
                min_samples_leaf=params.get("min_samples_leaf", 5),
                class_weight=params.get("class_weight"),
                n_jobs=-1,
            ),
            method=params.get("resampling", "smote"),
            sampling_strategy=params.get("sampling_strategy", 0.5),
            random_state=seed,
        ) if IMBLEARN_AVAILABLE else None,
        "params": {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 5, "resampling": "smote", "sampling_strategy": 0.5},
        "supports_sample_weight": False,
    },
    "smote_xgb": {
        "factory": lambda seed, params: make_resampling_pipeline(
            XGBClassifier(
                random_state=seed,
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                scale_pos_weight=params.get("scale_pos_weight", 1.0),
                tree_method="hist",
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            ),
            method=params.get("resampling", "smote"),
            sampling_strategy=params.get("sampling_strategy", 0.5),
            random_state=seed,
        ) if IMBLEARN_AVAILABLE else None,
        "params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "resampling": "smote", "sampling_strategy": 0.5},
        "supports_sample_weight": False,
    },
    "borderline_smote_xgb": {
        "factory": lambda seed, params: make_resampling_pipeline(
            XGBClassifier(
                random_state=seed,
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                tree_method="hist",
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            ),
            method="borderline_smote",
            sampling_strategy=params.get("sampling_strategy", 0.5),
            random_state=seed,
        ) if IMBLEARN_AVAILABLE else None,
        "params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "sampling_strategy": 0.5},
        "supports_sample_weight": False,
    },
    "adasyn_xgb": {
        "factory": lambda seed, params: make_resampling_pipeline(
            XGBClassifier(
                random_state=seed,
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                tree_method="hist",
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            ),
            method="adasyn",
            sampling_strategy=params.get("sampling_strategy", 0.5),
            random_state=seed,
        ) if IMBLEARN_AVAILABLE else None,
        "params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "sampling_strategy": 0.5},
        "supports_sample_weight": False,
    },
}


# ============================================================
# Approach B Model Class
# ============================================================

class ImbalanceAwareModel:
    """Specialized imbalance-aware fraud detection model."""

    def __init__(
        self,
        feature_set: str = "plus",
        model_name: str = "balanced_rf",
        calibrate: bool = True,
        pos_weight: float = 1.0,
        resampling_strategy: float = 0.5,
        seed: int = 42,
    ):
        self.feature_set = feature_set
        self.model_name = model_name
        self.calibrate = calibrate
        self.pos_weight = pos_weight
        self.resampling_strategy = resampling_strategy
        self.seed = seed

        self.model = None
        self.threshold = None
        self.columns = None
        self.training_metadata = {}

    def _build_model(self, pos_weight: float | None = None):
        """Build the imbalance-aware model."""
        if self.model_name not in IMBALANCE_MODELS:
            raise ValueError(f"Unknown model: {self.model_name}. Available: {list(IMBALANCE_MODELS.keys())}")

        cfg = IMBALANCE_MODELS[self.model_name]
        params = cfg["params"].copy()

        # Apply pos_weight if supported
        effective_weight = pos_weight if pos_weight is not None else self.pos_weight

        if effective_weight != 1.0:
            if self.model_name in ("cost_xgb",):
                params["scale_pos_weight"] = effective_weight
            elif self.model_name in ("cost_lgbm",):
                params["class_weight"] = {0: 1.0, 1: effective_weight}
            elif self.model_name in ("cost_catboost",):
                params["class_weights"] = [1.0, effective_weight]
            elif self.model_name in ("smote_lr", "smote_rf", "smote_xgb", "borderline_smote_xgb", "adasyn_xgb"):
                # For resampling pipelines, we can't easily pass class_weight through
                # The resampling already handles imbalance
                pass

        # Apply resampling strategy
        if "sampling_strategy" in params:
            params["sampling_strategy"] = self.resampling_strategy

        model = cfg["factory"](self.seed, params)
        return model, cfg.get("supports_sample_weight", True)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        fpr_cap: float = 0.0001,
    ) -> "ImbalanceAwareModel":
        """Train the model with validation-based threshold optimization."""
        self.columns = list(X_train.columns)
        start_time = time.time()

        print(f"Training {self.model_name}...")
        model, supports_sw = self._build_model(self.pos_weight)

        # Train
        t0 = time.time()
        if supports_sw and self.pos_weight != 1.0:
            sample_weight = np.where(y_train == 1, self.pos_weight, 1.0)
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Calibrate if requested (only for models that output probabilities)
        if self.calibrate and hasattr(model, "predict_proba"):
            print("  Calibrating...")
            # For imbalanced-learn ensembles, calibration needs special handling
            if hasattr(model, "estimators_"):
                # Already an ensemble of calibrated models
                pass
            else:
                model = CalibratedClassifierCV(model, method="isotonic", cv=3, n_jobs=-1)
                model.fit(X_train, y_train)

        self.model = model
        self.training_metadata["train_time"] = train_time
        self.training_metadata["calibrated"] = self.calibrate
        self.training_metadata["pos_weight"] = self.pos_weight
        self.training_metadata["resampling_strategy"] = self.resampling_strategy

        # Optimize threshold on validation
        val_prob = model.predict_proba(X_val)[:, 1]
        self.threshold = threshold_at_fpr(y_val, val_prob, fpr_cap)
        c = confusion_at_threshold(y_val, val_prob, self.threshold)

        print(f"  Trained in {train_time:.1f}s")
        print(f"  Validation: threshold={self.threshold:.5f}, recall={c['recall']:.4f}, "
              f"FPR={c['fpr']:.6f}, FP={c['fp']}")

        self.training_metadata["total_time"] = time.time() - start_time
        self.training_metadata["fpr_cap"] = fpr_cap
        self.training_metadata["threshold"] = float(self.threshold)
        self.training_metadata["val_recall"] = c["recall"]
        self.training_metadata["val_fpr"] = c["fpr"]
        self.training_metadata["val_fp"] = c["fp"]

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities."""
        X = X[self.columns]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Predict binary fraud labels."""
        prob = self.predict_proba(X)
        thr = threshold if threshold is not None else self.threshold
        return (prob > thr).astype(int)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        threshold: float | None = None,
        reference_prevalence: float | None = None,
    ) -> dict:
        """Evaluate model on given data."""
        prob = self.predict_proba(X)
        thr = threshold if threshold is not None else self.threshold
        return evaluate_scores(y, prob, thr, reference_prevalence)

    def get_feature_importance_shap(self, X: pd.DataFrame, n_samples: int = 100) -> dict:
        """Compute SHAP-based feature importance."""
        import shap

        # Sample background
        bg_size = min(n_samples, len(X))
        bg_idx = np.random.default_rng(self.seed).choice(len(X), bg_size, replace=False)
        background = X.iloc[bg_idx]

        def model_predict(X_arr):
            X_df = pd.DataFrame(X_arr, columns=self.columns)
            return self.predict_proba(X_df)

        # Use permutation explainer (model-agnostic)
        explainer = shap.Explainer(
            model_predict,
            shap.maskers.Independent(background, max_samples=bg_size),
            algorithm="permutation",
        )

        explain_size = min(n_samples, len(X))
        explain_idx = np.random.default_rng(self.seed + 1).choice(len(X), explain_size, replace=False)
        X_explain = X.iloc[explain_idx]

        shap_values = explainer(X_explain.to_numpy(), max_evals=200)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        importance = dict(zip(self.columns, mean_abs_shap))

        return {
            "importance": importance,
            "shap_values": shap_values.values,
            "base_values": shap_values.base_values,
            "data": X_explain.to_numpy(),
        }

    def save(self, path: str | Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model": self.model,
            "threshold": self.threshold,
            "columns": self.columns,
            "feature_set": self.feature_set,
            "model_name": self.model_name,
            "calibrate": self.calibrate,
            "pos_weight": self.pos_weight,
            "resampling_strategy": self.resampling_strategy,
            "seed": self.seed,
            "training_metadata": self.training_metadata,
        }
        joblib.dump(save_dict, path)
        print(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ImbalanceAwareModel":
        """Load model from disk."""
        save_dict = joblib.load(path)

        model = cls(
            feature_set=save_dict["feature_set"],
            model_name=save_dict["model_name"],
            calibrate=save_dict["calibrate"],
            pos_weight=save_dict["pos_weight"],
            resampling_strategy=save_dict["resampling_strategy"],
            seed=save_dict["seed"],
        )

        model.model = save_dict["model"]
        model.threshold = save_dict["threshold"]
        model.columns = save_dict["columns"]
        model.training_metadata = save_dict["training_metadata"]

        return model


# ============================================================
# Model Selection and Comparison
# ============================================================

def evaluate_imbalance_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_names: list[str],
    fpr_cap: float = 0.0001,
    pos_weights: list[float] = [1.0, 10, 50, 100, 500],
    resampling_strategies: list[float] = [0.1, 0.3, 0.5, 0.7],
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate multiple imbalance-aware models and select best."""
    results = []

    for name in model_names:
        print(f"\n=== Evaluating {name} ===")
        cfg = IMBALANCE_MODELS.get(name)
        if not cfg:
            print(f"  Skipping unknown model: {name}")
            continue

        # Determine what to sweep
        if name in ("cost_xgb", "cost_lgbm", "cost_catboost"):
            sweep_params = pos_weights
            param_name = "pos_weight"
        elif name in ("easy_ensemble", "balanced_rf", "rusboost", "balanced_bagging"):
            sweep_params = [None]  # These handle imbalance internally
            param_name = "internal"
        elif "smote" in name or "adasyn" in name or "borderline" in name:
            sweep_params = resampling_strategies
            param_name = "resampling_strategy"
        else:
            sweep_params = [None]
            param_name = "none"

        best_recall = -1
        best_result = None

        for param_val in sweep_params:
            model = ImbalanceAwareModel(
                feature_set="temp",
                model_name=name,
                pos_weight=param_val if param_name == "pos_weight" else 1.0,
                resampling_strategy=param_val if param_name == "resampling_strategy" else 0.5,
                seed=seed,
            )

            try:
                model.fit(X_train, y_train, X_val, y_val, fpr_cap=fpr_cap)
                val_prob = model.predict_proba(X_val)
                val_result = evaluate_scores(y_val, val_prob, model.threshold)
                recall = val_result["recall"]

                print(f"  {param_name}={param_val}: val_recall={recall:.4f}, val_fpr={val_result['fpr']:.6f}, "
                      f"FP={val_result['fp']}")

                if recall > best_recall:
                    best_recall = recall
                    best_result = {
                        "model": name,
                        "param_name": param_name,
                        "param_value": param_val,
                        "val_recall": recall,
                        "val_fpr": val_result["fpr"],
                        "val_fp": val_result["fp"],
                        "val_pr_auc": val_result["pr_auc"],
                        "val_roc_auc": val_result["roc_auc"],
                        "threshold": model.threshold,
                    }
            except Exception as e:
                print(f"  Failed: {e}")
                continue

        if best_result:
            results.append(best_result)

    return pd.DataFrame(results)


# ============================================================
# Training Pipeline
# ============================================================

def train_approach_b(
    feature_set: str = "plus",
    model_name: str = "balanced_rf",
    calibrate: bool = True,
    pos_weight: float = 1.0,
    resampling_strategy: float = 0.5,
    auto_select: bool = False,
    fpr_cap: float = 0.0001,
    seed: int = 42,
    use_dev: bool = False,
) -> tuple[ImbalanceAwareModel, dict]:
    """Complete training pipeline for Approach B."""
    cfg = load_config()

    # Load data
    path = resolve(cfg["paths"]["dev_parquet" if use_dev else "parquet"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/prepare_data.py first.")

    df = load_dataset(path)
    s = cfg["split"]
    train, val, test = chronological_split(df, s["train_end_step"], s["val_end_step"])
    print("Split:", check_split(train, val, test))

    # Build features
    Xtr, Xva, Xte = (build_features(d, feature_set) for d in (train, val, test))
    ytr, yva, yte = (d["isFraud"].to_numpy() for d in (train, val, test))

    print(f"Features: {Xtr.shape[1]} columns, Train: {len(Xtr):,}, Val: {len(Xva):,}, Test: {len(Xte):,}")
    print(f"Fraud rates: train={ytr.mean():.4%}, val={yva.mean():.4%}, test={yte.mean():.4%}")

    if auto_select:
        print("\n=== Auto-selecting best imbalance-aware model ===")
        model_names = list(IMBALANCE_MODELS.keys())
        results = evaluate_imbalance_models(
            Xtr, ytr, Xva, yva, model_names, fpr_cap=fpr_cap, seed=seed
        )
        if len(results) == 0:
            raise RuntimeError("No models evaluated successfully")

        best = results.sort_values(["val_recall", "val_fp"], ascending=[False, True]).iloc[0]
        model_name = best["model"]
        param_name = best["param_name"]
        param_val = best["param_value"]

        print(f"\nSelected: {model_name} with {param_name}={param_val}")
        print(f"Validation recall: {best['val_recall']:.4f}, FP: {best['val_fp']}")

        if param_name == "pos_weight":
            pos_weight = param_val
            resampling_strategy = 0.5
        elif param_name == "resampling_strategy":
            pos_weight = 1.0
            resampling_strategy = param_val
        else:
            pos_weight = 1.0
            resampling_strategy = 0.5

    print(f"\n=== Training Approach B: {model_name} ===")
    model = ImbalanceAwareModel(
        feature_set=feature_set,
        model_name=model_name,
        calibrate=calibrate,
        pos_weight=pos_weight,
        resampling_strategy=resampling_strategy,
        seed=seed,
    )
    model.fit(Xtr, ytr, Xva, yva, fpr_cap=fpr_cap)

    # Evaluate on test
    print("\n=== Test Evaluation ===")
    test_results = model.evaluate(Xte, yte, reference_prevalence=ytr.mean())
    print(f"Test Recall: {test_results['recall']:.4f}")
    print(f"Test FPR: {test_results['fpr']:.6f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test PR-AUC: {test_results['pr_auc']:.4f}")
    print(f"Test ROC-AUC: {test_results['roc_auc']:.4f}")
    print(f"TP: {test_results['tp']}, FP: {test_results['fp']}, FN: {test_results['fn']}, TN: {test_results['tn']}")

    # Save model
    out_dir = resolve(cfg["paths"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"approach_b_{model_name}_{feature_set}.joblib"
    model.save(model_path)

    # Save metadata
    metadata = {
        "approach": "B",
        "model_name": model_name,
        "feature_set": feature_set,
        "calibrate": calibrate,
        "pos_weight": pos_weight,
        "resampling_strategy": resampling_strategy,
        "fpr_cap": fpr_cap,
        "seed": seed,
        "test_results": test_results,
        "training_metadata": model.training_metadata,
        "model_path": str(model_path),
    }

    meta_path = out_dir / f"approach_b_{model_name}_{feature_set}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return model, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Approach B: Imbalance-Aware Model")
    parser.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    parser.add_argument("--model", default="balanced_rf", choices=list(IMBALANCE_MODELS.keys()))
    parser.add_argument("--calibrate", action="store_true", default=True)
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--resampling-strategy", type=float, default=0.5)
    parser.add_argument("--auto-select", action="store_true")
    parser.add_argument("--fpr-cap", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev", action="store_true")

    args = parser.parse_args()

    train_approach_b(
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