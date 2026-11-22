"""Approach A: Advanced Optimized Ensemble Learning for Fraud Detection.

This module implements a sophisticated ensemble architecture that combines complementary
machine learning algorithms with proper calibration, hyperparameter optimization,
and threshold optimization.
"""
from __future__ import annotations
import time
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize

# Scikit-learn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    VotingClassifier,
    BaggingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    matthews_corrcoef,
)

# Gradient boosting
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available. Install with: pip install catboost")

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
# Model Definitions
# ============================================================

ENSEMBLE_BASE_MODELS = {
    # Linear model (calibrated)
    "lr": {
        "factory": lambda seed, params: make_pipeline(
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
        "needs_calibration": True,
        "params": {"max_iter": 2000, "tol": 1e-8},
    },
    # Tree-based models
    "rf": {
        "factory": lambda seed, params: RandomForestClassifier(
            random_state=seed,
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 16),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            class_weight=params.get("class_weight"),
            n_jobs=-1,
        ),
        "needs_calibration": True,
        "params": {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 5},
    },
    "et": {
        "factory": lambda seed, params: ExtraTreesClassifier(
            random_state=seed,
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 16),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            class_weight=params.get("class_weight"),
            n_jobs=-1,
        ),
        "needs_calibration": True,
        "params": {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 5},
    },
    "brf": {
        "factory": lambda seed, params: RandomForestClassifier(
            random_state=seed,
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 16),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            class_weight="balanced",
            n_jobs=-1,
        ),
        "needs_calibration": True,
        "params": {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 5},
    },
    # Gradient boosting
    "xgb": {
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
        "needs_calibration": True,
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
    "lgbm": {
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
        "needs_calibration": True,
        "params": {
            "n_estimators": 300,
            "num_leaves": 63,
            "learning_rate": 0.1,
            "min_child_weight": 1.0,
            "max_delta_step": 1.0,
        },
    },
}

if CATBOOST_AVAILABLE:
    ENSEMBLE_BASE_MODELS["catboost"] = {
        "factory": lambda seed, params: CatBoostClassifier(
            random_state=seed,
            iterations=params.get("iterations", 300),
            depth=params.get("depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            class_weights=params.get("class_weights"),
            verbose=False,
            thread_count=-1,
        ),
        "needs_calibration": True,
        "params": {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.1,
        },
    }


# ============================================================
# Utility Functions
# ============================================================

def get_class_weights(y: np.ndarray, method: str = "balanced") -> dict:
    """Compute class weights for imbalanced data."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight(method, classes=classes, y=y)
    return dict(zip(classes, weights))


def calibrate_model(model, X: pd.DataFrame, y: np.ndarray, method: str = "isotonic", cv: int = 3) -> CalibratedClassifierCV:
    """Calibrate a model's predicted probabilities."""
    calibrated = CalibratedClassifierCV(model, method=method, cv=cv, n_jobs=-1)
    calibrated.fit(X, y)
    return calibrated


def optimize_ensemble_weights(
    val_probas: list[np.ndarray],
    y_val: np.ndarray,
    fpr_cap: float,
    method: str = "Nelder-Mead",
) -> np.ndarray:
    """Optimize ensemble weights to maximize recall at FPR cap on validation."""
    n_models = len(val_probas)

    def objective(weights):
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        ensemble_prob = np.average(val_probas, axis=0, weights=weights)
        thr = threshold_at_fpr(y_val, ensemble_prob, fpr_cap)
        c = confusion_at_threshold(y_val, ensemble_prob, thr)
        return -c["recall"]  # Minimize negative recall

    # Start with equal weights
    x0 = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]

    result = minimize(objective, x0, method=method, bounds=bounds)
    optimal_weights = np.maximum(result.x, 0)
    optimal_weights = optimal_weights / optimal_weights.sum()

    return optimal_weights


def optimize_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str = "f1",
) -> tuple[float, dict]:
    """Find optimal threshold for a given metric."""
    from sklearn.metrics import precision_recall_fscore_support

    if metric == "f1":
        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx]), {"f1": f1_scores[best_idx]}

    elif metric == "f2":
        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
        f2_scores = 5 * (precisions * recalls) / (4 * precisions + recalls + 1e-10)
        best_idx = np.argmax(f2_scores[:-1])
        return float(thresholds[best_idx]), {"f2": f2_scores[best_idx]}

    elif metric == "mcc":
        thresholds = np.unique(scores)
        best_mcc = -1
        best_thr = 0.5
        for thr in thresholds:
            pred = scores > thr
            mcc = matthews_corrcoef(y_true, pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thr = thr
        return float(best_thr), {"mcc": best_mcc}

    elif metric == "fpr_cap":
        # This is handled by threshold_at_fpr
        pass

    return 0.5, {}


def compute_model_diversity(probas_list: list[np.ndarray]) -> dict:
    """Compute diversity metrics among ensemble members."""
    n = len(probas_list)
    correlations = np.zeros((n, n))
    disagreements = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            # Pearson correlation
            corr = np.corrcoef(probas_list[i], probas_list[j])[0, 1]
            correlations[i, j] = corr
            correlations[j, i] = corr

            # Disagreement rate (at 0.5 threshold)
            pred_i = probas_list[i] > 0.5
            pred_j = probas_list[j] > 0.5
            dis = np.mean(pred_i != pred_j)
            disagreements[i, j] = dis
            disagreements[j, i] = dis

    return {
        "mean_correlation": float(np.mean(correlations[np.triu_indices(n, 1)])),
        "max_correlation": float(np.max(correlations[np.triu_indices(n, 1)])),
        "mean_disagreement": float(np.mean(disagreements[np.triu_indices(n, 1)])),
        "correlation_matrix": correlations.tolist(),
        "disagreement_matrix": disagreements.tolist(),
    }


# ============================================================
# Approach A Ensemble Class
# ============================================================

class AdvancedEnsemble:
    """Advanced ensemble with multiple architectures and optimization."""

    def __init__(
        self,
        feature_set: str = "plus",
        base_models: list[str] | None = None,
        architecture: str = "stacking",
        calibrate: bool = True,
        optimize_weights: bool = True,
        pos_weight: float = 1.0,
        seed: int = 42,
    ):
        self.feature_set = feature_set
        self.base_model_names = base_models or ["rf", "xgb", "lgbm", "et"]
        self.architecture = architecture
        self.calibrate = calibrate
        self.optimize_weights = optimize_weights
        self.pos_weight = pos_weight
        self.seed = seed

        self.models = {}
        self.calibrated_models = {}
        self.meta_learner = None
        self.ensemble_weights = None
        self.threshold = None
        self.columns = None
        self.diversity_metrics = None
        self.training_metadata = {}

    def _build_base_model(self, name: str, pos_weight: float | None = None):
        """Build a single base model."""
        if name not in ENSEMBLE_BASE_MODELS:
            raise ValueError(f"Unknown model: {name}. Available: {list(ENSEMBLE_BASE_MODELS.keys())}")

        cfg = ENSEMBLE_BASE_MODELS[name]
        params = cfg["params"].copy()

        if pos_weight is not None and pos_weight != 1.0:
            if name in ("lr", "rf", "et", "brf"):
                params["class_weight"] = {0: 1.0, 1: pos_weight}
            elif name in ("xgb",):
                params["scale_pos_weight"] = pos_weight
            elif name in ("lgbm",):
                params["class_weight"] = {0: 1.0, 1: pos_weight}
            elif name == "catboost":
                params["class_weights"] = [1.0, pos_weight]

        return cfg["factory"](self.seed, params), cfg["needs_calibration"]

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        fpr_cap: float = 0.0001,
    ) -> "AdvancedEnsemble":
        """Train the ensemble with validation-based optimization."""
        self.columns = list(X_train.columns)
        start_time = time.time()

        # Build base models
        print(f"Building {len(self.base_model_names)} base models...")
        val_probas = []

        for name in self.base_model_names:
            model, needs_cal = self._build_base_model(name, self.pos_weight)

            # Train
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            # Calibrate if needed
            if self.calibrate and needs_cal:
                print(f"  Calibrating {name}...")
                model = calibrate_model(model, X_train, y_train)

            # Get validation probabilities
            val_prob = model.predict_proba(X_val)[:, 1]
            val_probas.append(val_prob)

            self.models[name] = model
            self.training_metadata[name] = {
                "train_time": train_time,
                "calibrated": self.calibrate and needs_cal,
            }

            print(f"  {name}: trained in {train_time:.1f}s")

        # Compute diversity
        self.diversity_metrics = compute_model_diversity(val_probas)
        print(f"\nDiversity: mean_corr={self.diversity_metrics['mean_correlation']:.3f}, "
              f"mean_disagree={self.diversity_metrics['mean_disagreement']:.3f}")

        # Build ensemble based on architecture
        if self.architecture == "soft_voting":
            self.ensemble_weights = np.ones(len(self.base_model_names)) / len(self.base_model_names)

        elif self.architecture == "weighted_voting":
            if self.optimize_weights:
                print("Optimizing ensemble weights...")
                self.ensemble_weights = optimize_ensemble_weights(val_probas, y_val, fpr_cap)
                print(f"  Optimal weights: {dict(zip(self.base_model_names, self.ensemble_weights.round(3)))}")
            else:
                self.ensemble_weights = np.ones(len(self.base_model_names)) / len(self.base_model_names)

        elif self.architecture == "stacking":
            print("Training stacking meta-learner...")
            # Use base models' validation predictions as features for meta-learner
            val_meta_X = np.column_stack(val_probas)

            # Meta-learner: Logistic Regression on base model predictions
            self.meta_learner = LogisticRegression(
                random_state=self.seed,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )
            self.meta_learner.fit(val_meta_X, y_val)

            # Get ensemble weights from meta-learner coefficients
            self.ensemble_weights = np.abs(self.meta_learner.coef_[0])
            self.ensemble_weights = self.ensemble_weights / self.ensemble_weights.sum()
            print(f"  Stacking weights: {dict(zip(self.base_model_names, self.ensemble_weights.round(3)))}")

        # Optimize threshold on validation
        if self.architecture in ("soft_voting", "weighted_voting"):
            val_ensemble = np.average(val_probas, axis=0, weights=self.ensemble_weights)
        elif self.architecture == "stacking":
            val_ensemble = self.meta_learner.predict_proba(np.column_stack(val_probas))[:, 1]

        self.threshold = threshold_at_fpr(y_val, val_ensemble, fpr_cap)
        c = confusion_at_threshold(y_val, val_ensemble, self.threshold)
        print(f"\nValidation: threshold={self.threshold:.5f}, recall={c['recall']:.4f}, "
              f"FPR={c['fpr']:.6f}, FP={c['fp']}")

        self.training_metadata["total_time"] = time.time() - start_time
        self.training_metadata["architecture"] = self.architecture
        self.training_metadata["fpr_cap"] = fpr_cap
        self.training_metadata["threshold"] = float(self.threshold)
        self.training_metadata["ensemble_weights"] = self.ensemble_weights.tolist()
        self.training_metadata["diversity"] = self.diversity_metrics

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities."""
        # Ensure column order
        X = X[self.columns]

        probas = []
        for name in self.base_model_names:
            prob = self.models[name].predict_proba(X)[:, 1]
            probas.append(prob)

        probas = np.column_stack(probas)

        if self.architecture in ("soft_voting", "weighted_voting"):
            return np.average(probas, axis=1, weights=self.ensemble_weights)
        elif self.architecture == "stacking":
            return self.meta_learner.predict_proba(probas)[:, 1]

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
        """Evaluate ensemble on given data."""
        prob = self.predict_proba(X)
        thr = threshold if threshold is not None else self.threshold
        return evaluate_scores(y, prob, thr, reference_prevalence)

    def get_feature_importance_shap(self, X: pd.DataFrame, n_samples: int = 100) -> dict:
        """Compute SHAP-based feature importance for the ensemble."""
        import shap

        # Sample background
        bg_size = min(n_samples, len(X))
        bg_idx = np.random.default_rng(self.seed).choice(len(X), bg_size, replace=False)
        background = X.iloc[bg_idx]

        # Create ensemble prediction function
        def ensemble_predict(X_arr):
            X_df = pd.DataFrame(X_arr, columns=self.columns)
            return self.predict_proba(X_df)

        # Use permutation explainer (model-agnostic)
        explainer = shap.Explainer(
            ensemble_predict,
            shap.maskers.Independent(background, max_samples=bg_size),
            algorithm="permutation",
        )

        # Explain a subset
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
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "models": self.models,
            "meta_learner": self.meta_learner,
            "ensemble_weights": self.ensemble_weights,
            "threshold": self.threshold,
            "columns": self.columns,
            "feature_set": self.feature_set,
            "base_model_names": self.base_model_names,
            "architecture": self.architecture,
            "calibrate": self.calibrate,
            "optimize_weights": self.optimize_weights,
            "pos_weight": self.pos_weight,
            "seed": self.seed,
            "diversity_metrics": self.diversity_metrics,
            "training_metadata": self.training_metadata,
        }
        joblib.dump(save_dict, path)
        print(f"Saved ensemble to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "AdvancedEnsemble":
        """Load ensemble from disk."""
        save_dict = joblib.load(path)

        ensemble = cls(
            feature_set=save_dict["feature_set"],
            base_models=save_dict["base_model_names"],
            architecture=save_dict["architecture"],
            calibrate=save_dict["calibrate"],
            optimize_weights=save_dict["optimize_weights"],
            pos_weight=save_dict["pos_weight"],
            seed=save_dict["seed"],
        )

        ensemble.models = save_dict["models"]
        ensemble.meta_learner = save_dict["meta_learner"]
        ensemble.ensemble_weights = save_dict["ensemble_weights"]
        ensemble.threshold = save_dict["threshold"]
        ensemble.columns = save_dict["columns"]
        ensemble.diversity_metrics = save_dict["diversity_metrics"]
        ensemble.training_metadata = save_dict["training_metadata"]

        return ensemble


# ============================================================
# Hyperparameter Optimization with Optuna
# ============================================================

def optimize_hyperparameters_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_name: str,
    n_trials: int = 30,
    fpr_cap: float = 0.0001,
    seed: int = 42,
) -> dict:
    """Optimize hyperparameters for a single model using Optuna."""
    try:
        import optuna
    except ImportError:
        print("Optuna not available. Install with: pip install optuna")
        return ENSEMBLE_BASE_MODELS[model_name]["params"]

    def objective(trial):
        params = {}

        if model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 8, 24),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            }
            model = RandomForestClassifier(
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
                **params,
            )
        elif model_name == "et":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 8, 24),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            }
            model = ExtraTreesClassifier(
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
                **params,
            )
        elif model_name == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = XGBClassifier(
                random_state=seed,
                tree_method="hist",
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
                **params,
            )
        elif model_name == "lgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0, log=True),
            }
            model = LGBMClassifier(
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
                class_weight="balanced",
                **params,
            )
        else:
            raise ValueError(f"Optimization not implemented for {model_name}")

        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        thr = threshold_at_fpr(y_val, val_prob, fpr_cap)
        c = confusion_at_threshold(y_val, val_prob, thr)

        # Maximize recall at FPR cap
        return c["recall"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"Best {model_name} params: {best_params}")
    print(f"Best validation recall: {study.best_value:.4f}")

    return best_params


# ============================================================
# Training Pipeline
# ============================================================

def train_approach_a(
    feature_set: str = "plus",
    base_models: list[str] | None = None,
    architecture: str = "stacking",
    calibrate: bool = True,
    optimize_weights: bool = True,
    pos_weight: float = 1.0,
    optimize_hyperparams: bool = False,
    n_trials: int = 30,
    seed: int = 42,
    fpr_cap: float = 0.0001,
    use_dev: bool = False,
) -> tuple[AdvancedEnsemble, dict]:
    """Complete training pipeline for Approach A."""
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

    # Hyperparameter optimization (optional)
    if optimize_hyperparams and base_models:
        print("\n=== Hyperparameter Optimization ===")
        for name in base_models:
            if name in ENSEMBLE_BASE_MODELS:
                best_params = optimize_hyperparameters_optuna(
                    Xtr, ytr, Xva, yva, name, n_trials=n_trials, fpr_cap=fpr_cap, seed=seed
                )
                ENSEMBLE_BASE_MODELS[name]["params"].update(best_params)

    # Train ensemble
    print(f"\n=== Training Approach A Ensemble ({architecture}) ===")
    ensemble = AdvancedEnsemble(
        feature_set=feature_set,
        base_models=base_models,
        architecture=architecture,
        calibrate=calibrate,
        optimize_weights=optimize_weights,
        pos_weight=pos_weight,
        seed=seed,
    )
    ensemble.fit(Xtr, ytr, Xva, yva, fpr_cap=fpr_cap)

    # Evaluate on test
    print("\n=== Test Evaluation ===")
    test_results = ensemble.evaluate(Xte, yte, reference_prevalence=ytr.mean())
    print(f"Test Recall: {test_results['recall']:.4f}")
    print(f"Test FPR: {test_results['fpr']:.6f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test PR-AUC: {test_results['pr_auc']:.4f}")
    print(f"Test ROC-AUC: {test_results['roc_auc']:.4f}")
    print(f"TP: {test_results['tp']}, FP: {test_results['fp']}, FN: {test_results['fn']}, TN: {test_results['tn']}")

    # Save model
    out_dir = resolve(cfg["paths"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"approach_a_{architecture}_{feature_set}.joblib"
    ensemble.save(model_path)

    # Save metadata
    metadata = {
        "approach": "A",
        "architecture": architecture,
        "feature_set": feature_set,
        "base_models": base_models,
        "calibrate": calibrate,
        "optimize_weights": optimize_weights,
        "pos_weight": pos_weight,
        "fpr_cap": fpr_cap,
        "seed": seed,
        "test_results": test_results,
        "training_metadata": ensemble.training_metadata,
        "model_path": str(model_path),
    }

    meta_path = out_dir / f"approach_a_{architecture}_{feature_set}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return ensemble, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Approach A: Advanced Ensemble")
    parser.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    parser.add_argument("--models", nargs="+", default=["rf", "xgb", "lgbm", "et"])
    parser.add_argument("--architecture", default="stacking", choices=["soft_voting", "weighted_voting", "stacking"])
    parser.add_argument("--calibrate", action="store_true", default=True)
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    parser.add_argument("--optimize-weights", action="store_true", default=True)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--optimize-hyperparams", action="store_true")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--fpr-cap", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev", action="store_true")

    args = parser.parse_args()

    train_approach_a(
        feature_set=args.feature_set,
        base_models=args.models,
        architecture=args.architecture,
        calibrate=args.calibrate,
        optimize_weights=args.optimize_weights,
        pos_weight=args.pos_weight,
        optimize_hyperparams=args.optimize_hyperparams,
        n_trials=args.n_trials,
        fpr_cap=args.fpr_cap,
        seed=args.seed,
        use_dev=args.dev,
    )