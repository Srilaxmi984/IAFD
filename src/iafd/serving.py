"""Scoring and explanation of ONE transaction with the two final models (used by app/app.py).

The bundles are written by scripts/final_evaluate.py to models/final_single.joblib and models/final_ensemble.joblib.
Type gating: a transaction type with no fraud in the training data is scored exactly 0 and never flagged. That is a
property of PaySim (fraud only occurs in TRANSFER and CASH_OUT), not a guarantee for other data.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import TYPES
from .explain import logit
from .features import build_features
from .final import combine_spec

TREE_MODELS = {"rf", "xgb", "lgbm"}
OPS = ("cap", "recall", "min_cost")
OP_LABEL = {"cap": "false-positive rate at most 0.01%", "recall": "recall at least 99.5% (fewest alarms)",
            "min_cost": "minimum expected cost"}


def load_bundles(models_dir) -> dict:
    d, out = Path(models_dir), {}
    for key, fname in (("single", "final_single.joblib"), ("ensemble", "final_ensemble.joblib")):
        p = d / fname
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run: python scripts/final_evaluate.py")
        out[key] = joblib.load(p)
    return out


def auto_balances(tx_type: str, amount: float, old_orig: float, old_dest: float) -> tuple[float, float]:
    """'After' balances of a NORMAL transaction: the sender pays the amount (CASH_IN adds it), the receiver gains it
    for TRANSFER and CASH_OUT and is unchanged otherwise. The sender balance never goes below 0."""
    amount = max(float(amount), 0.0)
    new_orig = old_orig + amount if tx_type == "CASH_IN" else max(old_orig - amount, 0.0)
    new_dest = old_dest + amount if tx_type in ("TRANSFER", "CASH_OUT") else old_dest
    return round(float(new_orig), 2), round(float(new_dest), 2)


def make_row(tx_type: str, amount: float, old_orig: float, new_orig: float, old_dest: float, new_dest: float,
             step: int = 1) -> pd.DataFrame:
    if tx_type not in TYPES:
        raise ValueError(f"unknown transaction type {tx_type!r}; choose from {TYPES}")
    return pd.DataFrame([{"step": int(step), "type": tx_type, "orig_id": 1, "dest_id": 2, "amount": float(amount),
                          "oldbalanceOrg": float(old_orig), "newbalanceOrig": float(new_orig),
                          "oldbalanceDest": float(old_dest), "newbalanceDest": float(new_dest), "isFraud": 0}])


def features_for(bundle: dict, row: pd.DataFrame) -> pd.DataFrame:
    return build_features(row, bundle["feature_set"])[bundle["columns"]]


def model_proba(bundle: dict, X) -> np.ndarray:
    """Fraud probability of the bundle's model on feature rows (no gating applied here)."""
    if "estimator" in bundle:
        return bundle["estimator"].predict_proba(X)[:, 1]
    return combine_spec([m.predict_proba(X)[:, 1] for m in bundle["models"]], bundle["aggregator"], bundle.get("weights"))


def score_row(bundle: dict, row: pd.DataFrame) -> dict:
    t = str(row["type"].iloc[0])
    gated = bool(bundle.get("gate")) and t not in set(bundle.get("allowed_types", []))
    score = 0.0 if gated else float(model_proba(bundle, features_for(bundle, row))[0])
    return {"score": score, "gated": gated}


def decide(bundle: dict, score: float, op: str) -> tuple[bool, float]:
    thr = float(bundle["thresholds"][op])
    return bool(score > thr), thr


def _positive_class(sv) -> np.ndarray:
    if isinstance(sv, list):
        return np.asarray(sv[1])
    arr = np.asarray(sv)
    return arr[:, :, 1] if arr.ndim == 3 else arr


def _base_value(ev) -> float:
    arr = np.atleast_1d(np.asarray(ev, dtype="float64"))
    return float(arr[1] if len(arr) == 2 else arr[0])


def explain_row(bundle: dict, row: pd.DataFrame, background: np.ndarray, max_evals: int = 300):
    """SHAP values for one transaction: (Series of values per feature, base value, scale label).
    Single tree model: exact TreeSHAP. Ensemble (and non-tree models): permutation SHAP on the log-odds of the
    model's score against a background of typical transactions."""
    import shap
    X = features_for(bundle, row)
    cols = list(X.columns)
    if "estimator" in bundle and bundle.get("model") in TREE_MODELS:
        ex = shap.TreeExplainer(bundle["estimator"])
        values = _positive_class(ex.shap_values(X))[0]
        scale = "probability" if bundle["model"] == "rf" else "log-odds"
        return pd.Series(values, index=cols), _base_value(ex.expected_value), scale

    def f(Xn):
        return logit(model_proba(bundle, pd.DataFrame(np.asarray(Xn, dtype="float64"), columns=cols)))
    ex = shap.Explainer(f, shap.maskers.Independent(background, max_samples=len(background)), algorithm="permutation")
    out = ex(X.to_numpy(), max_evals=max_evals, silent=True)
    return pd.Series(out.values[0], index=cols), float(np.asarray(out.base_values)[0]), "log-odds"


def default_background(bundle: dict, n: int = 100, dev_parquet=None, seed: int = 0) -> np.ndarray:
    """Typical transactions of the allowed types: from the dev sample if present, else a small synthetic set."""
    from .data import compact_frame, load_dataset
    if dev_parquet is not None and Path(dev_parquet).exists():
        df = load_dataset(dev_parquet)
    else:
        from .synthetic import make_synthetic_paysim
        df = compact_frame(make_synthetic_paysim(5000, seed=seed))
    if bundle.get("gate"):
        df = df[df["type"].astype(str).isin(bundle.get("allowed_types", []))]
    df = df.sample(min(n, len(df)), random_state=seed)
    return features_for(bundle, df).to_numpy()
