"""Feature construction.

Feature sets
------------
paper       the 5 raw money columns, exactly what the base paper trained on.
velocity    plus + causal sender velocity (counter-structuring).
plus        engineered + drains_full_balance + amount_exceeds_balance (from validation error analysis).
no_dest     engineered minus all receiver-balance features (ablation).
engineered  money columns + log amount + balance-consistency errors + emptied/zero flags
            + amount-to-balance ratio + one-hot transaction type.

`step`, ids and `isFlaggedFraud` are never used as features.
Velocity features (experiment C) are added in a later phase.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import MONEY_COLS, TYPES
from .velocity import velocity_features

TYPE_COLS = [f"type_{t}" for t in TYPES]
ENGINEERED_EXTRA = ["log_amount", "error_orig", "error_dest", "orig_emptied",
                    "dest_zero_before", "dest_zero_after", "amount_to_balance"]
PLUS_EXTRA = ["drains_full_balance", "amount_exceeds_balance"]
VELOCITY_EXTRA = ["orig_prev_count", "orig_prev_ratio"]
FEATURE_SETS = {
    "paper": list(MONEY_COLS),
    "engineered": list(MONEY_COLS) + ENGINEERED_EXTRA + TYPE_COLS,
    # ablation: removes every receiver-balance feature (PaySim leaves the receiver balance
    # unchanged for about half of the fraud, which almost never happens for legit rows)
    # engineered + the two 'transfer the whole balance' features suggested by validation error analysis
    "plus": list(MONEY_COLS) + ENGINEERED_EXTRA + TYPE_COLS + PLUS_EXTRA,
    # plus + causal sender velocity (only useful against structuring; see src/iafd/velocity.py)
    "velocity": list(MONEY_COLS) + ENGINEERED_EXTRA + TYPE_COLS + PLUS_EXTRA + VELOCITY_EXTRA,
    "no_dest": ["amount", "oldbalanceOrg", "newbalanceOrig", "log_amount", "error_orig",
                "orig_emptied", "amount_to_balance"] + TYPE_COLS,
}


def build_features(df: pd.DataFrame, feature_set: str = "engineered") -> pd.DataFrame:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}")
    amt = df["amount"].to_numpy(dtype="float64")
    old_o = df["oldbalanceOrg"].to_numpy(dtype="float64")
    new_o = df["newbalanceOrig"].to_numpy(dtype="float64")
    old_d = df["oldbalanceDest"].to_numpy(dtype="float64")
    new_d = df["newbalanceDest"].to_numpy(dtype="float64")
    cols = {c: df[c].to_numpy(dtype="float64") for c in MONEY_COLS}
    if feature_set in ("engineered", "no_dest", "plus", "velocity"):
        cols["log_amount"] = np.log1p(amt)
        cols["error_orig"] = new_o + amt - old_o      # != 0 when sender balance does not add up
        cols["error_dest"] = old_d + amt - new_d      # != 0 when receiver balance does not add up
        cols["orig_emptied"] = ((old_o > 0) & (new_o == 0)).astype("float64")
        cols["dest_zero_before"] = (old_d == 0).astype("float64")
        cols["dest_zero_after"] = (new_d == 0).astype("float64")
        cols["amount_to_balance"] = amt / (old_o + 1.0)
        cols["drains_full_balance"] = ((old_o > 0) & (np.abs(amt - old_o) <= 0.01)).astype("float64")
        cols["amount_exceeds_balance"] = (amt > old_o + 0.01).astype("float64")
        t = df["type"].astype(str).to_numpy()
        for name in TYPES:
            cols[f"type_{name}"] = (t == name).astype("float64")
    if feature_set == "velocity":
        cols.update(velocity_features(df))        # needs orig_id and step; row order = stream order
    return pd.DataFrame({c: cols[c] for c in FEATURE_SETS[feature_set]})
