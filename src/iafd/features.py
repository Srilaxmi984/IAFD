"""Feature construction.

Feature sets
------------
paper       the 5 raw money columns, exactly what the base paper trained on.
engineered  money columns + log amount + balance-consistency errors + emptied/zero flags
            + amount-to-balance ratio + one-hot transaction type.

`step`, ids and `isFlaggedFraud` are never used as features.
Velocity features (experiment C) are added in a later phase.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import MONEY_COLS, TYPES

TYPE_COLS = [f"type_{t}" for t in TYPES]
ENGINEERED_EXTRA = ["log_amount", "error_orig", "error_dest", "orig_emptied",
                    "dest_zero_before", "dest_zero_after", "amount_to_balance"]
FEATURE_SETS = {
    "paper": list(MONEY_COLS),
    "engineered": list(MONEY_COLS) + ENGINEERED_EXTRA + TYPE_COLS,
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
    if feature_set == "engineered":
        cols["log_amount"] = np.log1p(amt)
        cols["error_orig"] = new_o + amt - old_o      # != 0 when sender balance does not add up
        cols["error_dest"] = old_d + amt - new_d      # != 0 when receiver balance does not add up
        cols["orig_emptied"] = ((old_o > 0) & (new_o == 0)).astype("float64")
        cols["dest_zero_before"] = (old_d == 0).astype("float64")
        cols["dest_zero_after"] = (new_d == 0).astype("float64")
        cols["amount_to_balance"] = amt / (old_o + 1.0)
        t = df["type"].astype(str).to_numpy()
        for name in TYPES:
            cols[f"type_{name}"] = (t == name).astype("float64")
    return pd.DataFrame({c: cols[c] for c in FEATURE_SETS[feature_set]})
