"""Tiny PaySim-like generator (raw schema) for tests and demos. NOT a substitute for the real data."""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_synthetic_paysim(n_rows: int = 40_000, fraud_rate: float = 0.004,
                          n_steps: int = 743, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"], size=n_rows,
                       p=[0.35, 0.34, 0.22, 0.08, 0.01])
    amount = np.round(rng.lognormal(mean=9.5, sigma=1.2, size=n_rows), 2)
    old_o = np.round(amount * rng.uniform(1.0, 5.0, n_rows), 2)
    new_o = np.round(np.maximum(old_o - amount, 0), 2)
    old_d = np.round(rng.lognormal(9.0, 1.5, n_rows) * (rng.random(n_rows) > 0.4), 2)
    new_d = np.round(old_d + amount, 2)
    step = rng.integers(1, n_steps + 1, size=n_rows)

    eligible = np.where(np.isin(types, ["TRANSFER", "CASH_OUT"]))[0]
    n_fraud = max(int(n_rows * fraud_rate), 10)
    fi = rng.choice(eligible, size=n_fraud, replace=False)
    amount[fi] = np.round(rng.uniform(1e5, 1e7, n_fraud), 2)      # fraud: large, drains the account
    old_o[fi], new_o[fi] = amount[fi], 0.0
    new_d[fi] = np.round(old_d[fi] + amount[fi], 2)
    is_fraud = np.zeros(n_rows, dtype=int)
    is_fraud[fi] = 1

    df = pd.DataFrame({
        "step": step, "type": types, "amount": amount,
        "nameOrig": [f"C{x}" for x in rng.integers(10**8, 10**9, n_rows)],
        "oldbalanceOrg": old_o, "newbalanceOrig": new_o,
        "nameDest": [f"{'M' if t == 'PAYMENT' else 'C'}{x}" for t, x in zip(types, rng.integers(10**8, 10**9, n_rows))],
        "oldbalanceDest": old_d, "newbalanceDest": new_d,
        "isFraud": is_fraud, "isFlaggedFraud": 0,
    })
    return df.sort_values("step", kind="stable").reset_index(drop=True)
