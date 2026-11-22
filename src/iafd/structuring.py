"""Structuring attack (experiment E): a fraudster splits one account drain into k smaller transfers.

All arithmetic is done in integer cents so balances add up EXACTLY: with residual=0 the last piece moves
the entire remaining balance (so it still looks like a full drain of what is left), and with residual>0 the
account is never emptied (the harder case for balance-based features).
Pieces keep the original step, type and ids. If the original fraud left the receiver balance unchanged
(a PaySim quirk), the pieces do too; otherwise the receiver balance grows piece by piece.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PIECE_COLS = ["step", "type", "orig_id", "dest_id", "amount", "oldbalanceOrg", "newbalanceOrig",
              "oldbalanceDest", "newbalanceDest", "isFraud", "event_id"]


def split_frauds(frauds: pd.DataFrame, rng: np.random.Generator, k_choices, residual: float = 0.0,
                 equal: bool = True) -> pd.DataFrame:
    """Replace each fraud row by k pieces (k drawn from k_choices). Returns piece rows with an `event_id`
    (position of the original fraud). equal=False draws random (Dirichlet) piece sizes."""
    n = len(frauds)
    if n == 0:
        raise ValueError("no fraud rows to split")
    k_choices = np.asarray(k_choices, dtype=int)
    if (k_choices < 1).any():
        raise ValueError("k must be >= 1")
    if not 0 <= residual < 1:
        raise ValueError("residual must be in [0, 1)")

    k = rng.choice(k_choices, size=n)
    total_pieces = int(k.sum())
    starts = np.r_[0, np.cumsum(k)[:-1]]
    last = starts + k - 1
    event = np.repeat(np.arange(n), k)

    cents = lambda col: np.round(frauds[col].to_numpy(dtype="float64") * 100).astype(np.int64)
    amt_c, old_o_c, old_d_c, new_d_c = cents("amount"), cents("oldbalanceOrg"), cents("oldbalanceDest"), cents("newbalanceDest")
    total_c = np.round(amt_c * (1.0 - residual)).astype(np.int64)

    if equal:
        w = 1.0 / np.repeat(k, k)
    else:
        raw = rng.gamma(1.0, size=total_pieces)
        w = raw / np.repeat(np.add.reduceat(raw, starts), k)
    ac = np.maximum(np.floor(np.repeat(total_c, k) * w).astype(np.int64), 1)
    prior = np.add.reduceat(ac, starts) - ac[last]
    ac[last] = np.maximum(total_c - prior, 1)                 # the last piece takes exactly the remainder

    cs = np.cumsum(ac)
    base = np.repeat(np.r_[0, cs[last[:-1]]], k)
    before = cs - ac - base                                    # cents moved by earlier pieces of the same event

    old_o = np.repeat(old_o_c, k) - before
    new_o = np.maximum(old_o - ac, 0)
    upd = np.repeat(new_d_c != old_d_c, k)
    old_d = np.repeat(old_d_c, k) + np.where(upd, before, 0)
    new_d = old_d + np.where(upd, ac, 0)

    rep = lambda col: np.repeat(frauds[col].to_numpy(), k)
    return pd.DataFrame({
        "step": rep("step"), "type": np.repeat(frauds["type"].astype(str).to_numpy(), k),
        "orig_id": rep("orig_id"), "dest_id": rep("dest_id"),
        "amount": ac / 100.0, "oldbalanceOrg": old_o / 100.0, "newbalanceOrig": new_o / 100.0,
        "oldbalanceDest": old_d / 100.0, "newbalanceDest": new_d / 100.0,
        "isFraud": 1, "event_id": event,
    })[PIECE_COLS]


def event_max(scores: np.ndarray, event_id: np.ndarray) -> np.ndarray:
    """Highest piece score per event (event_id must be sorted and contiguous, as split_frauds returns it).
    An event counts as detected when ANY of its pieces is flagged."""
    starts = np.r_[0, np.flatnonzero(np.diff(event_id)) + 1]
    return np.maximum.reduceat(scores, starts)
