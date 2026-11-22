"""Causal sender-velocity features (experiment C).

For each row: how many EARLIER rows (earlier step, or the same step and earlier position) did the same sender
have within the last `window` steps, and what fraction of the account's balance did they already move?

On clean PaySim almost no sender repeats (0.29% of rows), so these features carry no signal there. They exist to
counter structuring, where one fraud is split into several transfers from the same account: the pieces share a
sender and arrive close together. They only help when the training data contains such examples (see
scripts/run_structuring.py), because clean fraud rows always have velocity 0.
"""
from __future__ import annotations
import numpy as np

VELOCITY_WINDOW = 24          # steps (hours)
_BIG = 1 << 20                # larger than max step + window, keeps group keys apart


def sender_history(orig_id: np.ndarray, step: np.ndarray, amount: np.ndarray,
                   window: int = VELOCITY_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Return (prev_count, prev_outflow) per row, in the INPUT order. Row order within the same step is the
    input order. Only rows strictly earlier in (step, position) count; later rows never leak backwards."""
    n = len(orig_id)
    order = np.lexsort((np.arange(n), step, orig_id))
    oid, st, am = orig_id[order], step[order].astype(np.int64), amount[order].astype(np.float64)
    gid = np.cumsum(np.r_[True, oid[1:] != oid[:-1]]) - 1
    key = gid.astype(np.int64) * _BIG + st
    j0 = np.searchsorted(key, key - window, side="left")          # first row of the same sender within the window
    idx = np.arange(n)
    cs = np.r_[0.0, np.cumsum(am)]
    count_sorted, out_sorted = idx - j0, cs[idx] - cs[j0]
    prev_count, prev_out = np.empty(n), np.empty(n)
    prev_count[order], prev_out[order] = count_sorted, out_sorted
    return prev_count, prev_out


def velocity_features(df, window: int = VELOCITY_WINDOW) -> dict[str, np.ndarray]:
    prev_count, prev_out = sender_history(df["orig_id"].to_numpy(), df["step"].to_numpy(), df["amount"].to_numpy(), window)
    old_o = df["oldbalanceOrg"].to_numpy(dtype="float64")
    return {"orig_prev_count": prev_count,
            "orig_prev_ratio": prev_out / (old_o + prev_out + 1.0)}   # share of the balance (before those moves) already moved
