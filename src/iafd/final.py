"""Helpers for the two FINAL models (best ensemble, best single model under high class imbalance).

Selection uses VALIDATION data only. The main selection metric is LFR, the mean recall over a grid of low
false-positive-rate caps: it rewards catching fraud at EVERY false-alarm budget at once, so a lucky single cap
cannot decide the winner. Reporting adds counts of missed frauds (FN) and false alarms (FP), MCC, F2, PR-AUC and
a cost view (value of missed fraud plus a fixed cost per false alert).
"""
from __future__ import annotations
import itertools
import numpy as np
from .evaluate import confusion_at_threshold
from .explain import logit

LOW_FPR_CAPS = (1e-5, 5e-5, 1e-4, 5e-4, 1e-3)


def lfr_score(y: np.ndarray, p: np.ndarray, caps=LOW_FPR_CAPS) -> float:
    """Mean recall over FPR caps, thresholds chosen on the same data (selection use on validation)."""
    neg = np.sort(p[y == 0])[::-1]
    pos = np.sort(p[y == 1])
    if len(neg) == 0 or len(pos) == 0:
        raise ValueError("need both classes")
    recs = []
    for c in caps:
        k = int(np.floor(c * len(neg)))
        thr = neg[k] if k < len(neg) else -np.inf
        recs.append((len(pos) - np.searchsorted(pos, thr, side="right")) / len(pos))
    return float(np.mean(recs))


def mcc(tp, fp, fn, tn) -> float:
    tp, fp, fn, tn = float(tp), float(fp), float(fn), float(tn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def f_beta(tp, fp, fn, beta: float = 2.0) -> float:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    return float((1 + b2) * p * r / (b2 * p + r)) if (b2 * p + r) > 0 else 0.0


def cost_at_threshold(y, p, amount, thr, alert_cost: float) -> dict:
    """Cost = value of the frauds we MISS + alert_cost for every false alert. baseline = flag nothing."""
    pred = p > thr
    missed = float(amount[(~pred) & (y == 1)].sum())
    fp = int((pred & (y == 0)).sum())
    total = missed + alert_cost * fp
    baseline = float(amount[y == 1].sum())
    return {"missed_value": missed, "alert_cost_total": alert_cost * fp, "total_cost": total,
            "baseline_cost": baseline, "saving_vs_no_model": baseline - total}


def threshold_min_cost(y, p, amount, alert_cost: float) -> float:
    """Threshold minimising (value of missed frauds + alert_cost * false alerts) on the given data.
    Only cuts between distinct scores are allowed, so it is exact with ties. +inf means 'flag nothing'."""
    order = np.argsort(-p, kind="stable")
    ps, ys, am = p[order], y[order], amount[order]
    fraud_amt = np.where(ys == 1, am, 0.0)
    cum_fraud, cum_fp = np.cumsum(fraud_amt), np.cumsum((ys == 0).astype(float))
    total = float(fraud_amt.sum())
    valid = np.r_[ps[:-1] != ps[1:], True]
    cost = np.where(valid, (total - cum_fraud) + alert_cost * cum_fp, np.inf)
    i = int(np.argmin(cost))
    if cost[i] >= total:
        return float("inf")
    return float(np.nextafter(ps[i], -np.inf))


def fraud_subsample_mask(y: np.ndarray, keep_fraction: float, seed: int) -> np.ndarray:
    """Rows to keep when only `keep_fraction` of the fraud rows are retained (all legit rows stay): a more
    extreme class imbalance in TRAIN, with validation and test unchanged."""
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    keep = np.ones(len(y), dtype=bool)
    if keep_fraction < 1:
        fr = np.flatnonzero(y == 1)
        drop = np.random.default_rng(seed).choice(fr, size=int(round(len(fr) * (1 - keep_fraction))), replace=False)
        keep[drop] = False
    return keep


def strategy_pos_weight(y: np.ndarray, strategy: str) -> float:
    """'plain' = no weighting; 'sqrtw' = fraud weighted by sqrt(neg/pos) (a milder reweighting than full balancing)."""
    if strategy == "plain":
        return 1.0
    if strategy == "sqrtw":
        return float(np.sqrt((y == 0).sum() / max((y == 1).sum(), 1)))
    raise ValueError(f"unknown strategy {strategy!r}")


def combine(prob_list, mode: str = "mean") -> np.ndarray:
    P = np.vstack(prob_list)
    if mode == "mean":
        return P.mean(axis=0)
    if mode == "logit_mean":
        return 1.0 / (1.0 + np.exp(-logit(P).mean(axis=0)))
    raise ValueError(f"unknown aggregator {mode!r}")


def all_subsets(names, min_size: int = 2):
    for r in range(min_size, len(names) + 1):
        yield from itertools.combinations(names, r)
