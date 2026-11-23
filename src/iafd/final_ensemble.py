"""Final ensemble: a separate, leaner alternative to Approach A.

What changes compared with ``approach_a_ensemble.AdvancedEnsemble``
-------------------------------------------------------------------
1. Strong members only (default: rf, xgb, lgbm). No LR / DT, which diluted the
   vote in the baseline run (the 5-model ensemble had 11 FP vs 6 for RF alone).
2. No isotonic calibration. Isotonic creates flat score plateaus, i.e. many legit
   rows tie exactly at the threshold (see the "81 legit val rows tie" warning),
   which breaks a tight FPR cap. Members are combined in logit space instead,
   which keeps resolution at the extreme tail where a 1e-4 FPR cap lives.
3. Weights come from a validation grid search on PR-AUC (a smooth objective).
   The old Nelder-Mead on "recall at FPR cap" is piecewise constant and stalls
   at the equal-weight start point.
4. Type gating learned from TRAIN only: transaction types with no train fraud
   score exactly 0 and are not used to fit the members. They still count in
   every FPR denominator, so evaluation stays honest.
5. Threshold chosen on validation only (existing ``threshold_at_fpr``); test is
   evaluated once at that threshold.

Run:
    python scripts/run_final_ensemble.py --dev
    python scripts/run_final_ensemble.py
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from .config import load_config, resolve
from .data import check_split, chronological_split, load_dataset
from .evaluate import confusion_at_threshold, evaluate_scores, threshold_at_fpr
from .features import FEATURE_SETS, TYPE_COLS, build_features

warnings.filterwarnings("ignore", category=UserWarning)

_EPS = 1e-6
DEFAULT_MEMBERS = ["rf", "xgb", "lgbm"]
DEFAULT_CAPS = (1e-5, 5e-5, 1e-4, 5e-4, 1e-3)


# ------------------------------------------------------------------ members
def make_member(name: str, seed: int, pos_weight: float = 1.0, n_estimators: int | None = None):
    """Same hyper-parameters as ENSEMBLE_BASE_MODELS in approach_a_ensemble."""
    cw = {0: 1.0, 1: pos_weight} if pos_weight != 1.0 else None
    if name == "rf":
        return RandomForestClassifier(
            random_state=seed, n_estimators=n_estimators or 200, max_depth=16,
            min_samples_leaf=5, class_weight=cw, n_jobs=-1)
    if name == "et":
        return ExtraTreesClassifier(
            random_state=seed, n_estimators=n_estimators or 200, max_depth=16,
            min_samples_leaf=5, class_weight=cw, n_jobs=-1)
    if name == "xgb":
        return XGBClassifier(
            random_state=seed, n_estimators=n_estimators or 300, max_depth=6,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, tree_method="hist", n_jobs=-1,
            eval_metric="logloss", verbosity=0)
    if name == "lgbm":
        return LGBMClassifier(
            random_state=seed, n_estimators=n_estimators or 300, num_leaves=63,
            learning_rate=0.1, min_child_weight=1.0, max_delta_step=1.0,
            class_weight=cw, n_jobs=-1, verbose=-1)
    raise ValueError(f"Unknown member {name!r}; choose from rf, et, xgb, lgbm")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------- ensemble
class FinalEnsemble:
    def __init__(
        self,
        feature_set: str = "plus",
        members: list[str] | None = None,
        combine: str = "logit",
        gate: bool = True,
        pos_weight: float = 1.0,
        seed: int = 42,
        weight_step: float = 0.1,
        n_estimators: int | None = None,
    ):
        if combine not in ("logit", "prob"):
            raise ValueError("combine must be 'logit' or 'prob'")
        self.feature_set = feature_set
        self.member_names = list(members or DEFAULT_MEMBERS)
        self.combine = combine
        self.gate = gate
        self.pos_weight = pos_weight
        self.seed = seed
        self.weight_step = weight_step
        self.n_estimators = n_estimators

        self.models: dict = {}
        self.weights: np.ndarray | None = None
        self.threshold: float | None = None
        self.columns: list[str] | None = None
        self.gate_cols: list[str] = []
        self.metadata: dict = {}

    # ---- helpers
    def _prep(self, p: np.ndarray) -> np.ndarray:
        return _logit(p) if self.combine == "logit" else p

    def _gate_mask(self, X: pd.DataFrame) -> np.ndarray:
        if not self.gate_cols:
            return np.ones(len(X), dtype=bool)
        return X[self.gate_cols].to_numpy().sum(axis=1) > 0

    def _member_probas(self, X: pd.DataFrame) -> tuple[dict, np.ndarray]:
        """Member probabilities for every row; gated-out rows get exactly 0."""
        X = X[self.columns]
        gate = self._gate_mask(X)
        out = {}
        for name in self.member_names:
            p = np.zeros(len(X))
            if gate.any():
                p[gate] = self.models[name].predict_proba(X[gate])[:, 1]
            out[name] = p
        return out, gate

    def _combine_scores(self, probas: dict, gate: np.ndarray, weights=None) -> np.ndarray:
        w = self.weights if weights is None else np.asarray(weights, dtype=float)
        z = sum(wi * self._prep(probas[n]) for wi, n in zip(w, self.member_names))
        s = _sigmoid(z) if self.combine == "logit" else z
        return np.where(gate, s, 0.0)

    def _fit_weights(self, val_probas: dict, y_val: np.ndarray, gate: np.ndarray) -> np.ndarray:
        """Grid search on the weight simplex, maximising validation PR-AUC.

        Gated-out rows are all legit and sit at the very bottom of the ranking, so
        PR-AUC is computed on the gated subset only (same value, much faster).
        Ties are broken towards equal weights.
        """
        k = len(self.member_names)
        eq = np.full(k, 1.0 / k)
        if k == 1:
            return np.array([1.0])
        yg = y_val[gate]
        if yg.sum() == 0 or yg.sum() == len(yg):
            return eq
        prepped = np.column_stack([self._prep(val_probas[n][gate]) for n in self.member_names])
        step = self.weight_step if k <= 4 else 0.25
        grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)

        def key(w):
            ap = average_precision_score(yg, prepped @ w)
            return (round(float(ap), 6), -float(np.linalg.norm(w - eq)))

        best_w, best_key = eq, key(eq)
        for combo in itertools.product(grid, repeat=k - 1):
            s = float(sum(combo))
            if s > 1.0 + 1e-9:
                continue
            w = np.array(list(combo) + [1.0 - s])
            kk = key(w)
            if kk > best_key:
                best_w, best_key = w, kk
        return best_w

    # ---- API
    def fit(self, X_train, y_train, X_val, y_val, fpr_cap: float = 1e-4) -> "FinalEnsemble":
        t0 = time.time()
        self.columns = list(X_train.columns)
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)

        # type gating, learned from train only
        self.gate_cols = []
        if self.gate:
            for c in TYPE_COLS:
                if c in X_train.columns and X_train[c].to_numpy()[y_train == 1].sum() > 0:
                    self.gate_cols.append(c)
        g_tr = self._gate_mask(X_train)
        if y_train[g_tr].sum() == 0:
            raise ValueError("No fraud rows left in the training set after gating.")
        print(f"Gate types (fraud seen in train): {self.gate_cols or 'off'} | "
              f"train rows used: {g_tr.sum():,} of {len(g_tr):,}")

        Xg, yg = X_train[g_tr], y_train[g_tr]
        for name in self.member_names:
            t1 = time.time()
            m = make_member(name, self.seed, self.pos_weight, self.n_estimators)
            m.fit(Xg, yg)
            self.models[name] = m
            print(f"  {name}: trained in {time.time() - t1:.1f}s")

        val_probas, g_va = self._member_probas(X_val)
        missed = int(((y_val == 1) & ~g_va).sum())
        self.weights = self._fit_weights(val_probas, y_val, g_va)
        print("Weights:", {n: round(float(w), 3) for n, w in zip(self.member_names, self.weights)})

        val_scores = self._combine_scores(val_probas, g_va)
        self.threshold = float(threshold_at_fpr(y_val, val_scores, fpr_cap))
        c = confusion_at_threshold(y_val, val_scores, self.threshold)
        print(f"Validation: thr={self.threshold:.6f} recall={c['recall']:.4f} "
              f"FPR={c['fpr']:.6f} FP={c['fp']} | frauds lost to gate: {missed}")

        self.metadata = {
            "fpr_cap": fpr_cap, "combine": self.combine,
            "weights": self.weights.tolist(), "threshold": self.threshold,
            "gate_cols": self.gate_cols, "val_frauds_lost_to_gate": missed,
            "train_seconds": time.time() - t0,
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas, gate = self._member_probas(X)
        return self._combine_scores(probas, gate)

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        thr = self.threshold if threshold is None else threshold
        return (self.predict_proba(X) > thr).astype(int)

    def evaluate(self, X, y, threshold=None, reference_prevalence=None) -> dict:
        thr = self.threshold if threshold is None else threshold
        return evaluate_scores(np.asarray(y), self.predict_proba(X), thr, reference_prevalence)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "models": self.models, "weights": self.weights, "threshold": self.threshold,
            "columns": self.columns, "gate_cols": self.gate_cols, "metadata": self.metadata,
            "init": dict(feature_set=self.feature_set, members=self.member_names,
                         combine=self.combine, gate=self.gate, pos_weight=self.pos_weight,
                         seed=self.seed, weight_step=self.weight_step,
                         n_estimators=self.n_estimators),
        }, path)

    @classmethod
    def load(cls, path) -> "FinalEnsemble":
        d = joblib.load(path)
        ens = cls(**d["init"])
        ens.models, ens.weights, ens.threshold = d["models"], d["weights"], d["threshold"]
        ens.columns, ens.gate_cols, ens.metadata = d["columns"], d["gate_cols"], d["metadata"]
        return ens


# --------------------------------------------------------------- comparison
def compare_at_caps(ens: FinalEnsemble, Xva, yva, Xte, yte, caps=DEFAULT_CAPS) -> pd.DataFrame:
    """Each member, an equal-weight blend and FINAL, at several FPR caps.

    Every row uses a threshold chosen on validation and is evaluated once on test.
    """
    pv, gv = ens._member_probas(Xva)
    pt, gt = ens._member_probas(Xte)
    yva, yte = np.asarray(yva), np.asarray(yte)
    k = len(ens.member_names)
    cands = {n: (pv[n], pt[n]) for n in ens.member_names}
    eq = np.full(k, 1.0 / k)
    cands["equal_weights"] = (ens._combine_scores(pv, gv, eq), ens._combine_scores(pt, gt, eq))
    cands["FINAL"] = (ens._combine_scores(pv, gv), ens._combine_scores(pt, gt))
    rows = []
    for name, (sv, st) in cands.items():
        for cap in caps:
            thr = threshold_at_fpr(yva, sv, cap)
            c = confusion_at_threshold(yte, st, thr)
            rows.append({"model": name, "fpr_cap": cap, "recall": c["recall"],
                         "fp": c["fp"], "fpr": c["fpr"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- pipeline
def train_final(
    feature_set: str = "plus",
    members: list[str] | None = None,
    combine: str = "logit",
    gate: bool = True,
    pos_weight: float = 1.0,
    fpr_cap: float = 1e-4,
    seed: int = 42,
    use_dev: bool = False,
):
    cfg = load_config()
    path = resolve(cfg["paths"]["dev_parquet" if use_dev else "parquet"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/prepare_data.py first.")

    df = load_dataset(path)
    s = cfg["split"]
    train, val, test = chronological_split(df, s["train_end_step"], s["val_end_step"])
    print("Split:", check_split(train, val, test))

    Xtr, Xva, Xte = (build_features(d, feature_set) for d in (train, val, test))
    ytr, yva, yte = (d["isFraud"].to_numpy() for d in (train, val, test))

    ens = FinalEnsemble(feature_set, members, combine, gate, pos_weight, seed)
    ens.fit(Xtr, ytr, Xva, yva, fpr_cap=fpr_cap)

    print("\n=== Test (threshold fixed on validation) ===")
    res = ens.evaluate(Xte, yte, reference_prevalence=ytr.mean())
    print(f"Recall {res['recall']:.4f} | FPR {res['fpr']:.6f} | Precision {res['precision']:.4f} | "
          f"PR-AUC {res['pr_auc']:.4f}")
    print(f"TP {res['tp']} FP {res['fp']} FN {res['fn']} TN {res['tn']}")

    comp = compare_at_caps(ens, Xva, yva, Xte, yte)
    print("\nTEST recall by FPR cap (thresholds chosen on validation):")
    print(comp.pivot(index="model", columns="fpr_cap", values="recall").round(4).to_string())
    print("\nTEST false positives by FPR cap:")
    print(comp.pivot(index="model", columns="fpr_cap", values="fp").to_string())

    models_dir = resolve(cfg["paths"]["models_dir"])
    reports_dir = resolve(cfg["paths"].get("reports_dir", "reports"))
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tag = "dev_" if use_dev else ""
    ens.save(models_dir / f"final_ensemble_{tag}{feature_set}.joblib")
    comp.to_csv(reports_dir / f"final_ensemble_{tag}comparison.csv", index=False)
    meta = {"feature_set": feature_set, "members": ens.member_names, "test": res,
            "ensemble": ens.metadata}
    with open(reports_dir / f"final_ensemble_{tag}metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return ens, meta, comp


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train the final ensemble")
    ap.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    ap.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS)
    ap.add_argument("--combine", default="logit", choices=["logit", "prob"])
    ap.add_argument("--no-gate", dest="gate", action="store_false")
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--fpr-cap", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    train_final(a.feature_set, a.members, a.combine, a.gate, a.pos_weight,
                a.fpr_cap, a.seed, a.dev)


if __name__ == "__main__":
    main()
