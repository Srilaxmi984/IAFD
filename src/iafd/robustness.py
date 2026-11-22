"""Robustness Testing and Fraud Evasion Evaluation.

This module evaluates models against realistic fraud patterns:
- Transaction splitting (structuring)
- Multiple small-value fraudulent transactions
- Sudden transaction frequency increases
- Unusual transaction amounts
- Changes in transaction types
- Previously unseen transaction patterns
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

# Internal
from .config import load_config, resolve
from .data import load_dataset, chronological_split, check_split
from .features import build_features, FEATURE_SETS
from .approach_a_ensemble import AdvancedEnsemble
from .approach_b_imbalance import ImbalanceAwareModel
from .structuring import split_frauds, event_max
from .evaluate import threshold_at_fpr, confusion_at_threshold

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Attack Pattern Generators
# ============================================================

def generate_structuring_attacks(
    fraud_df: pd.DataFrame,
    rng: np.random.Generator,
    k_values: list[int] = [2, 4, 8, 16, 32],
    residual: float = 0.2,
    equal: bool = False,
    skew: float | None = None,
    spread: int = 0,
) -> dict[str, pd.DataFrame]:
    """Generate structuring (transaction splitting) attacks."""
    from .structuring import split_frauds

    attacks = {}
    for k in k_values:
        pieces = split_frauds(
            fraud_df,
            rng,
            [k],
            residual=residual,
            equal=equal,
            skew=skew,
            spread=spread,
        )
        attacks[f"structuring_k{k}_resid{residual}_spread{spread}"] = pieces
    return attacks


def generate_small_value_attacks(
    fraud_df: pd.DataFrame,
    rng: np.random.Generator,
    max_amounts: list[float] = [100, 500, 1000, 5000],
    num_transactions: list[int] = [2, 5, 10, 20],
) -> dict[str, pd.DataFrame]:
    """Generate multiple small-value transactions to evade amount-based detection."""
    attacks = {}

    for max_amt in max_amounts:
        for n_tx in num_transactions:
            pieces_list = []
            for _, fraud in fraud_df.iterrows():
                # Split the fraud amount into n_tx small pieces
                total_amt = fraud["amount"]
                if total_amt <= max_amt:
                    # Already small enough, keep as-is but repeat
                    pieces = pd.DataFrame([fraud] * n_tx)
                    pieces["amount"] = total_amt / n_tx
                else:
                    # Generate n_tx pieces that sum to total_amt, each <= max_amt
                    # Use Dirichlet distribution for realistic split
                    weights = rng.dirichlet(np.ones(n_tx))
                    amounts = weights * total_amt
                    # Cap at max_amt and redistribute excess
                    while np.any(amounts > max_amt):
                        excess = np.sum(np.maximum(amounts - max_amt, 0))
                        amounts = np.minimum(amounts, max_amt)
                        # Redistribute excess to non-capped amounts
                        non_capped = amounts < max_amt
                        if np.any(non_capped):
                            amounts[non_capped] += excess * (amounts[non_capped] / amounts[non_capped].sum())
                        else:
                            break

                    pieces_data = []
                    for i, amt in enumerate(amounts):
                        piece = fraud.copy()
                        piece["amount"] = amt
                        piece["oldbalanceOrg"] = fraud["oldbalanceOrg"] - np.sum(amounts[:i])
                        piece["newbalanceOrig"] = piece["oldbalanceOrg"] - amt
                        piece["oldbalanceDest"] = fraud["oldbalanceDest"] + np.sum(amounts[:i])
                        piece["newbalanceDest"] = piece["oldbalanceDest"] + amt
                        piece["event_id"] = fraud.name * n_tx + i
                        pieces_data.append(piece)
                    pieces = pd.DataFrame(pieces_data)

                pieces_list.append(pieces)

            if pieces_list:
                attacks[f"small_value_max{max_amt}_n{n_tx}"] = pd.concat(pieces_list, ignore_index=True)

    return attacks


def generate_frequency_burst_attacks(
    fraud_df: pd.DataFrame,
    rng: np.random.Generator,
    burst_sizes: list[int] = [2, 5, 10, 20],
    time_windows: list[int] = [1, 6, 12, 24],  # hours
) -> dict[str, pd.DataFrame]:
    """Generate sudden frequency increase attacks (velocity evasion)."""
    attacks = {}

    for burst_size in burst_sizes:
        for window in time_windows:
            pieces_list = []
            for _, fraud in fraud_df.iterrows():
                # Create burst of transactions in short time window
                pieces_data = []
                base_step = fraud["step"]
                amount_per_tx = fraud["amount"] / burst_size

                for i in range(burst_size):
                    piece = fraud.copy()
                    piece["amount"] = amount_per_tx
                    piece["step"] = base_step + int(i * window / burst_size)
                    piece["oldbalanceOrg"] = fraud["oldbalanceOrg"] - i * amount_per_tx
                    piece["newbalanceOrig"] = piece["oldbalanceOrg"] - amount_per_tx
                    piece["oldbalanceDest"] = fraud["oldbalanceDest"] + i * amount_per_tx
                    piece["newbalanceDest"] = piece["oldbalanceDest"] + amount_per_tx
                    piece["event_id"] = fraud.name * burst_size + i
                    pieces_data.append(piece)

                pieces_list.append(pd.DataFrame(pieces_data))

            if pieces_list:
                attacks[f"frequency_burst_n{burst_size}_window{window}h"] = pd.concat(pieces_list, ignore_index=True)

    return attacks


def generate_amount_anomaly_attacks(
    fraud_df: pd.DataFrame,
    rng: np.random.Generator,
    multipliers: list[float] = [0.01, 0.1, 0.5, 2.0, 5.0, 10.0],
) -> dict[str, pd.DataFrame]:
    """Generate unusual transaction amount attacks."""
    attacks = {}

    for mult in multipliers:
        pieces_list = []
        for _, fraud in fraud_df.iterrows():
            piece = fraud.copy()
            original_amt = fraud["amount"]
            new_amt = original_amt * mult

            # Adjust balances
            if mult < 1.0:
                # Smaller amount - sender retains more balance
                piece["amount"] = new_amt
                piece["newbalanceOrig"] = fraud["oldbalanceOrg"] - new_amt
                piece["newbalanceDest"] = fraud["oldbalanceDest"] + new_amt
            else:
                # Larger amount - may exceed sender balance
                piece["amount"] = new_amt
                piece["newbalanceOrig"] = max(0, fraud["oldbalanceOrg"] - new_amt)
                piece["newbalanceDest"] = fraud["oldbalanceDest"] + new_amt

            piece["event_id"] = fraud.name
            pieces_list.append(pd.DataFrame([piece]))

        if pieces_list:
            attacks[f"amount_anomaly_mult{mult}"] = pd.concat(pieces_list, ignore_index=True)

    return attacks


def generate_type_change_attacks(
    fraud_df: pd.DataFrame,
    rng: np.random.Generator,
    target_types: list[str] = ["CASH_IN", "DEBIT", "PAYMENT", "TRANSFER", "CASH_OUT"],
) -> dict[str, pd.DataFrame]:
    """Generate transaction type change attacks."""
    attacks = {}

    # Get original type distribution for fraud
    orig_types = fraud_df["type"].unique()

    for target_type in target_types:
        if target_type in orig_types:
            continue  # Already present in fraud

        pieces_list = []
        for _, fraud in fraud_df.iterrows():
            piece = fraud.copy()
            piece["type"] = target_type
            piece["event_id"] = fraud.name
            pieces_list.append(pd.DataFrame([piece]))

        if pieces_list:
            attacks[f"type_change_to_{target_type}"] = pd.concat(pieces_list, ignore_index=True)

    return attacks


def generate_unseen_pattern_attacks(
    fraud_df: pd.DataFrame,
    rng: np.random.Generator,
    n_patterns: int = 5,
) -> dict[str, pd.DataFrame]:
    """Generate previously unseen transaction patterns."""
    attacks = {}

    # Pattern 1: Round-number transactions (suspicious)
    pattern1 = fraud_df.copy()
    pattern1["amount"] = np.round(pattern1["amount"] / 1000) * 1000
    pattern1["event_id"] = pattern1.index
    attacks["unseen_round_amounts"] = pattern1

    # Pattern 2: Just-below-reporting-threshold (e.g., $9999)
    pattern2 = fraud_df.copy()
    pattern2["amount"] = 9999.00
    pattern2["newbalanceOrig"] = pattern2["oldbalanceOrg"] - 9999.00
    pattern2["newbalanceDest"] = pattern2["oldbalanceDest"] + 9999.00
    pattern2["event_id"] = pattern2.index
    attacks["unseen_threshold_evasion"] = pattern2

    # Pattern 3: Zero-balance sender (insider threat)
    pattern3 = fraud_df.copy()
    pattern3["oldbalanceOrg"] = 0
    pattern3["newbalanceOrig"] = 0
    pattern3["amount"] = pattern3["oldbalanceDest"] * 0.5  # Use receiver's balance as reference
    pattern3["event_id"] = pattern3.index
    attacks["unseen_zero_balance_sender"] = pattern3

    # Pattern 4: New receiver with zero history
    pattern4 = fraud_df.copy()
    pattern4["oldbalanceDest"] = 0
    pattern4["newbalanceDest"] = pattern4["amount"]
    pattern4["dest_id"] = rng.integers(10**15, 10**16, len(pattern4))
    pattern4["event_id"] = pattern4.index
    attacks["unseen_new_receiver"] = pattern4

    # Pattern 5: Rapid fire (same second)
    pattern5_list = []
    for _, fraud in fraud_df.iterrows():
        for i in range(3):
            piece = fraud.copy()
            piece["amount"] = fraud["amount"] / 3
            piece["step"] = fraud["step"]  # Same step
            piece["oldbalanceOrg"] = fraud["oldbalanceOrg"] - i * piece["amount"]
            piece["newbalanceOrig"] = piece["oldbalanceOrg"] - piece["amount"]
            piece["oldbalanceDest"] = fraud["oldbalanceDest"] + i * piece["amount"]
            piece["newbalanceDest"] = piece["oldbalanceDest"] + piece["amount"]
            piece["event_id"] = fraud.name * 3 + i
            pattern5_list.append(piece)
    attacks["unseen_rapid_fire"] = pd.DataFrame(pattern5_list)

    return attacks


# ============================================================
# Attack Evaluation
# ============================================================

def evaluate_attack(
    model,
    X_attack: pd.DataFrame,
    attack_df: pd.DataFrame,
    threshold: float,
    original_fraud_count: int,
) -> dict:
    """Evaluate model on attack data."""
    # Get predictions
    scores = model.predict_proba(X_attack)[:, 1]

    # For structuring attacks, use event_max
    if "event_id" in attack_df.columns:
        event_ids = attack_df["event_id"].to_numpy()
        event_scores = event_max(scores, event_ids)
        # An event is detected if ANY piece is flagged
        detected = event_scores > threshold
        n_detected = np.sum(detected)
        detection_rate = n_detected / original_fraud_count if original_fraud_count > 0 else 0
    else:
        # Direct detection
        detected = scores > threshold
        n_detected = np.sum(detected)
        detection_rate = n_detected / len(scores) if len(scores) > 0 else 0

    return {
        "detection_rate": detection_rate,
        "n_detected": int(n_detected),
        "n_total_events": original_fraud_count,
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
    }


def run_robustness_evaluation(
    model_a: AdvancedEnsemble,
    model_b: ImbalanceAwareModel,
    feature_set: str = "plus",
    fpr_cap: float = 0.0001,
    use_dev: bool = False,
    seed: int = 42,
) -> dict:
    """Run comprehensive robustness evaluation on both models."""
    cfg = load_config()

    # Load data
    path = resolve(cfg["paths"]["dev_parquet" if use_dev else "parquet"])
    df = load_dataset(path)
    s = cfg["split"]
    train, val, test = chronological_split(df, s["train_end_step"], s["val_end_step"])

    # Get test frauds
    test_fraud = test[test["isFraud"] == 1].reset_index(drop=True)
    n_fraud = len(test_fraud)

    if n_fraud == 0:
        raise ValueError("No fraud in test set")

    print(f"Test fraud events: {n_fraud}")

    rng = np.random.default_rng(seed)

    # Generate all attacks
    print("Generating attack patterns...")
    all_attacks = {}

    # Structuring attacks
    all_attacks.update(generate_structuring_attacks(test_fraud, rng))

    # Small value attacks
    all_attacks.update(generate_small_value_attacks(test_fraud, rng))

    # Frequency burst attacks
    all_attacks.update(generate_frequency_burst_attacks(test_fraud, rng))

    # Amount anomaly attacks
    all_attacks.update(generate_amount_anomaly_attacks(test_fraud, rng))

    # Type change attacks
    all_attacks.update(generate_type_change_attacks(test_fraud, rng))

    # Unseen pattern attacks
    all_attacks.update(generate_unseen_pattern_attacks(test_fraud, rng))

    print(f"Generated {len(all_attacks)} attack variants")

    # Evaluate each attack on both models
    results = {"Approach A": {}, "Approach B": {}}

    for attack_name, attack_df in all_attacks.items():
        print(f"  Evaluating {attack_name}...")
        X_attack = build_features(attack_df, feature_set)

        # Model A
        res_a = evaluate_attack(model_a, X_attack, attack_df, model_a.threshold, n_fraud)
        results["Approach A"][attack_name] = res_a

        # Model B
        res_b = evaluate_attack(model_b, X_attack, attack_df, model_b.threshold, n_fraud)
        results["Approach B"][attack_name] = res_b

    return results


def generate_robustness_report(
    results: dict,
    save_dir: Path,
) -> pd.DataFrame:
    """Generate robustness comparison report."""
    rows = []

    for approach_name, attack_results in results.items():
        for attack_name, metrics in attack_results.items():
            row = {
                "Approach": approach_name,
                "Attack": attack_name,
                "Detection Rate": round(metrics["detection_rate"], 4),
                "Detected Events": metrics["n_detected"],
                "Total Events": metrics["n_total_events"],
                "Mean Score": round(metrics["mean_score"], 4),
                "Max Score": round(metrics["max_score"], 4),
                "Min Score": round(metrics["min_score"], 4),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "robustness_results.csv", index=False)

    # Summary table: average detection rate by attack category
    df["Category"] = df["Attack"].apply(lambda x: x.split("_")[0] if "_" in x else x)
    summary = df.groupby(["Approach", "Category"])["Detection Rate"].mean().unstack()

    summary.to_csv(save_dir / "robustness_summary_by_category.csv")
    print("\n=== ROBUSTNESS SUMMARY BY CATEGORY ===")
    print(summary.round(4).to_string())

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run robustness testing")
    parser.add_argument("--approach-a", required=True, help="Path to Approach A model")
    parser.add_argument("--approach-b", required=True, help="Path to Approach B model")
    parser.add_argument("--feature-set", default="plus", choices=list(FEATURE_SETS))
    parser.add_argument("--fpr-cap", type=float, default=0.0001)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load models
    model_a = AdvancedEnsemble.load(args.approach_a)
    model_b = ImbalanceAwareModel.load(args.approach_b)

    # Run evaluation
    results = run_robustness_evaluation(
        model_a, model_b,
        feature_set=args.feature_set,
        fpr_cap=args.fpr_cap,
        use_dev=args.dev,
        seed=args.seed,
    )

    # Save results
    cfg = load_config()
    out_dir = resolve(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_robustness_report(results, out_dir)
    print(f"\nResults saved to {out_dir}")