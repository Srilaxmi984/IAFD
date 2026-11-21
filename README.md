# IAFD: Imbalance-Aware Fraud Detection

Explainable, imbalance-aware fraud detection on the PaySim mobile-money dataset.
Base paper: Vijayanand & Smrithy (2025), *Explainable AI-enhanced ensemble learning for financial
fraud detection in mobile money transactions*, Intelligent Decision Technologies 19(1), 52-67.

## What this rebuild changes versus the paper
| Paper | This project |
|---|---|
| Random stratified 80/20 split | Chronological split by `step` (no future leakage) |
| Accuracy as the headline (99.9%) | PR-AUC, recall at a fixed false-positive cap, cost |
| Threshold implicitly 0.5 | Threshold chosen on validation only, under an FPR cap |
| Global SHAP on 5 raw columns | Local + global SHAP on engineered features |

## Status
- [x] Phase 1: repo, config, chunked data pipeline, chronological split, tests
- [x] Phase 2a: balance-error features, evaluation under an FPR cap, baseline models (A0, A)
- [ ] Phase 2b: velocity features (only if the data diagnostic shows they carry signal)
- [ ] Phase 3: cost-sensitive (B), velocity (C), combined (D), ensemble
- [ ] Phase 4: validation-only threshold at FPR cap, structuring defense (E)
- [ ] Phase 5: SHAP, Streamlit app, final results

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Data (kept local, never committed)
The PaySim CSV is ~470 MB, and GitHub blocks files over 100 MB, so `data/` is git-ignored.
1. Download PaySim from Kaggle and place the CSV in `data/raw/` (or pass `--csv <path>`).
2. Convert it once (streams in chunks, so it does not need much RAM):
   ```bash
   python scripts/prepare_data.py
   ```
   This writes `data/processed/paysim.parquet` (full) and `paysim_dev.parquet`
   (all fraud + 5% of legit rows, for fast development only).

Edit `configs/config.yaml` for paths, split steps and the FPR cap.

## Run the baselines (experiment A)
```bash
python scripts/diagnose_data.py                         # facts about the data (run once)
python scripts/run_baseline.py --dev                    # quick check on the dev sample
python scripts/run_baseline.py --feature-set paper      # A0: the paper's 5 raw features
python scripts/run_baseline.py --feature-set engineered # A : + balance-error features
```
Results are written to `reports/experiment_*.csv`. The threshold for each model is chosen on
validation at the FPR cap; test is evaluated once at that threshold.

## Tests
```bash
python -m pytest tests -q
```
Tests run on a small synthetic PaySim-like dataset, so they need no real data.
