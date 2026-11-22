# Explainable AI-Based Fraud Detection in Financial Transactions
## Using Ensemble Learning and SHAP: Advanced Implementation Report

---

## 1. Project Overview

This report documents the implementation of two independent, state-of-the-art approaches for financial fraud detection on the PaySim dataset, building upon the existing research foundation.

### 1.1 Research Objectives
- **Approach A**: Advanced Optimized Ensemble Learning — Combine complementary algorithms with stacking, weighted voting, calibration, and hyperparameter optimization
- **Approach B**: Specialized Imbalance-Aware Models — Handle severe class imbalance with EasyEnsemble, BalancedRandomForest, RUSBoost, cost-sensitive GBM, and resampling techniques
- **Fair Comparison**: Identical temporal splits, features, and evaluation methodology
- **Explainability**: Comprehensive SHAP analysis for both approaches
- **Robustness**: Evaluation against realistic fraud evasion patterns

### 1.2 Dataset
- **Source**: PaySim synthetic financial transaction dataset (Kaggle)
- **Size**: ~6.36M transactions, ~0.13% fraud rate
- **Split**: Chronological by `step` (1 step = 1 hour)
  - Train: steps 1–240
  - Validation: steps 241–355
  - Test: steps 356–743
- **Features**: 5 raw money columns + engineered balance-consistency features + velocity features

---

## 2. Existing Implementation Assessment

### 2.1 Strengths
- ✅ Rigorous temporal splitting prevents data leakage
- ✅ Multiple feature sets (paper, engineered, plus, velocity, no_dest)
- ✅ FPR-cap-based evaluation (not accuracy)
- ✅ SHAP integration with permutation explainer + TreeSHAP cross-check
- ✅ Modular, well-tested codebase with pytest
- ✅ Structuring attack evaluation (Experiment E)
- ✅ Cost-sensitive learning (Experiment B)
- ✅ Hard-negative mining (Experiment F)

### 2.2 Limitations Identified
| Limitation | Impact |
|------------|--------|
| Simple soft-voting ensemble only | Misses stacking, weighted voting, calibration benefits |
| Only cost-sensitive class weights for imbalance | No EasyEnsemble, BalancedRF, RUSBoost, SMOTE variants |
| Single threshold (FPR cap) | No PR-curve analysis, F2 optimization, high-recall operating points |
| Fixed hyperparameters | No bayesian optimization |
| Limited model zoo | Missing CatBoost, ExtraTrees, BalancedRF, StackingClassifier |
| Single robustness test (structuring) | No frequency bursts, amount anomalies, type changes, unseen patterns |

---

## 3. Approach A: Advanced Optimized Ensemble

### 3.1 Architecture
Implemented in `src/iafd/approach_a_ensemble.py` with three ensemble architectures:

1. **Soft Voting**: Equal-weight averaging of base model probabilities
2. **Weighted Voting**: Validation-optimized weights via Nelder-Mead optimization
3. **Stacking**: LogisticRegression meta-learner on base model predictions

### 3.2 Base Models
| Model | Type | Calibration | Key Parameters |
|-------|------|-------------|----------------|
| LogisticRegression | Linear | Isotonic | max_iter=2000, tol=1e-8 |
| RandomForest | Bagging | Isotonic | n_estimators=200, max_depth=16 |
| ExtraTrees | Bagging | Isotonic | n_estimators=200, max_depth=16 |
| BalancedRandomForest | Balanced Bagging | Isotonic | n_estimators=200, class_weight=balanced |
| XGBoost | Gradient Boosting | Isotonic | n_estimators=300, max_depth=6 |
| LightGBM | Gradient Boosting | Isotonic | n_estimators=300, num_leaves=63 |
| CatBoost | Gradient Boosting | Isotonic | iterations=300, depth=6 |

### 3.3 Key Innovations
- **Probability Calibration**: Isotonic regression on all tree-based models
- **Ensemble Weight Optimization**: Maximize validation recall at FPR cap
- **Diversity Analysis**: Correlation and disagreement matrices
- **Hyperparameter Optimization**: Optuna TPE sampler (30 trials/model)
- **Multiple Thresholds**: Default 0.5, F1-optimal, F2-optimal, MCC-optimal, FPR-cap validated

### 3.4 Training Script
```bash
python scripts/run_approach_a.py \
    --feature-set plus \
    --models rf xgb lgbm et \
    --architecture stacking \
    --calibrate \
    --optimize-weights \
    --pos-weight 1.0 \
    --fpr-cap 0.0001 \
    --seed 42
```

---

## 4. Approach B: Specialized Imbalance-Aware Models

### 4.1 Implemented Methods
Implemented in `src/iafd/approach_b_imbalance.py`:

| Category | Methods |
|----------|---------|
| **Ensemble (imbalanced-learn)** | EasyEnsemble, BalancedRandomForest, RUSBoost, BalancedBagging |
| **Cost-Sensitive GBM** | XGBoost (scale_pos_weight), LightGBM (class_weight), CatBoost (class_weights) |
| **Resampling + Model** | SMOTE/Borderline-SMOTE/ADASYN/SMOTE-Tomek + LR, RF, XGB |

### 4.2 Key Features
- **Auto-Selection**: Automatic model/hyperparameter selection on validation
- **No Pre-Split Resampling**: All resampling applied only within training partition
- **Calibration Support**: Isotonic calibration for probability reliability
- **Focal Loss Ready**: XGBoost focal loss implementation included

### 4.3 Training Script
```bash
python scripts/run_approach_b.py \
    --feature-set plus \
    --model balanced_rf \
    --calibrate \
    --pos-weight 1.0 \
    --resampling-strategy 0.5 \
    --auto-select \
    --fpr-cap 0.0001 \
    --seed 42
```

---

## 5. Evaluation Methodology

### 5.1 Metrics (All Required)
| Metric | Purpose |
|--------|---------|
| Precision | False alarm control |
| Recall (Fraud) | Detection rate |
| F1-Score | Harmonic mean |
| F2-Score | Recall-weighted |
| PR-AUC | Threshold-independent precision-recall |
| ROC-AUC | Threshold-independent ranking |
| MCC | Balanced correlation |
| FPR / FNR | Error rates |
| False Positives / Negatives | Absolute counts |

### 5.2 Threshold Strategy
1. **Validated FPR-cap threshold** (primary, from validation)
2. **Default 0.5** (baseline)
3. **F1-optimal** (balanced)
4. **F2-optimal** (recall-favoring)
5. **MCC-optimal** (correlation-maximizing)

### 5.3 Temporal Integrity
- Identical chronological splits for both approaches
- Preprocessing fit only on training data
- Resampling only on training partition
- Test set completely untouched until final evaluation

---

## 6. Comparison Framework

### 6.1 Comparison Script
```bash
python scripts/run_comparison.py \
    --approach-a models/approach_a_stacking_plus.joblib \
    --approach-b models/approach_b_balanced_rf_plus.joblib \
    --feature-set plus \
    --fpr-cap 0.0001
```

### 6.2 Outputs Generated
- `reports/approach_comparison_table.csv` — Required comparison table
- `reports/approach_comparison_detailed.json` — Full metrics at all thresholds
- `reports/approach_comparison_predictions.csv` — Per-transaction predictions
- `reports/figures/confusion_matrices.png` — Side-by-side confusion matrices
- `reports/figures/threshold_analysis.png` — Threshold vs metrics curves

### 6.3 Expected Comparison Table Format
| Metric | Advanced Ensemble | Imbalance-Aware Model |
|--------|-------------------|----------------------|
| Precision | | |
| Recall | | |
| F1-score | | |
| F2-score | | |
| PR-AUC | | |
| ROC-AUC | | |
| MCC | | |
| False Positive Rate | | |
| False Negative Rate | | |
| False Positives | | |
| False Negatives | | |
| Training Time | | |
| Inference Time | | |

---

## 7. SHAP Explainability

### 7.1 Implementation
Implemented in `src/iafd/shap_explain.py`:

- **Global**: Mean |SHAP| on fraud/legit rows, beeswarm plots, dependence plots
- **Local**: Waterfall explanations for caught fraud, missed fraud, false positives
- **Comparative**: Normalized importance comparison between approaches
- **Model-Agnostic**: Permutation explainer for ensembles, TreeSHAP for tree models

### 7.2 SHAP Script
```bash
python scripts/run_shap_both.py \
    --approach-a models/approach_a_stacking_plus.joblib \
    --approach-b models/approach_b_balanced_rf_plus.joblib \
    --feature-set plus \
    --n-fraud 300 --n-legit 300 \
    --background 100 --max-evals 400
```

### 7.3 Outputs
- `reports/shap_importance_approach_a_fraud.csv`
- `reports/shap_importance_approach_b_fraud.csv`
- `reports/shap_importance_comparison.csv`
- `reports/figures/shap/global_importance_approach_a_fraud.png`
- `reports/figures/shap/shap_summary_approach_a.png`
- `reports/figures/shap/dependence_approach_a/*.png`
- `reports/figures/shap/waterfall_Approach_A_caught_fraud.png`
- `reports/shap_local_explanations.json`

---

## 8. Robustness Testing

### 8.1 Attack Patterns Tested
Implemented in `src/iafd/robustness.py`:

| Category | Variants |
|----------|----------|
| **Structuring** | k=2,4,8,16,32 pieces; residual 20%; spread 0/12/48/96 steps |
| **Small Value** | max amounts 100/500/1000/5000; 2/5/10/20 transactions |
| **Frequency Burst** | burst sizes 2/5/10/20; windows 1/6/12/24 hours |
| **Amount Anomaly** | multipliers 0.01/0.1/0.5/2/5/10 |
| **Type Change** | CASH_IN, DEBIT, PAYMENT, TRANSFER, CASH_OUT |
| **Unseen Patterns** | Round amounts, threshold evasion, zero-balance sender, new receiver, rapid fire |

### 8.2 Evaluation Metric
**Event-level detection rate**: Fraction of original fraud events where ≥1 piece is flagged

### 8.3 Robustness Script
```bash
python scripts/run_robustness.py \
    --approach-a models/approach_a_stacking_plus.joblib \
    --approach-b models/approach_b_balanced_rf_plus.joblib \
    --feature-set plus \
    --fpr-cap 0.0001
```

### 8.4 Outputs
- `reports/robustness_results.csv` — Per-attack results
- `reports/robustness_summary_by_category.csv` — Average detection rate by category

---

## 9. Automated Testing

### 9.1 Test Suite
Implemented in `tests/test_approaches.py`:

| Test Class | Coverage |
|------------|----------|
| `TestDataProcessing` | Compact schema, chronological split, dev sample |
| `TestFeatureEngineering` | All feature sets, consistency, no leakage |
| `TestModelTraining` | All base models fit/predict, soft voting |
| `TestApproachA` | Ensemble creation, training, prediction, weight optimization, save/load |
| `TestApproachB` | Imbalance model creation, cost-sensitive GBM, save/load |
| `TestEvaluation` | Threshold at FPR, confusion matrix, Wilson interval |
| `TestTemporalSplitting` | No future in train, preprocessing fit on train only |
| `TestSHAP` | Predict functions, explanation cases |
| `TestIntegration` | Full pipelines, both approaches same data |

### 9.2 Run Tests
```bash
python -m pytest tests/test_approaches.py -v
python -m pytest tests/test_data.py -v
```

---

## 10. File Structure

```
IAFD/
├── configs/
│   └── config.yaml                 # Single source of truth
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       ├── paysim.parquet          # Full dataset (6.36M rows)
│       └── paysim_dev.parquet      # Dev sample (5% legit + all fraud)
├── models/
│   ├── .gitkeep
│   ├── ensemble.joblib             # Original ensemble
│   ├── approach_a_*.joblib         # Approach A models
│   ├── approach_b_*.joblib         # Approach B models
│   └── *_metadata.json             # Training metadata
├── reports/
│   ├── .gitkeep
│   ├── experiment_*.csv            # Original experiments
│   ├── shap_*.csv                  # SHAP results
│   ├── approach_comparison_*.csv   # Comparison tables
│   ├── robustness_*.csv            # Robustness results
│   └── figures/
│       ├── ensemble_global_bar.png
│       ├── xgb_beeswarm.png
│       ├── waterfall_*.png
│       ├── confusion_matrices.png
│       ├── threshold_analysis.png
│       └── shap/
│           ├── global_importance_*.png
│           ├── shap_summary_*.png
│           ├── dependence_*/*.png
│           └── waterfall_*.png
├── scripts/
│   ├── prepare_data.py             # CSV → Parquet + dev sample
│   ├── diagnose_data.py            # Data diagnostics
│   ├── run_baseline.py             # Original baseline experiments
│   ├── run_costsensitive.py        # Experiment B (cost-sensitive)
│   ├── run_hardneg.py              # Experiment F (hard negatives)
│   ├── run_shap.py                 # Original SHAP
│   ├── run_structuring.py          # Experiment E (structuring)
│   ├── error_analysis.py           # Validation error analysis
│   ├── run_approach_a.py           # NEW: Train Approach A
│   ├── run_approach_b.py           # NEW: Train Approach B
│   ├── run_comparison.py           # NEW: Compare approaches
│   ├── run_robustness.py           # NEW: Robustness testing
│   └── run_shap_both.py            # NEW: SHAP for both
├── src/
│   └── iafd/
│       ├── __init__.py             # Package exports
│       ├── config.py               # Config loading
│       ├── data.py                 # Data loading/splitting
│       ├── synthetic.py            # Synthetic data generator
│       ├── features.py             # Feature engineering
│       ├── models.py               # Base model zoo
│       ├── evaluate.py             # Evaluation metrics
│       ├── explain.py              # SHAP helpers
│       ├── velocity.py             # Velocity features
│       ├── structuring.py          # Structuring attacks
│       ├── hardneg.py              # Hard negative mining
│       ├── approach_a_ensemble.py  # NEW: Advanced ensemble
│       ├── approach_b_imbalance.py # NEW: Imbalance-aware models
│       ├── compare_approaches.py   # NEW: Comparison framework
│       ├── robustness.py           # NEW: Robustness testing
│       └── shap_explain.py         # NEW: SHAP for both
├── tests/
│   ├── conftest.py
│   ├── test_data.py                # Original tests
│   └── test_approaches.py          # NEW: Comprehensive tests
├── requirements.txt
└── RESEARCH_REPORT.md              # This file
```

---

## 11. Usage Instructions

### 11.1 Prerequisites
```bash
pip install -r requirements.txt
# Or manually:
pip install numpy pandas pyarrow pyyaml scikit-learn xgboost lightgbm catboost shap matplotlib joblib streamlit pytest imbalanced-learn optuna scipy
```

### 11.2 Data Preparation
```bash
# Download PaySim CSV from Kaggle to data/raw/
python scripts/prepare_data.py
# Or with custom path:
python scripts/prepare_data.py --csv /path/to/PS_20174392719_1491204439457_log.csv
```

### 11.3 Train Approach A (Advanced Ensemble)
```bash
# Default: stacking ensemble with RF, XGB, LGBM, ET
python scripts/run_approach_a.py

# Weighted voting with hyperparameter optimization
python scripts/run_approach_a.py --architecture weighted_voting --optimize-hyperparams --n-trials 50

# Soft voting with cost-sensitive learning
python scripts/run_approach_a.py --architecture soft_voting --pos-weight 10
```

### 11.4 Train Approach B (Imbalance-Aware)
```bash
# Default: BalancedRandomForest
python scripts/run_approach_b.py

# Cost-sensitive XGBoost with auto-selection
python scripts/run_approach_b.py --model cost_xgb --auto-select

# EasyEnsemble
python scripts/run_approach_b.py --model easy_ensemble

# SMOTE + XGBoost
python scripts/run_approach_b.py --model smote_xgb --resampling-strategy 0.5
```

### 11.5 Compare Approaches
```bash
python scripts/run_comparison.py \
    --approach-a models/approach_a_stacking_plus.joblib \
    --approach-b models/approach_b_balanced_rf_plus.joblib
```

### 11.6 Run SHAP Explanations
```bash
python scripts/run_shap_both.py \
    --approach-a models/approach_a_stacking_plus.joblib \
    --approach-b models/approach_b_balanced_rf_plus.joblib
```

### 11.7 Run Robustness Testing
```bash
python scripts/run_robustness.py \
    --approach-a models/approach_a_stacking_plus.joblib \
    --approach-b models/approach_b_balanced_rf_plus.joblib
```

### 11.8 Run All Tests
```bash
python -m pytest tests/ -v
```

---

## 12. Expected Research Contributions

### 12.1 Methodological Advances
1. **First comprehensive comparison** of advanced ensemble vs. specialized imbalance-aware models on PaySim with identical methodology
2. **Leakage-free temporal evaluation** with proper FPR-cap thresholds
3. **Multi-threshold analysis** beyond single FPR-cap operating point
4. **Comprehensive robustness taxonomy** covering 6 fraud evasion categories
5. **Dual-model SHAP explainability** with comparative feature importance

### 12.2 Practical Contributions
1. **Production-ready models** with save/load, metadata, and threshold persistence
2. **Clear winner determination** based on multiple metrics (not just recall)
3. **Operating point guidance** for different fraud team priorities
4. **Robustness certification** against known evasion techniques

---

## 13. Limitations & Future Work

### 13.1 Current Limitations
- CatBoost optional (import handled gracefully)
- Optuna hyperparameter optimization optional
- Full dataset training requires sufficient RAM (~8GB+)
- SHAP permutation explainer slow on large samples (use `--n-fraud 200 --n-legit 200`)

### 13.2 Future Improvements
1. **Neural network baselines** (TabNet, FT-Transformer)
2. **Online learning** for concept drift adaptation
3. **Adversarial training** against generated attacks
4. **Fairness analysis** across transaction types/amounts
5. **Deployment artifacts** (ONNX export, Triton/Docker configs)

---

## 14. Reproducibility

All experiments use fixed random seeds (default 42). Key reproducibility elements:
- `config.yaml` — Single source of truth for paths, splits, constraints
- `requirements.txt` — Pinned dependencies (run `pip freeze > requirements.lock.txt` after first success)
- Model metadata JSON — Training config, thresholds, metrics
- Git version control — All code and configs tracked

---

## 15. Conclusion

This implementation provides a scientifically rigorous, production-ready fraud detection system with:

- **Two independent, state-of-the-art approaches** addressing different aspects of the fraud detection challenge
- **Fair, leakage-free comparison** on identical data with comprehensive metrics
- **Full explainability** via SHAP for regulatory compliance and model debugging
- **Robustness validation** against realistic fraud evasion patterns
- **Automated testing** ensuring correctness of all pipeline components

The modular design allows easy extension with new models, features, attack patterns, and evaluation metrics while maintaining the rigorous experimental methodology established in the original research.

---

*Report generated as part of the IAFD research project. All code, configs, and results are in the repository at `C:\Users\srila\Downloads\IAFD\IAFD` and https://github.com/Srilaxmi984/IAFD*