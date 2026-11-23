# Final results (test set, thresholds chosen on validation, 1 seed(s))

Final ensemble: rf(plain)+xgb(plain)+lgbm(sqrtw) [mean ]; final single model: rf (sqrtw); features: plus; type gating: on (CASH_OUT, TRANSFER). Paper baseline: raw 5 features, LR+DT+RF+XGB+LightGBM soft vote. Alert cost per false alarm: 10000.


## Operating point: FPR <= 0.01%

| system | missed (FN) | false alarms (FP) | recall | FPR | MCC | F2 | PR-AUC | LFR (test) | cost saving |
|---|---|---|---|---|---|---|---|---|---|
| final_ensemble | 0 | 15 | 1.0000 | 0.00001 | 0.998 | 0.999 | 1.0000 | 1.0000 | 6,682,731,544 |
| final_single | 0 | 40 | 1.0000 | 0.00003 | 0.995 | 0.998 | 1.0000 | 1.0000 | 6,682,481,544 |
| paper_ensemble | 1184 | 96 | 0.7214 | 0.00008 | 0.836 | 0.760 | 0.9375 | 0.7494 | 6,526,893,047 |


## Operating point: recall >= 99.5%

| system | missed (FN) | false alarms (FP) | recall | FPR | MCC | F2 | PR-AUC | LFR (test) | cost saving |
|---|---|---|---|---|---|---|---|---|---|
| final_ensemble | 44 | 0 | 0.9896 | 0.00000 | 0.995 | 0.992 | 1.0000 | 1.0000 | 6,431,929,500 |
| final_single | 45 | 0 | 0.9894 | 0.00000 | 0.995 | 0.992 | 1.0000 | 1.0000 | 6,511,919,627 |
| paper_ensemble | 16 | 42344 | 0.9962 | 0.03403 | 0.296 | 0.333 | 0.9375 | 0.7494 | 6,259,033,556 |


## Operating point: minimum cost

| system | missed (FN) | false alarms (FP) | recall | FPR | MCC | F2 | PR-AUC | LFR (test) | cost saving |
|---|---|---|---|---|---|---|---|---|---|
| final_ensemble | 20 | 0 | 0.9953 | 0.00000 | 0.998 | 0.996 | 1.0000 | 1.0000 | 6,635,227,330 |
| final_single | 11 | 0 | 0.9974 | 0.00000 | 0.999 | 0.998 | 1.0000 | 1.0000 | 6,682,128,625 |
| paper_ensemble | 587 | 617 | 0.8619 | 0.00050 | 0.858 | 0.861 | 0.9375 | 0.7494 | 6,626,131,993 |


## Early and late test regimes at FPR <= 0.01% (first seed)

| system | early recall | early FP | late recall | late FP | recall 95% CI |
|---|---|---|---|---|---|
| final_ensemble | 1.0000 | 14 | 1.0000 | 1 | [0.9991, 1.0000] |
| final_single | 1.0000 | 33 | 1.0000 | 7 | [0.9991, 1.0000] |
| paper_ensemble | 0.7422 | 74 | 0.7089 | 22 | [0.7077, 0.7347] |

## Sensitivity of the minimum-cost operating point to the alert cost (first seed)

The cost of one false alert is an assumption; a missed fraud costs its amount. Cheaper alerts push the optimum towards flagging more.

| alert cost | system | missed (FN) | false alarms (FP) | recall | cost saving |
|---|---|---|---|---|---|
| 100 | final_ensemble | 20 | 0 | 0.9953 | 6,635,227,330 |
| 100 | final_single | 11 | 0 | 0.9974 | 6,682,128,625 |
| 100 | paper_ensemble | 146 | 3008 | 0.9656 | 6,676,166,862 |
| 1,000 | final_ensemble | 20 | 0 | 0.9953 | 6,635,227,330 |
| 1,000 | final_single | 11 | 0 | 0.9974 | 6,682,128,625 |
| 1,000 | paper_ensemble | 152 | 2770 | 0.9642 | 6,673,300,752 |
| 10,000 | final_ensemble | 20 | 0 | 0.9953 | 6,635,227,330 |
| 10,000 | final_single | 11 | 0 | 0.9974 | 6,682,128,625 |
| 10,000 | paper_ensemble | 587 | 617 | 0.8619 | 6,626,131,993 |
| 50,000 | final_ensemble | 20 | 0 | 0.9953 | 6,635,227,330 |
| 50,000 | final_single | 11 | 0 | 0.9974 | 6,682,128,625 |
| 50,000 | paper_ensemble | 955 | 269 | 0.7753 | 6,558,566,026 |