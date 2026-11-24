# Final results (test set, thresholds chosen on validation, 3 seed(s))

Final ensemble: rf(sqrtw)+xgb(plain) [weighted_logit [0.8, 0.2]]; final single model: rf (sqrtw); features: plus; type gating: on (CASH_OUT, TRANSFER). Paper baseline: raw 5 features, LR+DT+RF+XGB+LightGBM soft vote. Alert cost per false alarm: 10000.


## Operating point: FPR <= 0.01%

| system | missed (FN) | false alarms (FP) | recall | FPR | MCC | F2 | PR-AUC | LFR (test) | cost saving |
|---|---|---|---|---|---|---|---|---|---|
| final_ensemble | 0 ± 0 | 66 ± 4 | 1.0000 ± 0.0000 | 0.00005 | 0.992 ± 0.000 | 0.997 ± 0.000 | 1.0000 ± 0.0000 | 0.9999 ± 0.0001 | 6,682,218,211 |
| final_single | 0 ± 0 | 41 ± 2 | 1.0000 ± 0.0000 | 0.00003 | 0.995 ± 0.000 | 0.998 ± 0.000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 6,682,471,544 |
| paper_ensemble | 1184 ± 11 | 100 ± 6 | 0.7214 ± 0.0026 | 0.00008 | 0.835 ± 0.002 | 0.760 ± 0.003 | 0.9364 ± 0.0015 | 0.7504 ± 0.0020 | 6,526,124,873 |


## Operating point: recall >= 99.5%

| system | missed (FN) | false alarms (FP) | recall | FPR | MCC | F2 | PR-AUC | LFR (test) | cost saving |
|---|---|---|---|---|---|---|---|---|---|
| final_ensemble | 44 ± 4 | 0 ± 0 | 0.9897 ± 0.0009 | 0.00000 | 0.995 ± 0.000 | 0.992 ± 0.001 | 1.0000 ± 0.0000 | 0.9999 ± 0.0001 | 6,451,927,939 |
| final_single | 41 ± 4 | 0 ± 0 | 0.9904 ± 0.0008 | 0.00000 | 0.995 ± 0.000 | 0.992 ± 0.001 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 6,511,924,237 |
| paper_ensemble | 18 ± 2 | 37868 ± 5661 | 0.9958 ± 0.0005 | 0.03043 | 0.313 ± 0.023 | 0.360 ± 0.036 | 0.9364 ± 0.0015 | 0.7504 ± 0.0020 | 6,303,775,271 |


## Operating point: minimum cost

| system | missed (FN) | false alarms (FP) | recall | FPR | MCC | F2 | PR-AUC | LFR (test) | cost saving |
|---|---|---|---|---|---|---|---|---|---|
| final_ensemble | 19 ± 1 | 0 ± 0 | 0.9955 ± 0.0001 | 0.00000 | 0.998 ± 0.000 | 0.996 ± 0.000 | 1.0000 ± 0.0000 | 0.9999 ± 0.0001 | 6,635,463,246 |
| final_single | 10 ± 1 | 0 ± 0 | 0.9976 ± 0.0001 | 0.00000 | 0.999 ± 0.000 | 0.998 ± 0.000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 6,682,364,541 |
| paper_ensemble | 549 ± 36 | 698 ± 97 | 0.8709 ± 0.0085 | 0.00056 | 0.856 ± 0.005 | 0.865 ± 0.004 | 0.9364 ± 0.0015 | 0.7504 ± 0.0020 | 6,629,701,248 |


## Early and late test regimes at FPR <= 0.01% (first seed)

| system | early recall | early FP | late recall | late FP | recall 95% CI |
|---|---|---|---|---|---|
| final_ensemble | 1.0000 | 53 | 1.0000 | 11 | [0.9991, 1.0000] |
| final_single | 1.0000 | 33 | 1.0000 | 7 | [0.9991, 1.0000] |
| paper_ensemble | 0.7422 | 74 | 0.7089 | 22 | [0.7077, 0.7347] |

## Sensitivity of the minimum-cost operating point to the alert cost (first seed)

The cost of one false alert is an assumption; a missed fraud costs its amount. Cheaper alerts push the optimum towards flagging more.

| alert cost | system | missed (FN) | false alarms (FP) | recall | cost saving |
|---|---|---|---|---|---|
| 100 | final_ensemble | 20 | 0 | 0.9953 | 6,635,227,330 |
| 100 | final_single | 11 | 0 | 0.9974 | 6,682,128,625 |
| 1,000 | final_ensemble | 20 | 0 | 0.9953 | 6,635,227,330 |
| 1,000 | final_single | 11 | 0 | 0.9974 | 6,682,128,625 |
| 10,000 | final_ensemble | 20 | 0 | 0.9953 | 6,635,227,330 |
| 10,000 | final_single | 11 | 0 | 0.9974 | 6,682,128,625 |
| 50,000 | final_ensemble | 20 | 0 | 0.9953 | 6,635,227,330 |
| 50,000 | final_single | 11 | 0 | 0.9974 | 6,682,128,625 |