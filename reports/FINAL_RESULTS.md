# Final results (test set, thresholds chosen on validation, 3 seed(s))

Final ensemble: lr+rf (mean); final single model: rf (sqrtw); features: plus. Paper baseline: raw 5 features, LR+DT+RF+XGB+LightGBM soft vote. Alert cost per false alarm: 10000.


## Operating point: FPR <= 0.01%

| system | missed (FN) | false alarms (FP) | recall | FPR | MCC | F2 | PR-AUC | LFR (test) | cost saving |
|---|---|---|---|---|---|---|---|---|---|
| final_ensemble | 7 ± 3 | 11 ± 6 | 0.9984 ± 0.0007 | 0.00001 | 0.998 ± 0.001 | 0.998 ± 0.001 | 0.9999 ± 0.0000 | 0.9984 ± 0.0007 | 6,682,375,833 |
| final_single | 1 ± 0 | 33 ± 3 | 0.9998 ± 0.0000 | 0.00003 | 0.996 ± 0.000 | 0.998 ± 0.000 | 1.0000 ± 0.0000 | 0.9998 ± 0.0000 | 6,682,155,833 |
| paper_ensemble | 1184 ± 11 | 100 ± 6 | 0.7214 ± 0.0026 | 0.00008 | 0.835 ± 0.002 | 0.760 ± 0.003 | 0.9364 ± 0.0015 | 0.7504 ± 0.0020 | 6,526,124,873 |


## Operating point: recall >= 99.5%

| system | missed (FN) | false alarms (FP) | recall | FPR | MCC | F2 | PR-AUC | LFR (test) | cost saving |
|---|---|---|---|---|---|---|---|---|---|
| final_ensemble | 39 ± 7 | 0 ± 0 | 0.9907 ± 0.0016 | 0.00000 | 0.995 ± 0.001 | 0.993 ± 0.001 | 0.9999 ± 0.0000 | 0.9984 ± 0.0007 | 6,477,154,350 |
| final_single | 38 ± 12 | 0 ± 0 | 0.9911 ± 0.0027 | 0.00000 | 0.996 ± 0.001 | 0.993 ± 0.002 | 1.0000 ± 0.0000 | 0.9998 ± 0.0000 | 6,528,594,331 |
| paper_ensemble | 18 ± 2 | 37868 ± 5661 | 0.9958 ± 0.0005 | 0.03043 | 0.313 ± 0.023 | 0.360 ± 0.036 | 0.9364 ± 0.0015 | 0.7504 ± 0.0020 | 6,303,775,271 |


## Operating point: minimum cost

| system | missed (FN) | false alarms (FP) | recall | FPR | MCC | F2 | PR-AUC | LFR (test) | cost saving |
|---|---|---|---|---|---|---|---|---|---|
| final_ensemble | 20 ± 0 | 0 ± 0 | 0.9953 ± 0.0000 | 0.00000 | 0.998 ± 0.000 | 0.996 ± 0.000 | 0.9999 ± 0.0000 | 0.9984 ± 0.0007 | 6,635,227,330 |
| final_single | 2 ± 0 | 1 ± 1 | 0.9995 ± 0.0000 | 0.00000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.0000 ± 0.0000 | 0.9998 ± 0.0000 | 6,682,121,959 |
| paper_ensemble | 549 ± 36 | 698 ± 97 | 0.8709 ± 0.0085 | 0.00056 | 0.856 ± 0.005 | 0.865 ± 0.004 | 0.9364 ± 0.0015 | 0.7504 ± 0.0020 | 6,629,701,248 |


## Early and late test regimes at FPR <= 0.01% (first seed)

| system | early recall | early FP | late recall | late FP | recall 95% CI |
|---|---|---|---|---|---|
| final_ensemble | 0.9987 | 15 | 0.9989 | 3 | [0.9972, 0.9995] |
| final_single | 1.0000 | 25 | 0.9996 | 5 | [0.9987, 1.0000] |
| paper_ensemble | 0.7422 | 74 | 0.7089 | 22 | [0.7077, 0.7347] |

## Sensitivity of the minimum-cost operating point to the alert cost (first seed)

The cost of one false alert is an assumption; a missed fraud costs its amount. Cheaper alerts push the optimum towards flagging more.

| alert cost | system | missed (FN) | false alarms (FP) | recall | cost saving |
|---|---|---|---|---|---|
| 100 | final_ensemble | 20 | 0 | 0.9953 | 6,635,227,330 |
| 100 | final_single | 2 | 1 | 0.9995 | 6,682,128,525 |
| 100 | paper_ensemble | 146 | 3008 | 0.9656 | 6,676,166,862 |
| 1,000 | final_ensemble | 20 | 0 | 0.9953 | 6,635,227,330 |
| 1,000 | final_single | 2 | 1 | 0.9995 | 6,682,127,625 |
| 1,000 | paper_ensemble | 152 | 2770 | 0.9642 | 6,673,300,752 |
| 10,000 | final_ensemble | 20 | 0 | 0.9953 | 6,635,227,330 |
| 10,000 | final_single | 2 | 1 | 0.9995 | 6,682,118,625 |
| 10,000 | paper_ensemble | 587 | 617 | 0.8619 | 6,626,131,993 |
| 50,000 | final_ensemble | 20 | 0 | 0.9953 | 6,635,227,330 |
| 50,000 | final_single | 2 | 1 | 0.9995 | 6,682,078,625 |
| 50,000 | paper_ensemble | 955 | 269 | 0.7753 | 6,558,566,026 |