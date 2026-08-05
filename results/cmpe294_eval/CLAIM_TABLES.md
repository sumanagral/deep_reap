# CMPE-294 DeepREAP Claim Tables

## 1. Demand Prediction Performance

| Model | CPU MAE | Memory MAE | Overall MSE |
| --- | --- | --- | --- |
| LinearRegression | 0.1871 | 1.1218 | 0.9512 |
| SVR | 0.0617 | 1.1052 | 1.3261 |
| RandomForest | 0.0043 | 1.1145 | 0.9724 |
| BayesianRidge | 0.1871 | 1.1212 | 0.9509 |
| DecisionTree | 0.0078 | 1.1305 | 1.1396 |
| **REAP Ensemble** | 0.0064 | 1.1181 | 0.9420 |

- Ensemble wins CPU MAE vs best single: **False**
- Ensemble wins Memory MAE vs best single: **False**

## 2. Overall Scheduling Efficiency

| Policy / System | Avg Turnaround Time (s) | Avg Wait Time (s) | Resource Utilization (%) |
| --- | --- | --- | --- |
| Shortest Job First (SJF) | 31.2420 | 26.2542 | 70.6933 |
| Vanilla DeepRM_Plus | 30.8856 | 25.9617 | 69.4310 |
| **DeepREAP (Proposed)** | 30.9538 | 26.0320 | 48.7252 |
| Oracle DRL | 30.8778 | 25.9759 | 48.1597 |

- Scheduling claim supported (DeepREAP & Oracle beat Vanilla on TAT, p<0.05): **False**
- Oracle vs Vanilla TAT p=0.9771296144902547
- DeepREAP vs Vanilla TAT p=0.6352230310248806
- DeepREAP vs SJF TAT p=0.09838273383650131

## 3. Phase C — Noise Robustness (Gaussian σ on REAP channels)

| σ (%) | Avg Turnaround | Beats Vanilla (mean) | Wilcoxon p (less) |
| --- | --- | --- | --- |
| 0 | 31.5405 | False | 0.6753 |
| 5 | 31.5106 | False | 0.5899 |
| 10 | 31.4647 | True | 0.5548 |
| 15 | 31.4579 | True | 0.5330 |
| 20 | 31.4312 | True | 0.5330 |
| 25 | 31.4301 | True | 0.5000 |
| 30 | 31.4599 | True | 0.5765 |

- Clean DeepREAP beats Vanilla (σ=0, p<0.05): **False**
- Robustness claim supported: **False**
