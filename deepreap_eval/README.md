# DeepREAP Offline Evaluation (CMPE-294 Protocol)

Minimal single-process evaluation matching the CMPE-294 methodology:

```
deepreap_eval/
├── data/               # Symlinks to Google usage + job traces
├── reap.py             # Inv-MSE REAP ensemble facade
├── deeprm_plus.py      # CNN + SJF imitation + PPO facade
└── run_experiments.py  # Phases A / B / C + claim tables
```

## Run

```bash
PYTHONPATH=. python3 -m deepreap_eval.run_experiments \
  --usage-csv data/real/resource_usage_google.csv \
  --job-trace data/real/google2011_jobs.csv \
  --out results/cmpe294_eval \
  --n-episodes 30 --ppo-updates 100
```

Outputs `results/cmpe294_eval/CLAIM_TABLES.md` with:

1. Demand prediction MAE/MSE (REAP vs LR / SVR / RF)
2. Scheduling efficiency (SJF / Vanilla DeepRM+ / DeepREAP / Oracle)
3. Gaussian noise robustness sweep (σ = 0–30%)
