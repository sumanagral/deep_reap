# DeepREAP

Reference implementation of the **DeepREAP** system from
*Efficient Cloud Resource Allocation: A Novel Integration of Predictive
Modeling and Reinforcement Learning* (Sharma, Gupta, Nagral,
Ramachandrapuram — SJSU CMPE-294, 2023). The original paper proposed
the architecture but did not implement it. This repository delivers a
working, end-to-end prototype:

1. **REAP** — Softmax-/Top-K-/Stacking ensemble of regressors (Linear,
   Bayesian Ridge, Decision Tree, Random Forest, SVR) with GA feature
   selection and online weight updates from cluster feedback.
2. **DeepRM_Plus** — CNN policy trained with imitation learning on an
   SJF expert and refined with PPO (cosine LR decay, clip=0.05, entropy
   annealing, KL early-stop) on a multi-objective cluster scheduling
   environment.
3. **DeepREAP integration** — REAP forecasts are appended as extra
   channels of the RL state image so the scheduler is aware of
   anticipated demand.
4. **REST API + monitoring** — FastAPI service exposing
   `/predict`, `/forecast`, `/schedule`, `/feedback`, `/metrics`,
   scraped by a Prometheus + Grafana stack defined in
   `docker-compose.yml`.
5. **Benchmark** — Multi-seed evaluation (n≥10) with 95% CIs and
   Wilcoxon p-values vs heuristics; supports Google / Alibaba / Azure
   trace loaders.

## Layout

```
deep-reap/
├── data/                       # synthetic generator + industry trace loaders
├── src/
│   ├── reap/                   # ensemble + GA feature selection
│   ├── deeprm/                 # env, CNN, baselines, imitation, PPO
│   ├── integration/            # DeepREAP glue + FastAPI app
│   ├── monitoring/             # Prometheus metrics
│   └── evaluation/             # multi-seed benchmark + plotting
├── scripts/                    # convenience entrypoints
├── config/                     # prometheus.yml + grafana provisioning
├── docker-compose.yml          # monitoring stack
├── tests/test_smoke.py         # end-to-end smoke tests
├── results/                    # benchmark.json + plots/  (generated)
├── models/                     # trained checkpoints       (generated)
└── report/REPORT.md            # methodology + results
```

## Quick start

```bash
pip install -r requirements.txt

# 1. one-shot pipeline (data → REAP → DeepRM_Plus → DeepREAP → benchmark)
python -m scripts.run_pipeline

# quick/CI mode (tiny PPO + 2 seeds)
python -m scripts.run_pipeline --quick

# 2. run the API
python -m scripts.run_api
# -> http://127.0.0.1:8000/docs  (Swagger UI)

# 3. start monitoring (Prometheus + Grafana)
docker compose up -d
# -> Prometheus  http://127.0.0.1:9090
# -> Grafana     http://127.0.0.1:3000   (anonymous viewer enabled)
```

## Step-by-step usage

```bash
# generate synthetic workload
python -m data.synthetic_generator --hours 1440 --jobs 3000 --horizon 1500

# train REAP (softmax ensemble; use --ensemble-scheme stacking|topk for alternatives)
python -m src.reap.train --target cpu_load --ensemble-scheme softmax --temperature 2.0
python -m src.reap.train --target memory_usage --ensemble-scheme softmax --temperature 2.0

# train vanilla DeepRM_Plus (~4e5 transitions by default)
python -m src.deeprm.train --reap-channels 0 \
    --imitation-episodes 40 --ppo-updates 400 --ppo-clip 0.05

# train DeepREAP (REAP forecast as extra channels)
python -m src.deeprm.train --reap-channels 2 \
    --imitation-episodes 40 --ppo-updates 400 --ppo-clip 0.05

# multi-seed benchmark with CIs (longer horizon for fair throughput)
python -m src.evaluation.benchmark --n-seeds 10 --max-steps 8000

# industry traces (Google / Alibaba / Azure CSV dumps)
python -m src.evaluation.benchmark \
    --job-trace path/to/google_sample.csv --trace-source google_cluster --n-seeds 10
```

Artifacts:

```
models/reap/reap_cpu_load.joblib
models/reap/reap_memory_usage.joblib
models/deeprm/{deeprm_plus,deepreap}_imitation.pt   # post-imitation
models/deeprm/{deeprm_plus,deepreap}.pt              # post-PPO
results/benchmark.json
results/benchmark_summary.json                     # mean/std/CI/p-values
results/plots/*.png
```

## Key fixes vs. the original prototype

| Area | Change |
|------|--------|
| PPO collapse | 10–100× training budget, cosine LR decay, clip=0.05, entropy anneal, KL early-stop |
| Reward mismatch | Multi-objective: slowdown + throughput bonus − backlog/wait penalties; eval horizon 2k–8k steps |
| Ensemble weights | Softmax(−MSE/τ), Top-K, Stacking meta-learner; online re-weight via `/feedback` |
| Experimental rigor | n≥10 seeds, 95% CIs, Wilcoxon tests; Google/Alibaba/Azure loaders |

## Tests

```bash
pytest -q
```

The smoke tests run every module on tiny inputs in under 30 seconds.

## API reference (summary)

| Method | Path        | Description                                         |
|--------|-------------|-----------------------------------------------------|
| GET    | /health     | Liveness + which models are loaded                  |
| POST   | /predict    | Single REAP prediction (CPU or memory)              |
| POST   | /forecast   | (channels, R, T) forecast tensor for the env state  |
| POST   | /schedule   | Action from the trained CNN policy                  |
| POST   | /feedback   | Observed value (+ optional features) → error gauge + online ensemble re-weight |
| GET    | /metrics    | Prometheus exposition                               |

See `report/REPORT.md` for methodology, design decisions, and results.
