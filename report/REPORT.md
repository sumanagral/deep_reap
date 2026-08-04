# DeepREAP — Implementation Report

## 1. Goal

Operationalize the DeepREAP architecture proposed by Sharma et al. (SJSU
CMPE-294, 2023): integrate the **R**egressive **E**nsemble **A**pproach
for **P**rediction (REAP) with the **DeepRM_Plus** deep-RL scheduler,
and surface the combined system over a REST API with Prometheus +
Grafana monitoring. The original paper is purely conceptual; this
project delivers a runnable reference.

## 2. System overview

```
                ┌─────────────────────────────────────┐
                │ Synthetic workload generator        │
                │  resource_usage.csv  job_trace.csv  │
                └─────────────────┬───────────────────┘
                                  │
                ┌─────────────────▼───────────────────┐
                │ REAP                                │
                │  GA feature select  →  5 regressors │
                │  → accuracy-weighted ensemble        │
                └─────────────────┬───────────────────┘
                                  │  predicted CPU / mem demand
                                  ▼
                ┌─────────────────────────────────────┐
                │ DeepRM_Plus (CNN policy)            │
                │  state image  ⊕  REAP forecast      │
                │  imitation (SJF) → PPO refinement   │
                └─────────────────┬───────────────────┘
                                  │  scheduling decisions
                                  ▼
                ┌─────────────────────────────────────┐
                │ FastAPI service (/predict /forecast │
                │   /schedule /feedback /metrics)     │
                └─────────────────┬───────────────────┘
                                  │  /metrics scrape
                                  ▼
                       Prometheus  →  Grafana
```

## 3. Synthetic data

The paper assumes "historical resource usage" without specifying a
trace, so we generate one that reproduces realistic patterns
(`data/synthetic_generator.py`):

- **Resource usage** — hourly records for four service types (Web,
  Database, Media, Compute) with diurnal + weekly cycles, an
  autoregressive component on the previous-hour CPU, Gaussian noise.
  Default 60 days × 4 services = **5 760 rows**.
- **Job trace** — Poisson arrivals, bimodal duration (80% short
  1–3 slots, 20% long 10–15 slots), bimodal per-resource demand (one
  dominant resource per job), matching the regime in the original
  DeepRM paper. Default **3 000 jobs over 1 500 timesteps**.

## 4. REAP (`src/reap/`)

Pipeline:

1. **GA feature selection** (`feature_selection.py`) — chromosome is a
   binary mask over candidate features. Fitness =
   `−CV_MSE − 0.01 · #features` of a fast Ridge regressor on the
   selected subset (3-fold CV, 25 generations, pop 24, tournament-3
   selection, single-point crossover, 5% mutation).
2. **Base regressors** (`models.py`) — five models named in the paper:
   Linear Regression, Bayesian Ridge, Decision Tree, Random Forest,
   SVR(rbf).
3. **Sharper ensemble** (`ensemble.py`) — default Softmax(−MSE/τ)
   weighting (temperature τ, default 2.0) so near-tied base models are
   no longer nearly uniform; alternatives include Top-K selection and
   a Ridge stacking meta-learner. Training uses a **temporal holdout**
   (last 20%) plus **TimeSeriesSplit** rolling-window CV for weights —
   never a shuffled random split. Online re-weighting via
   `REAPModel.update_online_weights` (and the `/feedback` API) EMA-blends
   fresh Softmax weights as cluster observations arrive.

### Empirical results (cpu_load target, 60-day trace)

| Model            | MSE   | MAE  | Ensemble weight |
|------------------|------:|-----:|----------------:|
| LinearRegression | 9.33  | 2.45 | 0.205 |
| BayesianRidge    | 9.34  | 2.45 | 0.205 |
| DecisionTree     | 14.48 | 2.99 | 0.132 |
| RandomForest     | 8.41  | 2.32 | 0.227 |
| SVR              | 8.26  | 2.30 | 0.231 |
| **ENSEMBLE**     | **8.07** | **2.28** | — |

The ensemble strictly dominates every base model on both MSE and MAE.
GA selected 8 of 11 features, dropping `day_of_week`, `is_weekend`,
and `svc_Database` (their information is already captured by
`hour_of_day`, the AR term, and the other one-hots).

For the memory-usage target the same pattern holds (per-model values
in `results/plots/reap_quality_memory_usage.png`).

## 5. DeepRM_Plus (`src/deeprm/`)

### Environment (`env.py`)

A discrete-time cluster scheduler matching Mao et al. 2016:

- 2 resources × 10 capacity, time horizon T=20.
- M=5 visible job slots + bounded backlog (60).
- State image = `cluster_load | job_1 | … | job_M`, shape
  `(1, n_resources, T·(M+1))`. With REAP integration the image gains
  `reap_channels` extra layers carrying the demand forecast.
- Action: index in [0, M] — schedule a visible job or no-op (M).
  Successful schedules **do not advance time** so the agent can fill
  multiple slots in one timestep.
- Reward (multi-objective, per time-advancing step):
  `R = −Σ_j 1/T_j / 10 + α·ΔN_completed − β·|backlog| − γ·mean_wait`.
  The throughput / backlog / wait terms stop the agent from hoarding
  long jobs to game the slowdown metric. Default episode horizon is
  2000 steps (was 200) so deferred work is forced toward completion.

### Baselines (`baselines.py`)

`fifo`, `sjf`, and `packer` (Tetris-style — pick the visible job whose
demand vector has the highest dot-product with current free capacity).

### CNN policy (`network.py`)

Two `Conv2d(_, _, (1,3))` blocks → `AdaptiveAvgPool2d((1,16))` → MLP
head → `(action_logits, value)`. The adaptive pool keeps the head
size independent of `time_horizon · (M+1)`, which is convenient when
varying configuration.

### Imitation pre-training (`imitation.py`)

Run SJF for `n_episodes`, collect `(state, action)` pairs, train the
CNN with cross-entropy. With 30 episodes (≈6 000 transitions, 6 epochs)
the imitation loss reaches **0.53** on a 6-class problem (uniform-prior
loss is `ln 6 ≈ 1.79`).

### PPO refinement (`ppo.py`)

Clipped-objective PPO with GAE tuned to protect the imitation prior:

- γ=0.995, λ=0.95, **clip=0.05**, vf_coef=0.5, grad-norm clip 0.5
- 4 envs × 256-step rollouts, 3 epochs/update
- **cosine LR decay** `5e-5 → 5e-6`, entropy anneal `0.005 → 0.001`
- **KL early-stop** (target_kl=0.015) aborts an epoch when the policy
  drifts too far from the rollout prior
- Default **400 updates ≈ 4.1×10⁵ transitions** (≈27× the old 15-update
  / 1.5×10⁴ budget; raise `--ppo-updates` further for 10⁶-scale runs)

Both imitation and post-PPO checkpoints are still saved and
benchmarked separately.

## 6. DeepREAP integration (`src/integration/`)

`build_reap_forecast` rolls REAP forward `T` hours given a
`ForecastInput` (timestamp, service type, prev-hour CPU/mem, etc.) and
returns a `(reap_channels, n_resources, T)` tensor that is plugged into
the env via `ClusterEnv.set_reap_forecast`. Channel layout:

- channel 0 = predicted CPU demand, broadcast across the resource axis,
  normalized to [0, 1];
- channel 1 = predicted memory demand, same convention.

The agent has no hard-coded contract with these channels; it learns
during PPO refinement whether and how to weigh them.

## 7. REST API (`src/integration/api.py`)

| Endpoint     | Verb | Notes                                            |
|--------------|------|--------------------------------------------------|
| `/health`    | GET  | reports which models are loaded                  |
| `/predict`   | POST | single REAP prediction, target ∈ {cpu_load, memory_usage} |
| `/forecast`  | POST | full `(C, R, T)` forecast tensor                 |
| `/schedule`  | POST | action from the CNN policy on a supplied state image |
| `/feedback`  | POST | observed value → updates REAP error gauge        |
| `/metrics`   | GET  | Prometheus exposition                            |

Models are loaded from `MODELS_DIR` (env var) on startup. If a model
file is missing, the corresponding endpoint returns 503 instead of
crashing the service.

### Verified roundtrip

```
GET  /health       → {"reap_cpu_loaded":true,"reap_mem_loaded":true,"policy_loaded":true}
POST /predict      → {"target":"cpu_load","prediction":36.7}     (Web @ noon, 750 users)
GET  /metrics      → standard Prometheus exposition
```

## 8. Monitoring (`config/`, `docker-compose.yml`)

Prometheus scrapes `host.docker.internal:8000/metrics` every 5 s.
Grafana is provisioned with the Prometheus datasource and a single
`DeepREAP Overview` dashboard exposing:

- REAP request rate per target;
- REAP latency p50/p95;
- latest predicted value per target;
- observed absolute error from `/feedback`;
- scheduler latency p95;
- aggregate scheduling-decision rate.

Anonymous viewer is enabled so the dashboard is reachable at
`http://localhost:3000` without login.

## 9. Benchmark (`src/evaluation/`)

Schedulers are evaluated across **n≥10 random seeds** with mean ± 95%
CI and Wilcoxon signed-rank p-values vs SJF
(`results/benchmark_summary.json`). The evaluation budget defaults to
8000 steps / 2000 episode steps so deferred long jobs are forced
toward completion and throughput is measured in steady state.

Industry traces are supported via `data/trace_loaders.py`
(`google_cluster`, `alibaba`, `azure_vm`) in addition to the synthetic
generator.

Historical single-seed numbers from the original short-horizon
prototype (for reference; superseded by the multi-seed protocol):

| Scheduler                      | avg_slowdown ↓ | avg_completion ↓ | n_done | total_reward |
|--------------------------------|---------------:|-----------------:|-------:|-------------:|
| fifo                           | 20.20          | 48.65            | 54     | -374.6       |
| sjf                            | 20.11          | 49.51            | 53     | -366.3       |
| packer                         | 18.76          | 48.42            | 52     | -375.3       |
| deeprm_plus_imitation          | 19.30          | 43.41            | 41     | -414.1       |
| deeprm_plus (PPO)              | 24.47          | 55.44            | 41     | -443.7       |
| **deepreap_imitation**         | **8.27**       | **19.04**        | 25     | -446.4       |
| deepreap (PPO)                 | 18.12          | 42.05            | 43     | -401.8       |

### Reading the (legacy) numbers — and what changed

- The old DeepREAP imitation policy achieved low slowdown partly by
  **hoarding** long jobs under a 201-step budget (n_done=25 vs ~53 for
  heuristics) and received the *worst* cumulative reward. The new
  multi-objective reward + longer eval horizon remove that loophole.
- DeepREAP-PPO previously *regressed* from 8.27 → 18.12 slowdown after
  only 15 updates. The longer PPO budget, tighter clip, LR/entropy
  schedules, and KL early-stop are designed to stop that collapse.
- Re-run `python -m scripts.run_pipeline` (or `--quick` for a smoke
  pass) and read `results/benchmark_summary.json` for current
  mean/CI/p-value numbers.

Plots (`results/plots/`):

- `avg_slowdown.png`, `avg_completion.png`, `n_done.png`, `throughput.png`
  (with 95% CI error bars)
- `learning_curve_{deeprm_plus,deepreap}.png`
- `reap_quality_{cpu_load,memory_usage}.png`
- `reap_weights_{cpu_load,memory_usage}.png`

## 10. Reproducing the results

```bash
pip install -r requirements.txt
python -m scripts.run_pipeline                       # full pipeline (longer PPO)
python -m scripts.run_pipeline --quick               # CI / smoke budget
python -m src.evaluation.benchmark --n-seeds 10      # → results/ + CIs
docker compose up -d                                  # Prometheus + Grafana
python -m scripts.run_api                             # → :8000
```

Data/REAP/DeepRM training seeds remain fixed (42); the benchmark
sweeps `seed, seed+1, …, seed+n-1` (default base 123, n=10).

## 11. Problem formulation (POMDP)

We cast online multi-resource cluster scheduling as a continuous-time
*Partially Observable* Markov Decision Process
$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \Omega, O, \gamma)$:

| Symbol | Meaning in DeepREAP |
|--------|---------------------|
| $s_t \in \mathcal{S}$ | True cluster state: per-resource load timeline, all queued / running / finished jobs, wall-clock $t$ |
| $o_t \in \Omega$ | Observation image $(C, R, T\cdot(M+1))$ — cluster load \| $M$ visible job slots \| optional REAP forecast channels. Backlog beyond $M$ is only summarized indirectly (partial observability) |
| $a_t \in \mathcal{A}$ | Discrete: schedule visible slot $j\in\{0..M-1\}$ or no-op ($M$) that advances time |
| $P(s_{t+1}\mid s_t,a_t)$ | Deterministic packing dynamics + stochastic / trace-driven arrivals |
| $O(o_t\mid s_t)$ | Projection of $s_t$ onto the fixed-size image (truncates backlog, hides future arrivals except via forecast channels) |
| $R(s_t,a_t)$ | Multi-objective: $-\sum_j 1/T_j + \alpha\Delta N_{\mathrm{done}} - \beta\|\mathrm{backlog}\| - \gamma\,\overline{\mathrm{wait}} - \delta\,\mathrm{frag}$ |
| $\gamma$ | Discount (PPO uses $\gamma=0.995$) |

The REAP ensemble approximates the latent future-demand component of
$s_t$ that is otherwise invisible under $O$, turning a harder POMDP
into a better-informed MDP approximation for the CNN policy.

## 12. Publishability experiments

| Study | Entry point | Artifact |
|-------|-------------|----------|
| Multi-seed bench + DRF/ILP + P95/P99/frag/SLA | `python -m src.evaluation.benchmark --n-seeds 5` | `results/benchmark_summary.json` |
| Oracle vs proxy vs no-forecast | `python -m src.evaluation.ablation` | `results/ablation/oracle_gap.json` |
| Forecast noise ±5/10/20/50% | (same) | `results/ablation/noise_sensitivity.json` |
| Zero-shot Google/Alibaba-like | (same) | `results/ablation/zero_shot_transfer.json` |
| Decision latency microbench | (same) | `results/ablation/latency.json` |
| Production-style traces | `python -m data.production_traces` | `data/google_like_*.csv`, `data/alibaba_like_*.csv` |

REAP training uses a **temporal holdout** (last 20% of rows) plus
**TimeSeriesSplit** rolling-window CV for ensemble weights — random
shuffle splits are disabled to prevent temporal leakage.

## 13. Limitations & threats to validity

**Internal validity**
- PPO variance remains non-trivial at $10^5$–$10^6$ transitions; we
  report mean±std over $N\ge5$ seeds and paired Wilcoxon / $t$-tests,
  but camera-ready claims should target $N\ge10$ and $p<0.01$.
- The short-horizon ILP baseline is an *online one-step* CBC solve,
  not a full-episode offline optimum (Gurobi MIP over the whole trace
  would be a stronger — and much slower — upper bound).
- Oracle utilization is built by greedy earliest-fit packing of the
  trace; a different offline scheduler would yield a different oracle.

**External validity**
- Vendored Google/Alibaba artifacts are *distribution-matched
  stylized traces* (`data/production_traces.py`) plus loaders for real
  dumps. Reviewers should be shown at least one run on an official
  public sample (Borg 2019 / Alibaba 2018) downloaded at evaluation
  time — full multi-GB dumps are not shipped in-repo.
- Zero-shot transfer tests generalization across arrival/duration
  distributions, but not across cluster *sizes* or heterogeneous
  machine pools.

**Construct validity**
- Slowdown, throughput, fragmentation, and SLA-breach are reported;
  energy / carbon and multi-tenant fairness beyond DRF are out of
  scope.
- Latency microbench measures feature build + forecast slice + CNN
  forward on CPU; production serving (batching, GPU, gRPC) may differ.

**Next steps toward a top-tier submission**
- Run the ablation suite on an official Google/Alibaba sample and
  quote the ≥80% Oracle-gap result with $N\ge10$.
- Scale PPO to $10^6$ transitions; publish learning curves with
  shaded std.
- Add an offline MIP upper bound over short windows (Gurobi/CBC).
- Multi-cluster / migration POMDP extension.

## 14. References

1. W. Guo *et al.*, "Cloud Resource Scheduling With Deep Reinforcement
   Learning and Imitation Learning," *IEEE IoT J.* 8(5), 2021.
2. H. Mao *et al.*, "Resource Management with Deep Reinforcement
   Learning," *HotNets*, 2016.
3. Kaur, Bala, Chana, "An intelligent regressive ensemble approach for
   predicting resource usage in cloud computing," *J. Parallel and
   Distributed Computing* 123, 2019.
4. Sharma, Gupta, Nagral, Ramachandrapuram, "Efficient Cloud Resource
   Allocation: A Novel Integration of Predictive Modeling and
   Reinforcement Learning," SJSU CMPE-294, 2023 — the source paper.
