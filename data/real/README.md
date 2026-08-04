# Real production-trace data

Public research dumps converted (and raw samples) for DeepREAP evaluation.

## Converted job traces (canonical schema)

```bash
python -m data.fetch_public_traces --max-jobs 20000
```

| File | Source | Rows |
|------|--------|------|
| `google2011_jobs.csv` | Google Borg 2011 `task_events` | 20 000 |
| `google2019_jobs.csv` | Google 2019 `collection_events` | 20 000 |
| `alibaba2018_jobs.csv` | Alibaba Cluster Trace 2018 `batch_task` | 20 000 |
| `azure2019_jobs.csv` | Azure Public Dataset V2 `vmtable` | 20 000 |

Schema: `job_id,arrival_time,duration,res_0,res_1`

## Raw dump samples (`raw/`)

Authentic excerpts suitable for git (not full multi-GB archives):

| Path | Source |
|------|--------|
| `raw/google2011/*.csv.gz` | Official Borg 2011 task_events shards 0,1,2,3,10 |
| `raw/google2011_task_events_sample.csv` | First 20k rows of shard 0 (with header) |
| `raw/google2011_task_usage_sample.csv` | First 10k rows of task_usage shard 0 |
| `raw/alibaba2018_batch_task_sample.csv` | First 50k rows + header |
| `raw/alibaba2018_machine_meta.csv` | Full machine_meta table |
| `raw/alibaba2018_container_meta_sample.csv` | First 20k rows |
| `raw/azure2019_vmtable_sample.csv` | First 20k rows + header |
| `raw/cloudsimplus_google_*_sample.csv` | CloudSim Plus google-trace samples |

## Re-download full archives

```bash
python -m data.fetch_public_traces --max-jobs 20000 --keep-raw
# → data/dumps/  (gitignored; can be hundreds of MB–GB)
```

## Benchmark

```bash
python -m src.evaluation.benchmark \
  --job-trace data/real/google2011_jobs.csv --trace-source canonical --n-seeds 5

python -m src.evaluation.benchmark \
  --job-trace data/real/alibaba2018_jobs.csv --trace-source canonical --n-seeds 5
```

See `SOURCES.json` for URLs, licenses, and citations.
