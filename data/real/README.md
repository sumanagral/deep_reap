# Real production-trace subsets

Converted evaluation subsets from public research releases. Reproducible via:

```bash
python -m data.fetch_public_traces --max-jobs 5000
```

| File | Source | Notes |
|------|--------|-------|
| `google2011_jobs.csv` | Google Borg Cluster Data 2011 (task_events shard 0) | CC-BY; authentic SUBMIT events with CPU/mem requests |
| `google2019_jobs.csv` | Google Cluster Data 2019 (collection_events shard 0) | Coarse collections; demand inferred from scheduling class |
| `alibaba2018_jobs.csv` | Alibaba Cluster Trace v2018 `batch_task` | plan_cpu / plan_mem → resource demand |
| `azure2019_jobs.csv` | Azure Public Dataset V2 `vmtable` | VM lifetime → duration; core/memory buckets → demand |
| `SOURCES.json` | Provenance + row counts | Citations / URLs |

Canonical schema for all CSVs:

```
job_id,arrival_time,duration,res_0,res_1
```

Benchmark example:

```bash
python -m src.evaluation.benchmark \
  --job-trace data/real/alibaba2018_jobs.csv \
  --trace-source canonical --n-seeds 5 --max-steps 4000
```

Raw multi-GB archives are **not** vendored (GitHub size limits). Use
`python -m data.fetch_public_traces --keep-raw` to retain them under
`data/dumps/`.
