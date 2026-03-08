# Cluster Trace Download Instructions

Arbiter Engine supports two real cluster trace formats:
- **Google Borg 2019** — task scheduling trace from Google's internal cluster
- **Alibaba 2018** — batch workload trace from Alibaba's production cluster

These files are **not included in the repository** because they are large (~10 GB each).
Download them separately and point the scripts to the local path.

---

## Google Borg Cluster Trace 2019

**Source:** https://research.google/tools/datasets/google-cluster-workload-traces-2019/

**Download:**
1. Fill out the data request form at the link above
2. Download one shard of `task_events`: `part-00000-of-00500.csv.gz`
3. Decompress: `gunzip part-00000-of-00500.csv.gz`

**Column schema** (no header row — positional):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | time | int | Microseconds since trace start |
| 1 | missing_info | int | Bitmask of unavailable fields |
| 2 | job_id | int | Unique job ID |
| 3 | task_index | int | Task index within job |
| 4 | machine_id | int | Assigned machine (empty if not SCHEDULE) |
| 5 | event_type | int | 0=SUBMIT, 1=SCHEDULE, 4=FINISH, 3=FAIL |
| 7 | scheduling_class | int | 0–3 (higher = latency-sensitive) |
| 8 | priority | int | 0–11 (higher = important) |
| 9 | cpu_request | float | Normalised CPU [0.0–1.0] |
| 10 | memory_request | float | Normalised memory [0.0–1.0] |

**Usage:**
```bash
python scripts/trace_replay.py \
    --trace google \
    --trace-file data/part-00000-of-00500.csv \
    --tasks 2000 --workers 50
```

---

## Alibaba Cluster Trace 2018

**Source:** https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018

**Download:**
```bash
# From the Alibaba GitHub releases
wget https://github.com/alibaba/clusterdata/releases/download/v2018/batch_task.tar.gz
tar -xzf batch_task.tar.gz
# This extracts batch_task.csv (~2.5 GB)
```

**Column schema** (CSV with header):
| Column | Type | Description |
|--------|------|-------------|
| task_name | str | Unique task identifier |
| instance_num | int | Parallel instances |
| job_name | str | Parent job |
| task_type | str | Task type label |
| status | str | Terminated / Waiting / Running / Failed |
| start_time | float | Seconds since trace start |
| end_time | float | Seconds since trace start |
| plan_cpu | float | CPU request (100 = 1 core) |
| plan_mem | float | Memory fraction [0.0–1.0] |

**Usage:**
```bash
python scripts/trace_replay.py \
    --trace alibaba \
    --trace-file data/batch_task.csv \
    --tasks 2000 --workers 50
```

---

## Quick Test Without Real Data

Both loaders gracefully fail with a clear error if the file is missing.
To test the loader code itself using synthetic data shaped like the real traces:

```bash
python scripts/trace_replay.py --trace synthetic --tasks 500 --workers 20
```

---

## Field Mapping Summary

| Arbiter Field | Borg Source | Alibaba Source |
|--------------|-------------|----------------|
| `compute_cost` | `cpu_request * 16` | `plan_cpu / 100` |
| `estimated_duration` | `cpu_request * 60` (heuristic) | `end_time - start_time` |
| `deadline` | `arrival + duration * 1.5–4.0` | `arrival + duration * 2.5` |
| `priority` | `priority (0–11) → (1–10)` | `5` (not in trace) |
| `failure_probability` | `0.05–0.15` based on priority | `0.03–0.08` based on status |
| `resource_type` | `gpu` if scheduling_class=3 | `cpu` |
