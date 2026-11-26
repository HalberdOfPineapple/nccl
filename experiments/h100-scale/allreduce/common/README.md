# Harness for the AllReduce scalability series

Scripts that run in-tree NCCL through nccl-tests (`/root/nccl-tests/build/all_reduce_perf`) and turn the runs into tables and figures. There is no kernel and nothing to build. The series is in [`../scalability/`](../scalability/PLANS.md).

| Script | What it does |
|---|---|
| [`scripts/run_allreduce.py`](scripts/run_allreduce.py) | Runs one session file: rounds of cells, arm order rotated between rounds. For each run: lock the GPUs of the set, check that they are idle (stop with exit code 10 if not), snapshot, run, snapshot, check that the NCCL log names the intended algorithm or kernel and that there are 0 wrong values. Appends one JSON line per run and skips runs already in the file, so a stopped session can be restarted |
| [`scripts/run_chain.sh`](scripts/run_chain.sh) | Runs all sessions of one GPU count in order: calibration (not at 8 GPUs), scaling, link-byte pairs, payload sweep, GPU-set control |
| [`scripts/make_configs.py`](scripts/make_configs.py) | Writes the session files for 2, 4 and 6 GPUs and the 6-GPU follow-up. The 8-GPU session files were written by hand before it existed |
| [`scripts/summarize.py`](scripts/summarize.py) | JSON lines → one CSV row per cell and payload: median, min and max algBW over valid runs, and busBW with each arm's own factor |
| [`scripts/link_bytes.py`](scripts/link_bytes.py) | Link bytes per payload byte from long and short run pairs (plan 04) |
| [`scripts/make_figures.py`](scripts/make_figures.py) | Figures from the summary CSVs |

## Usage

From `../scalability/`:

```
python3 ../common/scripts/make_configs.py
bash ../common/scripts/run_chain.sh 6 2026-11-16
python3 ../common/scripts/summarize.py gpu_count_scaling/results/2026-11-16/scaling_summary.csv gpu_count_scaling/results/2026-11-16/scaling.jsonl
uv run --no-project --with matplotlib python ../common/scripts/make_figures.py scaling gpu_count_scaling/results/2026-11-16/scaling_summary.csv gpu_count_scaling/results/2026-11-16/figures/gpu_count_scaling.png
uv run --no-project --with matplotlib python ../common/scripts/make_figures.py scaling gpu_count_scaling/results/2026-11-16/scaling_summary.csv gpu_count_scaling/results/2026-11-16/figures/gpu_count_scaling_no6.png --exclude-gpus 6
```

The system Python has no matplotlib, so figures use a throwaway `uv` environment.

## Session file

```
{"rounds": 5, "cells": [{"id": "n8_ring_auto", "arm": "ring", "devices": "0,1,2,3,4,5,6,7",
  "min_size": "2G", "max_size": "2G", "ctas": 24, "iters": 20, "warmup": 5}]}
```

`arm` is one of `ring`, `nvls`, `sym_multicast`, `sym_plain`; the environment for each arm is in `ARMS` in `run_allreduce.py` and in the [common setup](../scalability/00_common_setup.md#how-each-arm-is-selected). `ctas`, `iters`, `warmup` and `extra_env` are optional.

## Pitfalls met while running

- The NCCL log writes algorithm names in capitals (`RING/SIMPLE`); the selection check compares without case.
- Threads per CTA for Ring and NVLS are not printed by NCCL; they come from the code (512 and 640).
- A waiting loop must not search for a process by a pattern that appears in its own command line (`pgrep -f`): it finds itself and never ends.
- `run_chain.sh` resolves its own directory to an absolute path before changing directory.
- The lock files `/tmp/nccl_thread_store_gpu_<n>.lock` are shared with the other harnesses on this machine, so two harnesses never use the same GPU at once.
