# GB200 NVLS harness: unattended nccl-tests sessions

Runs the performance-validation part of the [GB200 series](../rca/PLANS.md) ([plan 01](../rca/degradation_check/01_degradation_check.md), E1.1–E1.4) without intervention: a batch job builds NCCL and nccl-tests if needed, runs every planned nccl-tests invocation with snapshots around it, and exits. Analysis happens later, in an interactive session or on the login node, from the saved records.

The role-loop harness (`mm_mix`, plan 01 E1.5 and plan 02) is not here yet; it is the H100 harness in [`h100x4/nvls/common`](../../../h100x4/nvls/common/) and needs the port described in plan 01 E1.5.

## What runs where

| File | Runs on | Does |
|---|---|---|
| [`jobs/plan01.sbatch`](jobs/plan01.sbatch) | Host of the batch node (no CUDA toolkit) | `#SBATCH` settings, session directory, time budget; one `srun --container-image=...` step |
| [`scripts/run_in_container.sh`](scripts/run_in_container.sh) | Container (`nvcr.io/nvidia/pytorch:25.06-py3`) | Checks it is in the container, cleans PATH, incremental build of NCCL (`build-gb200/`, `sm_100`) and nccl-tests, runs each plan with the runner |
| [`scripts/nccl_runner.py`](scripts/nccl_runner.py) | Container | Expands a plan into runs, runs them one by one, saves everything per run, resumes after interruption |
| [`scripts/summarize_nccl.py`](scripts/summarize_nccl.py) | Anywhere with Python 3 (no GPU) | Session directories → `runs.csv`, `configs.csv`, `counters.csv` |
| [`scripts/pick_calibration.py`](scripts/pick_calibration.py) | Anywhere | E1.2 `configs.csv` → `calibrated_ctas.json` by the plan's stopping rule |
| Plans: [`../rca/degradation_check/configs/plans/`](../rca/degradation_check/configs/plans/) | — | One JSON per sub-experiment (`e1_1`, `e1_2`, `e1_3`, `e1_4`, `e1_3_repeat`) |

## Arms

Defined in `ARMS` in `nccl_runner.py`. Every run counts only if the NCCL log names the intended kernel (status `ok`); otherwise its status is `kernel_mismatch` and it is left out of every table.

| Arm | Collective, path | Selection | CTA knob |
|---|---|---|---|
| AG-MC | AllGather, symmetric `AllGather_STMC` (`multimem.st`) | `NCCL_SYM_KERNEL=allgather:AllGather_STMC`, `NCCL_SYM_CE_THRESHOLD=2^40`, `-R 2` | `NCCL_SYM_CTAS` (1–64) |
| AG-CE | AllGather, copy engine (NCCL's choice for registered AllGather above 8 MiB) | `-R 2`, nothing forced | none |
| RS-MC | ReduceScatter, symmetric `ReduceScatter_LDMC` (`multimem.ld_reduce`) | `NCCL_SYM_KERNEL=reducescatter:ReduceScatter_LDMC`, `-R 2` | `NCCL_SYM_CTAS` |
| AR-MC | AllReduce, symmetric `AllReduce_RSxLDMC_AGxSTMC` (both, concurrently) | `NCCL_SYM_KERNEL=allreduce:...`, `-R 2` | `NCCL_SYM_CTAS` |
| AR-TMA | AllReduce, symmetric `AllReduce_RSxTmaLD_AGxTmaST` (TMA plain loads/stores) | as above | `NCCL_SYM_CTAS` |
| AR-LDST | AllReduce, symmetric `AllReduce_RSxLD_AGxST` (plain loads/stores) | as above | `NCCL_SYM_CTAS` |
| AR-NVLS | AllReduce, non-symmetric NVLS | `NCCL_ALGO=allreduce:nvls` | `NCCL_NVLS_NCHANNELS` |
| AR-Ring, AG-Ring, RS-Ring | Ring, Simple protocol | `NCCL_ALGO=<collective>:ring NCCL_PROTO=Simple` | `NCCL_MIN_NCHANNELS = NCCL_MAX_NCHANNELS` |
| AR-Tuner | AllReduce, NCCL's own choice | nothing forced | none |

Pitfall found 2026-11-26: without `NCCL_SYM_CE_THRESHOLD`, a registered 2 GiB AllGather runs on the copy engine even with `NCCL_SYM_KERNEL` set, (937–945 GB/s algBW; that is 709 GB/s of data per direction, because the copy engine sends each part to the 3 peers as unicast, 0.75 data bytes per payload byte each way); the log line is `AllGather [Copy Engine]: ... -> cudaMemcpy`.

## Plan file format

```json
{"name": "e1_2", "seed": 12, "blocks": [
  {"name": "e1_2", "arms": ["AR-MC", "AR-Ring"],
   "ctas": {"AR-MC": [1, 8, 64], "AR-Ring": "calibrated"},   // per arm; "calibrated" reads --calibration
   "ctas_default": [null],                                      // arms not in "ctas"; null = NCCL default
   "sizes": ["2G"], "rounds": 3, "iterations": 30, "warmup": 5, "check": 0,
   "counter_pair": {"long": {"n": 200, "w": 20}, "short": {"n": 40, "w": 20}},  // optional
   "environment": {"NCCL_FOO": "1"}}                            // optional, added to every run
]}
```

Runs are ordered round by round, shuffled inside a round with the plan's seed; a long/short counter pair stays back to back. Run ids start with the position in this order, so a plan must not be edited between a session's attempts (start a new session directory instead).

## What each run saves (`<session>/raw/<run_id>/`)

| File | Content | Metric |
|---|---|---|
| `result.json` | Run configuration, command, NCCL environment, parsed nccl-tests rows, launched kernel and CTAs, status, NVLink data and raw counters of all GPUs and links before and after, GPU clocks/temperature/power/throttle reasons before and after, other compute processes | S1, B3, R8–R10, G1, C8 |
| `nccl.<pid>.log` | NCCL log, `INIT,ENV,TUNING` | C8 |
| `stdout.txt`, `stderr.txt` | nccl-tests output | |

`result.json` is written last; its presence marks a finished run and is what resuming checks. Per session: `raw/session_context_<time>.json` (node, rack, GPUs, `nvidia-smi -q`, topology, nvcc, git commits and modified files of NCCL and nccl-tests), `raw/build_*.log`, `runs.jsonl` (one line per finished run), `plan_<name>.json` (copy of each plan used). `raw/` is ignored by git.

## How to run

Interactive (`beta_test` session started by the user with `--container-name=dev`), from a VS Code / SSH shell on the node:

```bash
cd /mnt/home/wli10/nccl
C=experiments/gb200x4/nvls/common
P=experiments/gb200x4/nvls/rca/degradation_check/configs/plans
S=experiments/gb200x4/nvls/rca/degradation_check/results/$(date -u +%F)/interactive_$(hostname | cut -c1-5)
.vscode/in_container.sh bash $C/scripts/run_in_container.sh --session $PWD/$S --minutes 90 $P/e1_1.json $P/e1_2.json
```

Batch (on the login node; the job writes `results/<date>/job_<id>/`):

```bash
# once: pre-import the image so jobs do not pull ~20 GB on billed GPU time (the launcher uses it if present)
mkdir -p /ddn/scratch/cornell/wli10/containers
enroot import --arch arm64 -o /ddn/scratch/cornell/wli10/containers/pytorch-25.06-py3.sqsh 'docker://nvcr.io#nvidia/pytorch:25.06-py3'

cd /mnt/home/wli10/nccl
P=experiments/gb200x4/nvls/rca/degradation_check/configs/plans
sbatch --test-only -p beta_test --qos=test experiments/gb200x4/nvls/common/jobs/plan01.sbatch $P/smoke.json   # validity only; beta rejects --test-only
sbatch experiments/gb200x4/nvls/common/jobs/plan01.sbatch $P/e1_3.json $P/e1_4.json
```

The launcher's own lines run on the host; only its single `srun --container-image` step runs in the container, with `--cpus-per-task` passed explicitly. A requeued job (same id) reuses its session directory and skips finished runs.

Check a plan without GPUs: `python3 scripts/nccl_runner.py <plan> --session /tmp/x --calibration <json> --dry-run`.

Analysis (no GPU):

```bash
python3 scripts/summarize_nccl.py <session_dir>... --output <dir>
python3 scripts/pick_calibration.py <dir>/configs.csv --block e1_2 --output ../rca/degradation_check/configs/calibrated_ctas.json
```

## Counter method

Long and short runs of the same configuration, back to back (`-n 200 -w 20` and `-n 40 -w 20`, `-c 0`). nccl-tests times each iteration as one out-of-place and one in-place call, so long minus short = 2 × 160 calls: payload bytes = 320 × buffer size, time = 160 × (out-of-place + in-place time per call). Data and raw counters are summed over the 18 links of each GPU per direction.
