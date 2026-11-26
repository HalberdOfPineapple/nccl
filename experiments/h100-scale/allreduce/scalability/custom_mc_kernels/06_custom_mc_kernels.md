# Plan 06: NCCL's example multicast AllReduce kernels at the series settings

**Status (2026-11-18): ran on 2026-11-17, 23:14–23:57 UTC, all 8 GPUs; 196 runs, 0 rejected. Report: [README.md](README.md).** The harness additions below were made before the run. Deviations, recorded in the report: the CTA sweep ran 1–256 (256 added after 8 GPUs still rose at 128; the poison kernel still rises at 256, near the 264 resident CTAs); payload runs start at 16 MiB and discard it (the NCCL column was sometimes slow at the first size); link pairs ran at 4 and 8 GPUs only, as planned. Expected signatures below are kept as written; the results confirm the 4-times CTA count for two-shot and the factor N for one-shot.

**Matrix cell:** NVLS-based collective × request generation and concurrency (how many multimem loads each thread keeps in flight) and synchronization and progress (end barrier, poison wait, or none). There is no injected problem; the plan compares implementations of the same AllReduce.

**Question:** at 2 to 8 GPUs and payloads of 32 MiB to 2 GiB per GPU, how fast are the three example kernels of [`contrib/custom_algos/allreduce`](../../../../../contrib/custom_algos/allreduce/README.md) against NCCL's library kernel, and what limits each one?

Arms, metrics and run rules of the series are in [00_common_setup.md](../00_common_setup.md). The upstream README reports only latency up to 1 MB on GB200; this plan measures large payloads on H100.

## The kernels under test

All four use symmetric windows (`ncclCommWindowRegister`) and the multicast address of the window. Threads/CTA is 1024 for the three example kernels (fixed in their source).

| Arm | Code | What one thread does | Part of the buffer each GPU handles | Sync across GPUs in the kernel |
|---|---|---|---|---|
| One-shot multicast | [`oneshot_mc_allreduce_kernel.cuh`](../../../../../contrib/custom_algos/allreduce/oneshot_mc_allreduce_kernel.cuh) | `multimem.ld_reduce` of 16 B, then a plain store to its own output | **All of it**: every GPU computes every sum | None |
| Two-shot multicast, end barrier | [`twoshot_mc_simple_allreduce_kernel.cuh`](../../../../../contrib/custom_algos/allreduce/twoshot_mc_simple_allreduce_kernel.cuh) | `ld_reduce` of 16 B, then `multimem.st` of 16 B; one load in flight at a time | One contiguous 1/N | Barrier at the end |
| Two-shot multicast, poison | [`twoshot_mc_poison_allreduce_kernel.cuh`](../../../../../contrib/custom_algos/allreduce/twoshot_mc_poison_allreduce_kernel.cuh) | Same as above | One contiguous 1/N | None in steady state; each GPU waits until every output element is no longer the poison value, and three output buffers rotate |
| NCCL library (reference) | `ncclAllReduce` on the same windows, inside the same program; forced with `NCCL_SYM_KERNEL=allreduce:AllReduce_RSxLDMC_AGxSTMC` | 8 `ld_reduce` of 16 B in flight, then 8 `multimem.st` | 4 KiB pieces dealt to the GPUs in turn | Barrier at the start and at the end |

## How this harness differs from plans 01–04

The example program is its own harness. Its numbers are compared first with its own NCCL column, which runs in the same process layout and timing; comparison with nccl-tests numbers of plans 01–04 is a secondary check.

| Item | Plans 01–04 (nccl-tests) | This plan (example program) |
|---|---|---|
| Process layout | One process, N GPUs | One MPI process per GPU (`mpirun -np N`) |
| Timing | Mean of 20 calls on rank 0 | One CUDA graph of 10 back-to-back calls (sizes ≥ 2 MiB), largest time over ranks; 5 warm-up calls |
| Cache | Not flushed | A 128 MiB memset before each kernel's warm-up |
| CTA count | Per arm, chosen by NCCL or set by a knob | One count (`-c`/`-C`, powers of 2) shared by the three example kernels in a run; the NCCL column chooses its own |
| Correctness | Every size, every run | Built-in test only at N, 64 N and 1024 N lines of 16 B (at most 128 KiB) |
| Sizes | Any | Upstream stops at 128 MiB; the build raises the limit to 2 GiB and changes nothing else |
| Two-shot sizes | Exact | Rounded down to a multiple of N lines; the printed algBW uses the rounded size. At 6 GPUs, 2 GiB loses 2 lines (32 B) |
| Input ordering | NCCL orders the calls | The example kernels have no barrier before reading. Correct here only because the input does not change during timing |

### Harness additions needed before the run

Each is a change to the copied `main.cu` in `common/bin/custom_mc/` made by the build script, never to `contrib/`.

1. **Result check at every measured size.** After the timed calls of each kernel and size, compare the output with the expected value (the input is 1.0 on every GPU, so every element must equal N) and print the number of wrong elements. Without it, a fast but wrong run could count.
2. **Run one kernel only (environment variable).** The link-byte step (E6.3) reads whole-run counters, which cannot be split between four kernels in one run.
3. **Machine-readable output.** One line per kernel, size and CTA count, so the runner writes JSONL like the other plans.

A runner `run_custom_mc.py` (GPU locks, idle guard, before/after snapshots, as in [`run_allreduce.py`](../../common/scripts/run_allreduce.py)) wraps `mpirun`.

## E6.0 Check

### Motivation

The device API may refuse multicast at 2 GPUs (NCCL's library turns it off there), and the NCCL column must run the intended kernel.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **GPU count: 2, 4, 6, 8** (main sets of the series) |
| Sizes | 32 MiB and 2 GiB per GPU |
| CTAs × threads/CTA | Example kernels: 4 × 1024. NCCL column: NCCL's choice |
| Logging | `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,TUNING` |

### Procedure

1. One run per GPU count.
2. A run counts if: the built-in test prints PASS for all three kernels; the added result check finds 0 wrong elements; the NCCL log shows `-> Kernel AllReduce_RSxLDMC_AGxSTMC` for the NCCL column (not at 2 GPUs, where the library kernel is not available; there the NCCL column is recorded as whatever NCCL picks and is not used as a reference).
3. Record whether multicast works at 2 GPUs. If not, the example kernels are measured at 4, 6 and 8 GPUs only.

### Privileged access

Free GPUs of each set. Nothing else.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| S3 | Kernel names as above | A wrong NCCL kernel from the intended one |
| A11 | 0 wrong elements | |

## E6.1 CTA calibration

### Motivation

The two-shot example kernels keep one 16 B load in flight per thread: 16 KiB per 1024-thread CTA. NCCL's library kernel keeps 8 per thread: 64 KiB per 512-thread CTA. If the CTA count is set by bytes in flight, the two-shot example kernels need about 4 times as many CTAs as the library kernel (about 8 at 8 GPUs, where the library kernel needs 2). This tests the explanation of the CTA counts discussed in the walkthrough.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **CTAs: 1, 2, 4, 8, 16, 32, 64, 128** × 1024 threads/CTA (128 is about one per SM pair on the 132-SM H100) |
| Second variable, each value a separate curve | GPU count: 8 first, then 4, 6, 2 (if E6.0 allows) |
| Sizes | 32 MiB and 2 GiB per GPU (one run steps through 32 MiB to 2 GiB, factor 2) |
| Rounds | 3 |

### Procedure

1. For each GPU count, one run per CTA count, CTA counts in rotated order inside each round.
2. Flat part: as in plan 01 (less than 1% gain over two successive increases). A curve still rising at 128 CTAs is an incomplete calibration.
3. Output: the smallest flat CTA count of each example kernel at each GPU count, used in E6.2.

### Privileged access

Free GPUs; 8 GPUs means the whole machine.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| S1 against CTAs | Two-shot kernels flat from about 4 times the library kernel's CTA count; one-shot flat early, because it is limited by link traffic, not by loads in flight | In-flight limit from link limit |
| S3 | As E6.0 | |

## E6.2 GPU count and payload

### Motivation

Plans 02 and 03 gave the library kernel's rate against GPU count and payload. This step puts the example kernels on the same axes. One-shot sends N × S per GPU (each GPU answers every GPU's loads for the whole buffer), so its large-payload algBW should fall roughly as 1/N; two-shot sends S + S/N, like the library kernel.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Kernel: one-shot, two-shot end barrier, two-shot poison, NCCL library** |
| Second variable, each value a separate comparison | GPU count 2 (if allowed), 4, 6, 8 on the main sets; payload 32 MiB to 2 GiB per GPU in one run |
| CTAs × threads/CTA | Example kernels: flat CTA count from E6.1 × 1024. NCCL column: NCCL's choice (recorded from the log) |
| Rounds | 5 |

### Procedure

1. 8 GPUs first, then 6, 4, 2.
2. One run per round and GPU count. The program measures the four kernels one after another at each size; the order is fixed by the program and cannot be rotated without changing it (recorded as a limitation).
3. Analysis: algBW and busBW per kernel; ratio of each example kernel to the NCCL column in the same run; NCCL column against the plan 02 and 03 values for the same GPU set and size (cross-harness check).

### Privileged access

As E6.1.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| S1 | Two-shot kernels close to the NCCL column at 2 GiB, if E6.1's CTA count is used; one-shot far below at large sizes and falling with N (link limit 478/N GB/s algBW if it sends N × S) | Two-step against one-step design at large payloads |
| S1, 32 MiB | Poison kernel at or above the end-barrier kernel; the barrier costs a few µs, small against a call of more than 100 µs | Whether sync cost matters at these sizes |
| S6 | busBW factors: one-shot N (sent bytes), others 1 + 1/N; to be confirmed in E6.3 | |
| A11 | 0 wrong elements | |
| B2 | No clock or power limit events | |

## E6.3 Link bytes of one-shot and two-shot

### Motivation

The factor N for one-shot assumes that the switch fetches each element once for every GPU that asks for it. If the switch combined identical requests, one-shot would send less. Only the link counters can tell.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Kernel: one-shot, two-shot end barrier** (one kernel per run, harness addition 2) |
| GPU count | 4 and 8 |
| Size | 2 GiB per GPU, CTA count from E6.1 |
| Rounds | 3, as long-minus-short pairs (plan 04 method) |

### Procedure

1. Plan 04 method: counters before and after a long and a short run, difference ÷ (payload × extra calls).
2. Compare with N and 1 + 1/N.

### Privileged access

As E6.1. `nvidia-smi nvlink -gt d` is read-only.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| S4 | One-shot: N sent, 1 received per payload byte. Two-shot: 1 + 1/N each way | Whether the switch combines identical loads |
| S5 | Share of the 478 GB/s rating | Whether one-shot is at the link limit |

## Run order and time

| Step | GPUs | Runs | Time (estimate: about 30 s per run including MPI start-up, not yet measured) |
|---|---|---|---|
| Harness additions and build | none | | about 1 hour of code work, no GPU |
| E6.0 | each GPU count | 4 | 3 min |
| E6.1 at 8 GPUs | 0–7 | 8 CTA counts × 3 rounds = 24 | 12 min |
| E6.2 at 8 GPUs | 0–7 | 5 | 3 min |
| E6.3 at 8 GPUs | 0–7 | 2 kernels × 3 pairs = 12 | 6 min |
| E6.1, E6.2, E6.3 at 6, 4, 2 GPUs | subsets | about 100 | 50 min |
