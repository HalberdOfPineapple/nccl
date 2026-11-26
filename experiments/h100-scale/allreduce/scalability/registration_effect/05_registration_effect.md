# Plan 05: classic NVLS with and without registration, against the symmetric kernel

**Status (2026-11-17): plan only. Nothing has run. Needs a GPU allocation from the user.**

**Matrix cell:** NVLS-based collective implementation × request generation and concurrency (how many warps issue multimem) and synchronization and progress (per-step flags). There is no injected problem; the plan compares three implementations of the same AllReduce.

**Question:** of the gap between classic NVLS and the symmetric multicast kernel, how much comes from the copies through NCCL's buffer (with only 4 of 20 warps issuing multimem), and how much from the per-step synchronisation?

Arms, command, metrics and run rules are in [00_common_setup.md](../00_common_setup.md). Background: §6.2, §9.1 and §10 of the [NVLS AllReduce walkthrough](../../../../../notes/nvls_allreduce_walkthrough.html).

## Why three arms

| Arm | Copies through NCCL's buffer | Warps issuing multimem per CTA (from the code) | Per-step flags across GPUs |
|---|---|---|---|
| NVLS, unregistered (as in plans 01–04) | yes | 4 of 20 | yes |
| **NVLS, registered** (new) | no | 17 of 20 | yes |
| Symmetric multicast (as in plans 01–04) | no | 16 of 16 | no (one barrier at each end) |

- NVLS unregistered against NVLS registered isolates the copies and the warp count; the pipeline and its flags are the same.
- NVLS registered against symmetric multicast isolates the per-step flags; neither copies.
- The warp counts come from [`all_reduce.h:396-399`](../../../../../src/device/all_reduce.h) and [`prims_simple.h:626`](../../../../../src/device/prims_simple.h). They are expectations from the code, not measurements.

## E5.0 Registration check

### Motivation

NCCL falls back to the copying path without an error when NVLS registration does not take effect. In this harness, one process drives all GPUs, and NCCL turns NVLS registration off in that case unless told otherwise ([`init.cc:1342-1355`](../../../../../src/init.cc), [`group.cc:680`](../../../../../src/group.cc)). A run that silently fell back would look like "registration does not help".

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **GPU count: 4, 8**, each at 32 MiB and 2 GiB per GPU |
| Arm | NVLS, registered: `NCCL_ALGO=NVLS NCCL_PROTO=Simple`, `-R 1` (`ncclCommRegister`), `NCCL_SINGLE_PROC_MEM_REG_ENABLE=1` set before communicator init |
| Logging | `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,TUNING,NVLS,REG` |
| Rounds | 1 per cell |

### Procedure

1. One run per cell.
2. A run counts only if: the log names `Algo NVLS proto SIMPLE`; every rank prints `NVLS UB registered` and none prints `NVLS UB fell back for` or `using internal buffer path` ([`nvls_ub.cc:417`](../../../../../src/transport/nvls_ub.cc), [`nvls_ub.cc:510-512`](../../../../../src/transport/nvls_ub.cc)); 0 wrong values.
3. Record CTAs chosen by NCCL. Threads per CTA are 640 from the code.
4. Stop rule: if registration falls back in any cell, do not run E5.1 for that GPU count. Record the log lines and investigate first (allocation through `ncclMemAlloc`, the registration range `NCCL_NVLS_UB_SIZE`, which by default covers the GPU memory, [`nvls_ub.cc:103-114`](../../../../../src/transport/nvls_ub.cc)).

### Privileged access

Free GPUs of each set. Nothing else.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| S3 | `Algo NVLS proto SIMPLE` and the `NVLS UB registered` line on every rank | A registered run from a silent fallback |
| A11 | 0 wrong values | |

## E5.1 Three-way comparison

### Motivation

LLM training and inference move both small (tens of MiB) and large (GiB) AllReduce payloads. Plans 01–03 showed that the symmetric kernel needs 2 CTAs where NVLS needs 16, and is 50% faster at 32 MiB on 8 GPUs, but those runs changed two things at once. Throughput alone cannot say which change mattered.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Arm: NVLS unregistered, NVLS registered, symmetric multicast** |
| Second variable, each value a separate comparison | Payload per GPU: 32, 64, 128, 256, 512 MiB, 1, 2 GiB (one run steps through all, factor 2) |
| GPU count and set | 4 GPUs = {4,5,6,7}; 8 GPUs = {0,…,7}. 8 GPUs first |
| CTAs × threads/CTA | Chosen by NCCL in the main runs (recorded per run). A CTA sweep (2, 4, 8, 16, 24, 32) at 32 MiB and 2 GiB for the registered arm only, to see whether 17 issuing warps per CTA lower the CTA count it needs |
| Fixed | `float` sum, 5 warm-up and 20 timed calls, 1 check |
| Rounds | 5, arms rotated inside each round |

All three arms use buffers from `ncclMemAlloc`, so allocation is the same; only registration differs (`-R 0`, `-R 1`, `-R 2`).

### Procedure

1. E5.0 passes for the GPU count.
2. Main runs: 3 arms × 2 GPU counts × 5 rounds = 30 runs, each stepping through all payloads.
3. CTA sweep: registered arm × 2 payloads × 6 CTA counts × 2 GPU counts × 3 rounds = 72 runs.
4. Link bytes: one long-minus-short pair (plan 04 method) for the registered arm at 2 GiB per GPU count, 3 rounds, to confirm 1 + 1/N.
5. Separate profiler pass after the timing runs, only if time allows: one `ncu` run per arm at 32 MiB and 2 GiB on 8 GPUs, DRAM bytes (C1) and per-warp stall reasons.
6. Analysis: algBW and busBW per arm and payload; the ratios registered ÷ unregistered and symmetric ÷ registered at each payload; the 95% payload of each arm.

### Privileged access

Free GPUs; 8 GPUs means the whole machine. The `ncu` pass needs counter access (check `ncu` permission before the session; without it, skip step 5 and say so).

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| S1 | Registered NVLS between the other two at 32 MiB. If it reaches the symmetric kernel, the copies and warp count were the cost; if it stays near unregistered NVLS, the per-step flags were | Copy and warp-count cost from flag cost |
| S1, CTA sweep | Registered NVLS needs fewer CTAs than 16 to reach its rate | Whether issuing warps were the reason NVLS needed 16 CTAs |
| S4 | 1 + 1/N for all three arms; registration does not change link bytes | |
| S6 | All three about 300–315 GB/s at 2 GiB, as for plans 02 and 04 | Whether the large-payload ceiling is the same for the registered path |
| C1 (if profiled) | DRAM bytes per payload byte about 6 for unregistered NVLS, about 2 for the other two | Confirms or rejects the byte-count estimate of the walkthrough's §10.4 |
| B2 | No clock or power limit events | |

## Run order and time

| Step | GPUs | Runs | Time (estimate from plans 01–04: about 14 s per fixed-size run, 27 s per payload sweep) |
|---|---|---|---|
| E5.0 at 8 GPUs | 0–7 | 2 | 1 min |
| E5.1 at 8 GPUs | 0–7 | 15 sweeps + 36 CTA + 6 link | about 17 min |
| E5.0 and E5.1 at 4 GPUs | 4–7 | same | about 18 min |
| Profiler pass (optional) | 0–7 | 6 | about 15 min |
