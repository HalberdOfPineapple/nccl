# Plan 01: AllGather and ReduceScatter with symmetric multicast, 2 to 8 GPUs

**Status (2026-11-18): ran on 2026-11-18, 02:16–06:05 UTC, all 8 GPUs; 1,062 runs, 0 rejected. Report: [README.md](README.md).** The harness extension below was made and checked before the runs. Deviations, recorded in the report: fixed CTA settings were added to E1.2 and E1.3 after calibration, as the E1.1 rule asks (plain load/store 32 CTAs; Ring and classic NVLS 64 CTAs); the Ring calibration is incomplete at 64 CTAs (NCCL's limit); the reverse GPU-count repeat was not run. Expected signatures below are kept as written: the link model and the few-CTA property were confirmed; the expectation that Ring and plain busBW would be about 360 GB/s holds for Ring at 64 CTAs; multicast is ahead of classic NVLS at 32 MiB and behind Ring at 8 GiB, as expected.

**Matrix cell:** NVLS-based collective (and, as references, Ring and thread-issued peer loads and stores) × GPU-count scaling. No injected problem; the plan compares implementations.

**Question:** for AllGather and ReduceScatter, how does the rate of each implementation change from 2 to 8 GPUs and from 32 MiB to 8 GiB, and does symmetric multicast keep the advantages it showed for AllReduce (full rate with few CTAs, high rate at 32 MiB) even though it needs more link bytes in its busiest direction than Ring for these collectives?

Arms, payload definition and byte counts are in [00_common_setup.md](../00_common_setup.md); everything else follows the [AllReduce series setup](../../../allreduce/scalability/00_common_setup.md).

## Harness extension (before any GPU run)

| Change | File | Constraint |
|---|---|---|
| A `collective` field per cell (`allreduce` default, `allgather`, `reducescatter`) choosing the nccl-tests binary, the `NCCL_SYM_KERNEL` prefix and the log line to check | [`run_allreduce.py`](../../common/scripts/run_allreduce.py) | AllReduce runs and their output keys unchanged; check by re-summarizing one recorded AllReduce session with the new code |
| Arms per collective (table in the common setup) | same | |
| busBW factor per collective and arm | [`summarize.py`](../../common/scripts/summarize.py) `bus_factor` | AllReduce factors unchanged |
| Link model per collective | [`link_bytes.py`](../../common/scripts/link_bytes.py) | Sent and received reported separately, since they differ for multicast |
| Figures with collective in titles | [`make_figures.py`](../../common/scripts/make_figures.py) | |

## E1.0 Selection check

### Motivation

Each arm must run the intended algorithm or kernel at each GPU count; for AllReduce, NCCL also ran classic NVLS at 2 GPUs, which was not expected.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Cell: collective (AllGather, ReduceScatter) × arm (4) × GPU count (2, 4, 6, 8)**, less symmetric multicast at 2 GPUs: 30 cells |
| Payload | 2 GiB |
| CTAs | NCCL's choice |

### Procedure

1. One run per cell, 8 GPUs first.
2. A run counts if the log names the intended algorithm or kernel and no other, and both passes report 0 wrong values.
3. Record CTAs × threads/CTA from the log.

### Privileged access

Free GPUs of each set.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| S3 | Intended algorithm or kernel; classic NVLS may or may not run at 2 GPUs | A wrong selection from the intended one |
| A11 | 0 wrong values | |

## E1.1 CTA calibration

### Motivation

For AllReduce, NCCL's CTA choice was on the flat part for every arm except plain load/store at 2 and 4 GPUs. The same must be checked for these collectives before comparing rates. It also tests whether the multicast kernels again need far fewer CTAs than Ring.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **CTAs: NCCL's choice, 2, 4, 8, 16, 32, 64** |
| Curves | Every cell at 8 and 4 GPUs (16 cells); at 6 and 2 GPUs only NCCL's choice and twice it, to check that doubling gains less than 2% (14 cells) |
| Payload | 2 GiB |
| Rounds | 3 |

### Procedure

1. Flat part: as in AllReduce plan 01 (less than 1% gain over two successive increases). A curve still rising at 64 CTAs is incomplete.
2. If NCCL's choice is below 0.95 of the highest value in any cell, add a fixed-CTA setting for that arm in E1.2 and E1.3, as the AllReduce series did for plain load/store.

### Privileged access

As E1.0.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| S1 against CTAs | Symmetric multicast flat from 2–4 CTAs, Ring and classic NVLS from 16–24, as for AllReduce | Whether the few-CTA property carries over |

## E1.2 GPU-count scaling at 2 GiB

### Motivation

The busiest-direction bytes per payload byte grow with N for Ring and plain ((N − 1)/N) but stay 1 for multicast. If both were limited only by the link, Ring's algBW would fall from 2 to 8 GPUs by a factor 0.5/0.875, while multicast stays flat; Ring should then lead at every N ≥ 4 by N/(N − 1).

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **GPU count: 2, 4, 6, 8** (main sets {4,5}, {4–7}, {1–6}, {0–7}) |
| Curves | Collective × arm × CTA setting (NCCL's choice, plus any fixed setting from E1.1) |
| Payload | 2 GiB |
| Rounds | 5, arms rotated |

### Procedure

1. GPU counts in the order 8, 6, 4, 2 (the AllReduce series showed drift between sessions below 0.5%).
2. Report algBW, busBW with the arm's factor, and the scaling ratio against the link-only expectation.

### Privileged access

As E1.0; 8 GPUs means the whole machine.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| S1, S2 | Ring falls with N as 1/((N − 1)/N); multicast flat in N | Link-limited arms from others |
| S6 | Ring and plain about 360 GB/s (the per-GPU sending rate measured for AllReduce Ring and the one-shot kernel); multicast below, if its busiest direction is received bytes | Whether sending and receiving have the same limit |

## E1.3 Payload size

### Motivation

FSDP-style AllGather and ReduceScatter calls are per layer: often tens to hundreds of MiB. For AllReduce, multicast was 50% faster than classic NVLS at 32 MiB on 8 GPUs.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Payload S: 32 MiB, 128 MiB, 512 MiB, 2 GiB, 8 GiB** (one run steps through all, factor 4, as AllReduce plan 03) |
| Comparison | Each GPU count and collective separately |
| CTAs | NCCL's choice (recorded per size) |
| Rounds | 5 |

### Procedure

1. One run per cell and round.
2. Report algBW against payload and the payload where each arm reaches 95% of its 8 GiB rate.

### Privileged access

As E1.2.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| S1 | Multicast ahead of classic NVLS at 32 MiB; Ring ahead of multicast at 8 GiB | Small-payload advantage from large-payload limit |

## E1.4 Link bytes per payload byte

### Motivation

The byte counts in the common setup say multicast needs S in its busiest direction for these collectives, because the GPU's own piece goes through the switch. Only the link counters can confirm it.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Arm** (all four at 8 GPUs; Ring and symmetric multicast at 4 GPUs) |
| Collective | AllGather and ReduceScatter |
| Payload | 2 GiB, NCCL's CTA choice |
| Runs | Long (200 calls) and short (20) pairs, 3 rounds (AllReduce plan 04 method) |

### Procedure

1. Counters before and after each run; long minus short, per GPU and direction, ÷ (2 × 180 × S) (in-place and out-of-place passes).
2. Compare with the table in the common setup.

### Privileged access

As E1.2. `nvidia-smi nvlink -gt d` is read-only.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| S4 | AllGather multicast: 1/N sent, 1 received; ReduceScatter multicast: 1 sent, 1/N received; Ring and plain (N − 1)/N each way | Whether the own piece crosses the link |
| S5 | Share of 478 GB/s in the busiest direction | Which arm is at the link limit |

## Run order and time

Estimates from the AllReduce series (about 12 s per fixed-size run, 25 s per payload sweep); not measured for these collectives.

| Step | Runs | Time |
|---|---|---|
| Harness extension and check against recorded AllReduce data | none | about 1 hour, no GPU |
| E1.0 | 30 | 6 min |
| E1.1 | 16 cells × 7 CTA settings × 3 + 14 cells × 2 × 3 = 420 | 85 min |
| E1.2 | 30 × 5 = 150 | 30 min |
| E1.3 | 30 × 5 = 150 | 65 min |
| E1.4 | 2 collectives × 6 cells × 2 × 3 = 72 | 20 min |
| Total GPU time | about 820 | about 3.5 hours; 8-GPU parts of all steps first (about 1.5 hours) |
