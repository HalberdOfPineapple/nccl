# Plan 02: busBW and algBW of AllReduce, AllGather and ReduceScatter, Ring against symmetric multicast

**Status (2026-11-18): analysis plan, done the same day from recorded data; E2.2 added and run the same day (30 runs). Report: [README.md](README.md).**

**Matrix cell:** NVLS-based collective and Ring × GPU-count scaling; cross-cutting analysis, no injected problem and no new runs.

**Question:** with the same GPUs, payload sizes and CTA settings, how fast does each collective drive the NVLink link in its busiest direction when implemented by Ring and by symmetric multicast?

## E2.1 Link rate per collective

### Motivation

nccl-tests' `busbw` uses the Ring factor for every algorithm, so it cannot compare arms. busBW with the measured bytes in the busiest direction is the link rate and can be compared across collectives.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Collective: AllReduce, AllGather, ReduceScatter** |
| Arms | Ring (NCCL's CTA choice; 64 CTAs fixed), symmetric multicast (NCCL's CTA choice) |
| GPU counts | 2, 4, 6, 8, the main sets of both series |
| Payload | 2 GiB and 32 MiB |

### Procedure

1. Read the saved summaries of AllReduce plans 01–03 (2026-11-16) and of plan 01 of this series (2026-11-18) with [`compare_busbw.py`](../../common/scripts/compare_busbw.py).
2. busBW = median algBW × bytes in the busiest direction per payload byte (confirmed by link counters in AllReduce plan 04 and plan 01 of this series).
3. Table and figure per payload size.
4. Added 2026-11-18 on request: algBW per collective, multicast ÷ Ring per collective, and AllReduce against a ReduceScatter followed by an AllGather of the same size (rate 1 ÷ (1/algBW_RS + 1/algBW_AG)), in the same report.

### Privileged access

None.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| S6 | Ring the same for all collectives at enough CTAs; multicast close to Ring at 2 GiB for all collectives | Whether a collective's multicast kernel uses the link as well as the others |

## E2.2 Raw link bytes (added 2026-11-18)

### Motivation

Multicast AllReduce's busBW is 11–24% below the full-duplex expectation (the busBW of multicast AllGather, ReduceScatter and Ring AllReduce on the same GPUs), and no CTA, memory or drift cause fits (report, "Expected busBW of multicast AllReduce"). The data counter does not count read requests and write responses; the raw counter does.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Pattern: Ring AllReduce, Ring AllGather, multicast AllReduce, multicast AllGather, multicast ReduceScatter** |
| GPUs, payload | 8 = {0–7}, 2 GiB, NCCL's CTA choice |
| Runs | Long (200 calls) and short (20) pairs, 3 rounds |
| Counters | `nvidia-smi nvlink -gt d` and `-gt r` before and after each run |

### Procedure

1. Add the raw counters to the run snapshots.
2. Run the pairs; compute data and raw bytes per payload byte per direction.
3. Compare raw link rates and raw ÷ data between patterns.

### Privileged access

All 8 GPUs free. Counters are read-only.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| S4 raw, S5 raw | Same raw rate in the busiest direction for all patterns; raw ÷ data higher for multicast AllReduce | Protocol traffic from a switch-internal limit |

