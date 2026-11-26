# Scalability series: AllGather and ReduceScatter from 2 to 8 GPUs

**Status (2026-11-18): plan 01 ran on 2026-11-18, 02:16–06:05 UTC (allocation: all 8 GPUs for 5 hours from 02:13 UTC); 1,062 runs, 0 rejected.**

## Goal

Measure AllGather and ReduceScatter implemented with NCCL's symmetric multicast kernels (`AllGather_STMC`, `ReduceScatter_LDMC`) at 2, 4, 6 and 8 GPUs and payloads of 32 MiB to 8 GiB, on the same harness (nccl-tests) and GPU sets as the [AllReduce series](../../allreduce/scalability/PLANS.md), against Ring, classic NVLS and the symmetric plain load/store kernels. In LLM training these two collectives carry sharded parameters (AllGather) and gradients (ReduceScatter), for example in FSDP and ZeRO.

The common question: **when the GPU count grows, how does the rate of AllGather and ReduceScatter change for each implementation, and do the multicast kernels keep the advantages they showed for AllReduce (few CTAs, high rate at 32 MiB)?**

## Where this series sits

| Item | Choice | Note |
|---|---|---|
| Tree | `experiments/h100-scale/` | GPU count is the controlled variable |
| Data path level | `allgather_reducescatter` | Two collectives that share arms, sizes and analysis; named after the collectives, as `allreduce` is |
| Harness | [`common/`](../common/) is a link to [`../allreduce/common/`](../../allreduce/common/README.md) | One harness, extended with a collective setting (see the plan). AllReduce behaviour and output keys stay unchanged |

## Plans

| Plan | Report | Question | **Controlled variable** | Runs (about) | Needs |
|---|---|---|---|---|---|
| [01](symmetric_multicast/01_symmetric_multicast.md) | [Report](symmetric_multicast/README.md): multicast is fastest only at 32 MiB on 6–8 GPUs; from 128 MiB Ring (64 CTAs) leads by about N/(N − 1), because the own piece crosses the switch (link counters: 1.000 against (N − 1)/N in the busiest direction) | As above | **Arm × collective × GPU count**; payload as second variable | 1,062 runs | Done |
| [02](busbw_comparison/02_busbw_comparison.md) | [Report](busbw_comparison/README.md): multicast AllReduce is 11–24% below the full-duplex expectation (the busBW of multicast AllGather/ReduceScatter and Ring AllReduce); the lost link time is protocol traffic of its access pattern (1.35 raw bytes per data byte against 1.13–1.18 for all other patterns). With enough CTAs Ring drives the link at 346–367 GB/s for all three collectives; multicast AllGather and ReduceScatter reach that rate at 2 GiB, multicast AllReduce stays 14–24% below; at 32 MiB multicast is 17–37% ahead for AllGather and ReduceScatter | How fast does each collective drive the link in its busiest direction with Ring and with multicast? | **Collective** (AllReduce, AllGather, ReduceScatter), Ring against multicast; raw link counters (E2.2) | 30 (raw-counter check) | Done 2026-11-18 |

## Rules for this series

The rules of the [AllReduce series](../../allreduce/scalability/PLANS.md#rules-for-this-series) apply, with "algBW" defined as nccl-tests defines it for these collectives (see [00_common_setup.md](00_common_setup.md)).

## Decisions

| # | Decision | Choice |
|---|---|---|
| 1 | GPU allocation (2026-11-18) | All 8 GPUs for 5 hours from 02:13 UTC |
| 2 | Harness | The AllReduce harness, extended with a `collective` per cell; checked to reproduce every recorded AllReduce summary (calibration, scaling, payload, link bytes) unchanged |
| 3 | GPU allocation for the raw-counter check (2026-11-18) | All 8 GPUs, about 15 minutes; ran 17:39–17:47 UTC |
