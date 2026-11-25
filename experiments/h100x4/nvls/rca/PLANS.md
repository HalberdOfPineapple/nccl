# Root-cause series: why concurrent `st_multicast` and `ld_reduce` contend

**Status (2026-11-19): plan 01 done (E1.1–E1.8 on GPUs 4–7, E1.9 on GPUs 0–7). Results: [plan 01 report](multimem_contention/README.md); 8-GPU part in [`h100x8/nvls/rca/`](../../../h100x8/nvls/rca/multimem_contention/README.md).**

## Goal

Explain, at the level of NVLink packets and request scheduling, why a GPU that issues `multimem.st` and `multimem.ld_reduce` at the same time moves 11–24% less data per link direction than when it issues either one alone, although NVLink is full duplex and Ring AllReduce, which fills both directions, does not lose.

The common question of the series: **which packets does each multimem operation put on each link direction, how are they scheduled when both operations share a direction, and which of them takes the link time that the data loses?**

## What is already established

| Finding | Where |
|---|---|
| NVLS AllReduce busBW is 0.86 of the mean of NVLS AllGather and ReduceScatter at 4 GPUs (harness and NCCL kernels); same and separate CTAs lose alike; plain load/store loses only 3% | [Observation report](../observation/multimem_load_store_mix/README.md); peer report section 3.4 (`notes/references/`) |
| When different GPUs issue each operation, read replies keep their rate (0.96) and multicast stores lose 64% on the shared direction | Same report, split mix |
| At 8 GPUs every pattern runs its busiest direction at the same raw rate, 412–423 GB/s; multicast AllReduce needs 1.35 raw bytes per data byte, every other pattern 1.13–1.18 | [busBW comparison, raw link bytes](../../../h100-scale/allgather_reducescatter/scalability/busbw_comparison/README.md#raw-link-bytes) |
| Multicast ReduceScatter receives about 0.9 raw bytes per payload byte of read requests on its low-data direction; multicast AllGather sends about 0.74 of write responses | Same |
| ~~The protocol traffic of the two operations does not add up when they are mixed~~ **Corrected 2026-11-19:** the *raw counter* bytes do not add up, the *packets* do. In multicast AllReduce at 8 GPUs each data-full direction carries about 0.2 non-data bytes per payload byte beyond the data headers, far less than the 0.74 or 0.90 each operation puts on its low-data direction alone. Plan 01 (E1.6, 4 GPUs) showed that the low-data-direction excess is raw-counter traffic outside every packet type (0.55–0.66 per payload byte), which almost vanishes on a busy direction; the request and response packets of the two operations add up in the mix (sent direction to 0.001, received direction +0.077) | 8-GPU raw counters (2026-11-18); [plan 01 report](multimem_contention/README.md#packet-anatomy-of-the-mix) |

## The two problems this series explains

| Problem | Where seen | How it reduces to one quantity |
|---|---|---|
| 1. NVLS AllReduce busBW is below that of NVLS AllGather and ReduceScatter | 4 GPUs: 0.865 of their mean ([observation report](../observation/multimem_load_store_mix/README.md)); 8 GPUs: 311 against 364–373 GB/s per direction | At 8 GPUs every pattern runs its busiest direction at about 420 GB/s raw. Data per direction = raw rate ÷ (raw ÷ data). Multicast AllReduce has raw ÷ data 1.35 against 1.13–1.18 for the others |
| 2. NVLS AllReduce gains less over Ring than the byte counts allow | 8 GPUs: 1.31 times Ring's algBW at 8 GiB ([payload sweep](../../../h100-scale/allreduce/scalability/payload_size/README.md)); 4 GPUs: 1.03; Shouxu's report on GB300 at 4 GPUs: about no gain | Best gain from bytes per payload byte per direction = 2(N − 1)/N ÷ (1 + 1/N): 1.20 at 4 GPUs, 1.56 at 8. Measured ÷ best = 0.86 (4 GPUs) and 0.84 (8 GPUs), the same size as problem 1. At 8 GPUs, Ring and multicast AllReduce run the same raw rate (421 GB/s), so the algBW ratio equals the ratio of raw bytes per payload byte, 2.03 ÷ 1.52 = 1.33 (an identity, not a prediction). With Ring's raw ÷ data of 1.16, multicast AllReduce would need 1.31 raw bytes per payload byte and gain about 1.55 |

So both problems reduce to one question: **why multicast AllReduce spends 1.35 raw bytes per data byte where every other pattern spends 1.13–1.18**.

**Answer from plan 01 (2026-11-19, 4 GPUs):** the same split holds at 4 GPUs (425–427 GB/s raw on every busy direction; 1.34 raw bytes per data byte for AllReduce). Each busy direction of AllReduce carries packets that Ring never sends: the switch's read requests for `ld_reduce` (0.231 bytes per payload byte, received side) and one 16-byte response per 128-byte multicast write packet (0.125, sent side), plus the GPU's own read requests (0.062). Plain stores, which Ring uses, get almost no responses (0.003) and Ring issues no remote reads. NCCL's instruction width and address pattern already give the fewest packets (128-byte packets at any access width); neither operation has priority on the link. E1.9 confirms the same packets with NCCL's kernels on 8 GPUs: Ring needs 1.14 packet bytes per data byte, multicast AllReduce 1.26–1.35, and without the read requests and write responses the gain over Ring would be about 1.45 instead of the measured 1.33 (byte-count bound 1.56); an earlier version said 1.54 from the packet counters only (corrected 2026-11-21, see the [plan 01 summary](multimem_contention/README.md#summary-and-takeaways)).

## Where this series sits

| Item | Choice | Note |
|---|---|---|
| Tree | `experiments/h100x4/` | The question does not need a GPU-count sweep; 4 GPUs reproduce the effect. An 8-GPU check with NCCL kernels is part of plan 01 |
| Data path | `nvls` | Same row of the problem matrix as the observation series |
| Series | `rca` | Root-cause analysis of the effect found in `observation` |
| Harness | [`../common/`](../common/) (`mm_mix`), extended | Role loops shaped like NCCL's symmetric kernels, fixed time window, roles per SM |

## Plans

| Plan | Question | **Controlled variable** | Runs (about) | Needs |
|---|---|---|---|---|
| [01](multimem_contention/01_multimem_contention.md) | Raw link bytes at 4 GPUs, how protocol traffic scales with issue rate, packet anatomy, granularity, arbitration and latency of mixed multimem traffic | **Operation mix, issue rate, instruction width (bytes per thread per memory instruction) and address pattern, share of CTAs per operation, background traffic** | about 260 | GPU allocation; E1.3 onward also harness additions |

## Rules for this series

The rules of the [observation series](../observation/PLANS.md) and the [experiment conventions](../../../CLAUDE.md) apply. In addition:

1. **Expectations come from the link and the instructions, not from the implementation.** A loss that the full-duplex link does not require is a shortfall to explain, never a "reasonable" result.
2. **No circular checks.** A quantity derived from a measurement is never compared with that measurement as if it were a prediction. Every "predicted against measured" pair names its two independent sources.
3. **Every ratio names its numerator and denominator.**
