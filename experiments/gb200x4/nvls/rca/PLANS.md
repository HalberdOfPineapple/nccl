# Root-cause series on GB200: does concurrent `multimem.st` and `multimem.ld_reduce` lose link time, and why

**Status (2026-11-26): plan 01 E1.1–E1.4 done on one node per step. The degradation exists on GB200: the multicast AllReduce reaches 0.82 of its halves' busBW at 64 MiB – 8 GiB and gains only 2–13% over Ring. Link counters: case B by the plan's rule (raw link rate 3.5% lower), with the case-A mechanism (non-data bytes on busy directions, 1.25 against 1.062 raw bytes per data byte) explaining about four fifths of the loss. Results: [degradation check README](degradation_check/README.md). Pending: E1.3 repeat on a second node, E1.5. Plan 02 not started.**

## Goal

The H100 series explained why NVLS AllReduce (concurrent `multimem.st` and `multimem.ld_reduce`, the AllGather and ReduceScatter halves of the collective run at the same time) moves less data per link direction than NVLS AllGather or NVLS ReduceScatter alone. This series repeats the question on GB200, where the link, the switch and NCCL's kernel choices are different, so the H100 answer cannot be assumed.

The series has two steps, in this order:

1. **Measure first.** Show, with completed-collective throughput against matched references, whether the degradation exists on GB200 at all, how large it is, and whether each link direction is then at a common ceiling. This is plan 01. It is a gate: plan 02 runs only if plan 01 finds a degradation, and its form depends on what plan 01 finds.
2. **Then explain.** Find which traffic takes the link time (or other resource) that the data loses. This is plan 02.

The common question of the series: **on GB200 at 4 GPUs, does NVLS AllReduce move less data per link direction than its two halves alone, and if so, what takes the time the data loses: non-data packets on a full link (the H100 answer), idle link time, or a limit before the link?**

## What the H100 series established (reference only, not assumed here)

| Finding on H100 (4 and 8 GPUs, NVLink4, 18 links × 25 GB/s per direction) | Where |
|---|---|
| NVLS AllReduce busBW is 0.865 of the mean of NVLS AllGather and ReduceScatter at 4 GPUs; plain load/store mixes lose only about 3% | [H100 observation report](../../../h100x4/nvls/observation/multimem_load_store_mix/README.md) |
| Every pattern runs its busy direction at the same raw link rate, 425–427 GB/s at 4 GPUs (95% of the 450 GB/s nominal) | [H100 plan 01 report](../../../h100x4/nvls/rca/multimem_contention/README.md) |
| NVLS AllReduce spends 1.34 raw bytes per data byte on each direction; AllGather 1.135, ReduceScatter 1.189, Ring AllReduce 1.16 | Same |
| The extra bytes are read requests that the switch sends to every GPU for `ld_reduce` (0.23 bytes per payload byte, received side) and one 16-byte response per 128-byte `multimem.st` packet (0.125, sent side). Alone, each operation puts these on its low-data direction, where they cost nothing; mixed, both directions are full and they take data time | Same |
| Neither operation has priority on the link; each one's share follows its CTA count. Warps are merged into 128-byte packets at every instruction width | Same |

## Why GB200 may differ

| Item | H100 machine | GB200 (Empire AI Beta) | Consequence for this series |
|---|---|---|---|
| Link | NVLink4, 18 links × 25 GB/s = 450 GB/s per direction | NVLink5, 18 links × 53.125 GB/s = 956 GB/s per direction (`nvidia-smi nvlink -s`) | The link is twice as fast; whether the GPU can fill it is open |
| Switch | NVSwitch in the same 8-GPU box | NVL72 rack: the 4 GPUs of a node go through switch trays shared with the other 17 nodes of the rack | Other jobs in the rack may load the switch; this cannot be controlled or observed from the node |
| SMs | 132 | 152 | Harness needs `maxSms` and SM-to-GPC maps for GB200 |
| NCCL default at 1 GiB, 4 GPUs (pilot) | NVLS kernels chosen | Ring chosen for AllReduce from 256 MiB up | The tuner already "knows" NVLS does not win; the gain over Ring is a main result |
| Symmetric kernels | `AllReduce_RSxLDMC_AGxSTMC`, `RSxLD_AGxST` | Also `AllReduce_RSxTmaLD_AGxTmaST` (TMA plain loads and stores, compute capability ≥ 10.0) | An extra non-multicast AllReduce reference |
| Hardware packet counters (Nsight Systems GPU metrics) | Available (root) | **Not available**: `RmProfilingAdminOnly=1` and sessions run as a normal user | Packet-type split (request/response, data/protocol) needs an admin; driver data and raw counters (`nvidia-smi nvlink -gt d/r`) work without it |
| GPU time | Long allocation, fixed GPUs 4–7 | `beta_test` sessions of at most 2 hours, 1 node of 4 GPUs, one of 4 nodes in racks 2 and 3 | Every comparison must be complete inside one session; results are replicated across sessions and nodes |

## Plans

| Plan | Question | **Controlled variable** | Runs (about) | Needs |
|---|---|---|---|---|
| [01](degradation_check/01_degradation_check.md) | Does the degradation exist on GB200, how big is it, and is each busy direction at a common ceiling? | **Collective and kernel; CTAs per kernel (calibration); payload size** | about 470, in 3–4 sessions of 2 h | `beta_test` sessions; harness port for E1.5 |
| [02](multimem_contention/02_multimem_contention.md) | What takes the time the data loses? Branch chosen by plan 01 | **Issue rate, instruction width and address pattern, CTA share per operation, background traffic; packet counters if granted** | about 250, in 3 sessions | Plan 01 gate passed; packet counters need admin help |

### Decision after plan 01

| Plan 01 finds | Meaning | Plan 02 |
|---|---|---|
| A. AllReduce loses data rate against its halves **and** its busy directions run at the same raw link rate as the halves | Link full; loss is non-data bytes (the H100 case) | Branch A: account for the extra raw bytes (E2.2–E2.5) |
| B. AllReduce loses data rate **and** its raw link rate is lower than the halves' | Link not full in the mix; loss is idle link time or a limit before the link | Branch B: find the limit (E2.1, E2.5, E2.6) |
| C. No loss beyond 3% (the plain-mix loss on H100) at the calibrated configuration | No degradation on GB200 at 4 GPUs | Stop; report the negative result. Optionally the 8-GPU check (E2.7) |

In every branch, plan 01 also records how far the ceiling of every pattern is below the 956 GB/s nominal. A common ceiling far below nominal is a separate finding: something other than the link limits every pattern.

## Where this series sits

| Item | Choice | Note |
|---|---|---|
| Tree | `experiments/gb200x4/` | 4 GPUs = one GB200 node. An 8-GPU check (two nodes of one NVL72 rack) would go in `gb200x8/` with a short report linking back |
| Data path | `nvls` | Same row of the [problem matrix](../../../../notes/scale_up_communication_problem_matrix.md) as the H100 series |
| Series | `rca` | Measures the effect first (plan 01), then explains it (plan 02) |
| Harness | nccl-tests sessions: [`gb200x4/nvls/common`](../common/README.md) (batch launcher, container script, runner, summary, figures). Role loops (`mm_mix`): [`h100x4/nvls/common`](../../../h100x4/nvls/common/) after the port in plan 01 E1.5 | The nccl-tests runner is GB200-specific; `mm_mix` stays one harness for both GPU types. The port adds GPU list, SM limit and target architecture as settings; H100 defaults stay |
| NCCL | `build-gb200/` at the repository root (NCCL 2.32.3, `sm_100`, CUDA 12.9.86), not `build/` | Home is shared with the x86_64 H100 cluster; separate build directories keep the two from overwriting each other |
| nccl-tests | `/mnt/home/wli10/nccl-tests/build-gb200/` (2.22.0, commit `826b3e6`), linked against `build-gb200` | Run with `LD_LIBRARY_PATH=<repo>/build-gb200/lib`, otherwise the container's NCCL 2.27.3 is loaded |

## Rules for this series

The rules of the [H100 root-cause series](../../../h100x4/nvls/rca/PLANS.md#rules-for-this-series) and the [experiment conventions](../../../CLAUDE.md) apply. In addition:

1. **Measure before explaining.** No mechanism is assumed from H100. Plan 02 starts only after plan 01's gate.
2. **One session is one cohort.** Every compared pair (healthy and perturbed, arm and reference, long and short run) runs in the same 2-hour session on the same node. Never pair runs across sessions.
3. **Replicate across sessions and nodes.** A headline number needs at least two sessions, on at least two different nodes when available. Report the spread between sessions next to the spread inside a session.
4. **Record the node and the rack in every run record.** The rack's switch trays are shared with other jobs; a session whose results differ from the others is kept and flagged, not dropped silently.
5. **Kernel choice is verified, not assumed.** A forced-kernel run counts only if NCCL's log names the forced kernel or algorithm.
