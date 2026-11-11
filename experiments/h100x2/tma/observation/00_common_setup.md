# Common setup for the TMA observation series

Everything not listed here is inherited from the [thread-store common setup](../../thread_store/observation/00_common_setup.md): machine, roles, completion protocol, run and analysis protocol, access table, and the metric ids A1–A12, B1–B3, C1–C9. This file states only what differs for a TMA push and defines the metric ids added for it.

## Workload

| Item | Setting |
|---|---|
| Ranks | Two GPUs of one NVLink pair. Rank 0 = issuing GPU `I` (source), rank 1 = destination `D`. No third party |
| Copy | TMA push: thread 0 of each copy CTA starts `cp.async.bulk.global.shared::cta` stores of one chunk each from a ring of shared-memory staging slots into `D`'s window (NCCL LSA peer pointer). The other copy threads fill each slot first with the plan-07 values (position and transfer dependent). No source HBM reads |
| Payload order | Each copy CTA owns a contiguous run of whole chunks, in order |
| Completion | Thread 0 waits for its bulk stores to complete (`cp.async.bulk.wait_group 0`), orders them (`fence.proxy.async.global`, then a system-scope fence), publishes "done"; the receiver on `D` acknowledges after every "done". Only thread 0 writes to the peer, so no other copy thread fences |
| Launch | One CTA per SM on **both** GPUs (cooperative launch, 1024 threads/CTA); roles from per-GPU role tables built from the plan-07 SM-to-GPC maps |
| Shared memory | One reservation for every run of a comparison (196,864 bytes by default), so L1 size never differs between arms |
| Correctness | `D`'s payload filled with 0xFF before each transfer and every word checked after it. Co-runner buffers checked after every launch |

## Metric ids added in this series

A13 and A14 are taken by the thread-store series (plan 08) and are not used here.

| Id | Observation | Definition | Limit |
|---|---|---|---|
| A15 | Copy slot wait | `slot_wait_cycles`: SM clock cycles thread 0 of a copy CTA spent in `cp.async.bulk.wait_group.read` waiting for a staging slot to be read out; reported as a fraction of `issue_loop_cycles` | The bulk-copy unit's read of shared memory, not the NVLink transfer; SM cycles, so a clock change scales it. Detailed mode only |
| A16 | Copy fill time | `fill_cycles`: thread 0's cycles from a free slot to the start of its bulk store (the fill by all copy threads plus two barriers); fraction of `issue_loop_cycles` | Includes the slowest fill warp; zero-cost when slots are filled once (`copy_generate=0`) except for barriers |
| A17 | Bulk completion wait | `bulk_wait_end_ns - issue_end_ns` per copy CTA: from the last bulk store started to `wait_group 0` returning | The unit's report of completed writes; it is not the latency of one chunk, and the ordering fence that follows is in A4 |

In this series A3 (issue loop) ends when thread 0 has started its last bulk store, A4 (ordering) runs from A17's end to the end of the system fence, and A9 for TMA co-runners counts 64 KiB chunks.

## Binary

`bin/tma_congest`, built from [../common/src/tma_congest.cu](../common/src/tma_congest.cu) with `make` in `../common`. Config keys are in the [harness README](../common/README.md).
