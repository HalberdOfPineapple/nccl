# Observation series: observing TMA pushes in scale-up communication

## Goal

Find which observations explain performance changes when a GPU pushes data to a peer GPU with the Hopper bulk-copy unit (TMA) instead of with thread stores. The series reuses the thread-store series' roles, completion protocol, placement method and co-runners, so a result here can be set beside the matching thread-store result.

This series covers the **TMA-based transfer** row of the [problem matrix](../../../../notes/scale_up_communication_problem_matrix.md), for one route: source shared memory → NVLink → destination HBM, through an NCCL LSA peer pointer. The harness is in [../common/](../common/README.md). Shared setup and metric ids: [00_common_setup.md](00_common_setup.md).

## Plans

| Plan | Problem family | Matching thread-store plan | What it does |
|---|---|---|---|
| [01_data_store_congest.md](data_store_congest/01_data_store_congest.md) | Shared-resource contention: TMA push next to TMA memory co-runners | [Plan 07](../../thread_store/observation/data_store_congest/07_data_store_congest.md) | TMA push from some SMs of the source GPU while TMA co-runners store to (or read from) local memory on other SMs, on the same SMs, or on the destination GPU |

## Rules for this series

The [experiment conventions](../../../CLAUDE.md) and the [thread-store series rules](../../thread_store/observation/PLANS.md#rules-for-exploratory-observation-plans) apply in full. Added for TMA:

1. **One thread issues, the bulk-copy unit moves the bytes.** Timers on the issuing thread measure when requests were accepted and when the unit reported completion, not when bytes crossed NVLink. Name each timer by its boundary (issue, slot free, bulk completion, acknowledgement).
2. **The TMA unit is per SM.** Placement levels are SM, GPC and GPU. Spreading roles over different SMs does not share a TMA unit; putting them in one CTA does.
3. **Compare with thread stores only on matched tuples.** A thread-store result is a reference here only when endpoints, payload, completion protocol, placement and shared-memory reservation match; otherwise it is context, not a baseline.
