# TMA experiment harness

Build from this directory with `make`, using the in-tree NCCL build (`../../../../build`). The executable `bin/tma_congest` is ignored. CUDA C++17 and `sm_90` are required. The TMA read co-runner comes from [hbm_read_pressure.cuh](../../../h100x1/memory_co_runner/common/include/hbm_read_pressure.cuh).

## TMA congestion harness (plan 01)

`bin/tma_congest CONFIG OUT.json --gpus S,D --gpc-map SOURCE_MAP.json --destination-gpc-map DESTINATION_MAP.json` runs one config on one GPU pair. One process has two host threads, one per GPU, each with its NCCL rank, registered window and cooperative launch of one CTA per SM. Each CTA reads its SM ID and takes the role its GPU's role table gives that SM (copy, co-runner, receiver, idle). Role tables are built as in the [thread-store harness](../../thread_store/common/README.md#store-congestion-harness-plan-07), from the plan-07 SM-to-GPC maps. The harness accepts only GPUs in `authorizedGpu` (the user's allocation for this plan).

Copy CTAs synchronise with named barrier 1 over the fill threads and co-runner threads with named barrier 2, so a co-runner in a copy CTA (`writer_colocated=1`) never waits for the copy, or the other way round.

| Key | Meaning | Default |
|---|---|---|
| `copy_ctas`, `copy_threads` | Copy CTAs × fill threads/CTA (thread 0 also starts the bulk stores) | 16, 512 |
| `copy_chunk_kib`, `copy_chunks_in_flight` | Chunk size (4–128 KiB, power of two) and staging slots per copy CTA (1–8) | 64, 2 |
| `copy_generate` | 1 = fill every chunk with its own values; 0 = fill each slot once (destination check then catches only missing writes) | 1 |
| `copy_gpcs`, `writer_gpcs`, `receiver_gpcs` | Bit mask of GPCs (map numbers) the role may use | 255 |
| `copy_placement`, `writer_placement` | 0 spread, 1 packed | 0 |
| `writer_kind` | Co-runner: 0 TMA store writer, 1 TMA read co-runner, 2 thread-store writer | 0 |
| `writer_ctas`, `writer_threads`, `chunks_in_flight` | Co-runner CTAs; threads/CTA (thread-store writer only); 64 KiB chunks in flight per CTA (TMA kinds, 1–8) | 0, 1024, 1 |
| `writer_colocated` | 1 = one co-runner in the last two warps of each copy CTA; needs `writer_ctas = copy_ctas`, `writer_rank=0`, `copy_threads ≤ block_threads − 64`. TMA kinds use threads 0 and 32 of those warps; the thread-store writer (allowed from 2026-11-10, host check only) needs `writer_threads=64` | 0 |
| `writer_rank` | GPU of the co-runner: 0 source, 1 destination | 0 |
| `target_gbps` | Total target rate of all co-runners, GB/s; 0 = no limit | 0 |
| `writer_idle`, `writer_alternate` | Idle co-runners (reference); alternate idle and active launches inside one run (1 idle first, 2 active first) | 0, 0 |
| `communication`, `solo_ns` | 0 = no copy; copy CTA 0 (source co-runner) or the receiver (destination co-runner) times a window of `solo_ns` | 1, 20 ms |
| `shared_bytes` | Dynamic shared memory per CTA, the same for every CTA; must cover the busiest role | just over half an SM |
| `payload_mib`, `warmup`, `iterations`, `detailed`, `block_threads`, `use_unmapped` | As in `store_congest` | 2048, 3, 12, 1, 1024, 0 |

Output: one JSON per run. Keys follow `store_congest` schema 2 plus `copy_chunk_bytes`, `copy_chunks_in_flight`, `copy_generate`, `writer_kind`, `chunks_in_flight`, `writer_colocated`, `writer_rank`, both GPUs' maps and role tables, and per CTA record `colocated`, `bulk_wait_end_ns` (A17), `slot_wait_cycles` (A15), `fill_cycles` (A16) and `issue_loop_cycles`. Co-runners that share a copy SM are a second record with the same `cta` and `colocated: 1`.

`scripts/run_tma_congest.py PHASE --gpus S,D [--date D]` writes configs under `observation/data_store_congest/configs/` and runs each through the thread-store guard [run_one.py](../../thread_store/common/scripts/run_one.py) (per-device locks, idle check, snapshots, telemetry). Phases and the `selection.json` choices between them are listed in the script's docstring and in the [plan](../observation/data_store_congest/01_data_store_congest.md).
