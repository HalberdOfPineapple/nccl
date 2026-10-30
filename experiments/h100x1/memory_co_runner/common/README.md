# Memory co-runner harness

This folder holds a memory-heavy GPU role that other experiments can put inside their own fused kernel. It is not a communication path. It makes HBM traffic so that other experiments can test what that traffic does to communication.

The reusable part is one header: [include/hbm_read_pressure.cuh](include/hbm_read_pressure.cuh). The program [src/hbm_read_calibration.cu](src/hbm_read_calibration.cu) runs the role alone on one GPU to measure how much HBM read rate it makes. The measurements and the report are in [calibration/hbm_read_rate](../calibration/hbm_read_rate/README.md).

## How the role works

| Part | What it does |
|---|---|
| Copy thread (thread 0 of the CTA) | Asks the Hopper bulk-copy unit (TMA) to copy one chunk from a 1 GiB global buffer into this CTA's shared memory. It keeps several chunks in flight. When one chunk arrives, it starts the next one. The data is not used. |
| Stop-watch thread (thread 32, a second warp) | Waits for the global stop flag and copies it into shared memory. The copy thread only reads the shared copy. |
| Other threads | Do nothing and wait at the CTA's final barrier. |

Only one thread starts the copies, because the copy unit, not the threads, moves the bytes. So the role needs few SMs and few registers.

Each pressure CTA reads its own part (slice) of the 1 GiB buffer from start to end and then starts again. 1 GiB is about 20 times the 50 MiB L2 cache, so most reads must come from HBM.

## Knobs

| Knob | Config key | Meaning |
|---|---|---|
| Pressure CTAs | `pressure_ctas` | How many CTAs run the role. One CTA per SM. |
| Chunk size | `chunk_kib` | Bytes per bulk copy, in KiB. |
| Chunks in flight | `chunks_in_flight` | How many copies one CTA keeps started at the same time. Bytes in flight per CTA = chunk size × chunks in flight. |
| **Target read rate** | `target_gbps` | Total bytes per second for all pressure CTAs, in GB/s (10^9 bytes/s). 0 means no limit. |
| Threads per CTA | `block_threads` | Physical block size, at least 64. Only two threads work. |
| Window | `window_ns`, `warmup`, `windows` | How long each measured window lasts and how many windows run. |

How the target rate works: chunk number k of a CTA may start only at `start time + k × interval`. The interval is `chunk bytes × pressure CTAs / target rate`. If a chunk is late (because the memory system was slow), it starts at once. So a CTA can catch up, but only as far as its chunks in flight allow. If the achieved rate stays below the target, the memory system could not keep up.

The GPU timer (`%globaltimer`) moves in steps of 32–64 ns on these GPUs (measured in every run). This is much shorter than the time between chunks at the tested rates.

## Using the role inside a fused kernel

1. Give each pressure CTA `chunks_in_flight × chunk bytes` of shared memory (128-byte aligned), plus 8 bytes per chunk for barriers, plus one shared `int` stop copy.
2. Set the shared stop copy to 0, then `__syncthreads()`.
3. Thread 0 calls `hbm_read_pressure::readUntilStopped(...)`. Thread 32 calls `hbm_read_pressure::forwardStopFlag(...)`. All other threads skip both.
4. `__syncthreads()` at the end. The copy thread waits for every started copy before it returns, so the shared memory is free after this barrier.

Shared memory: the thread-store fused harness reserves 116,736 bytes (114 KiB) per CTA today. Any value above half of an SM's shared memory keeps one CTA per SM, and one CTA may use up to 227 KiB. A larger reservation for the pressure role is possible, but it also gives the other CTAs less L1 cache, so their reference must be measured again.

## Two problems found while building it

| Problem | Effect | Fix |
|---|---|---|
| The copy thread read the global stop flag with a system-wide "acquire" load after every chunk | Each such load waits a full memory round trip (about 0.5 µs). The SM could only start about one chunk at a time. | A second warp watches the global flag and writes a shared copy. |
| Small chunks | Each SM finishes about one bulk copy every 0.4–0.6 µs, almost independent of its size, with about two copies overlapping. 16 KiB chunks give only about 43 GB/s per SM. | Use large chunks (64–96 KiB) and at least 2 in flight. |

The second row is measured behaviour, not an explained cause. The calibration report gives the numbers.

## Running

```bash
make -C experiments/h100x1/memory_co_runner/common
python3 experiments/h100x1/memory_co_runner/common/scripts/run_hbm_read_rate.py chunks   # then ctas, then rate
/tmp/thread_store_analysis_env/bin/python experiments/h100x1/memory_co_runner/common/scripts/analyze_hbm_read_rate.py
```

`run_one.py` runs one config under the same guards as the thread-store wrapper: a per-GPU lock shared with the thread-store runs, an idle check, before/after snapshots and telemetry, and refusal to overwrite. Configs use physical GPU indices; do not set `CUDA_VISIBLE_DEVICES`. The program accepts only GPUs 4–7. Do not rebuild while a campaign runs: the runner pins source and binary hashes. The binary is ignored under `bin/`.
