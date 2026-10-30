# 01 · HBM read rate of the memory co-runner

This is a calibration experiment on one GPU. It does not test communication. It measures a memory-heavy role that other experiments will place in their fused kernels, such as the source-memory experiment in [04_shared_resource_contention.md](../../../../h100x2/thread_store/observation/shared_resources/04_shared_resource_contention.md).

**Question.** Can one kernel role read HBM at a rate we choose, from near zero to near the HBM peak, while using few SMs and fitting the fused-kernel design?

## Why

In a fused kernel, communication CTAs and other CTAs share one launch and sit on different SMs. Earlier source-memory tests made memory traffic with plain loads. To get near the HBM peak, those loads needed about 128 SMs × 1024 threads. That leaves no SMs for communication, so strong memory traffic and a strong communication reference could not run together.

Earlier tests also set the traffic level by changing CTA count and threads per CTA. That changes two things at once: how many SMs the role holds, and how much traffic it makes.

This experiment builds a role that changes only one thing: a **target read rate**. CTA count and bytes in flight stay fixed while the target changes.

## Design

| Part | Choice | Reason |
|---|---|---|
| How bytes are read | Hopper bulk copy (TMA): `cp.async.bulk` global → shared memory, completion by `mbarrier` | One thread can keep many bytes in flight; the copy unit moves them, so few threads and registers are needed |
| Threads that work | Thread 0 starts copies; thread 32 watches the stop flag | Fits the idle lanes of a fused co-runner CTA |
| Rate knob | Chunk k of a CTA starts no earlier than `start + k × interval` | A direct, single knob; achieved rate below the target shows the memory system could not keep up |
| Buffer | 1 GiB, split into one slice per CTA, read in order and repeated | About 20 times the 50 MiB L2, so most reads come from HBM |
| Correctness | The buffer holds a known byte pattern; the copy thread reads the first word of each chunk and the harness checks the sum | Shows that the bytes arrived |
| Placement | One CTA per SM (shared-memory reservation, occupancy check, recorded SM IDs) and a control CTA that times the window | Same launch shape as the thread-store fused harness |

## Configuration

| Knob | Role | Values |
|---|---|---|
| GPU | Constant | Physical GPU 6 (H100 80GB HBM3, 132 SMs) |
| HBM peak, the 100% line | Constant, from device | 2 × 2619 MHz × 5120 bits / 8 = 3352 GB/s |
| Threads per CTA | Constant | 1024 (only 2 work) |
| Window | Constant | 2 warm-up windows, then 5 windows of 20 ms per run |
| Repeats | Constant | 5 runs per setting, in rotated order |
| **Stage 1: bytes in flight per CTA** | **Independent variable** | **chunk size 16, 32, 64, 96, 128, 192 KiB × chunks in flight 1, 2, 3, 4, 6**, where the total is ≤ 224 KiB; 8 pressure CTAs; no rate limit |
| **Stage 2: number of pressure CTAs** | **Independent variable** | **1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128**, for two per-CTA settings chosen from stage 1; no rate limit |
| **Stage 3: target read rate** | **Independent variable (the knob)** | **5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100% of peak**, plus no limit; CTA count and bytes in flight fixed per setting |

## Selection rules (fixed before the results)

- Stage 1 → stage 2: (a) the per-CTA setting with the highest rate; (b) the highest-rate setting that fits the shared memory the thread-store fused harness reserves today (114 KiB per CTA). If two settings are within 2% of each other, take the one with fewer bytes in flight.
- Stage 2 → stage 3: for each per-CTA setting, the smallest CTA count whose rate is at least 95% of that setting's highest measured rate in stage 2.
- A rate that still rises at the largest tested CTA count is reported as "still rising", not as a maximum.

## Measured responses

| Response | How it is measured |
|---|---|
| Achieved read rate | Chunks read by all pressure CTAs × chunk size, divided by the time from the first CTA start to the last CTA end, per window. Shown as % of the 3352 GB/s peak. |
| Balance between CTAs | Rate of the slowest pressure CTA divided by the fastest, per window |
| Timer step | Smallest and largest step of `%globaltimer`, measured by the control CTA |
| Operating state | Sampled memory clock, SM clock, power and temperature during each run |

The read rate counts bytes the program asked for. It is not counted HBM traffic. Hardware memory counters were not validated in earlier sessions; a profiler check is optional and separate.

## Privileged access

None beyond the existing GPU allocation (physical GPUs 4–7; this experiment uses GPU 6). No clock or power changes.
