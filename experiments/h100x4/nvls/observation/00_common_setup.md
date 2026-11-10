# Common setup for the NVLS observation series

This file defines the setup and metrics. Exact results belong in each experiment's README.

## Words used in this series

| Word | Meaning |
|---|---|
| CTA | One thread block. The report calls it a "block". We run one CTA per SM |
| Multimem pointer | One address that maps to the same buffer on every GPU of the group. It comes from `ncclGetLsaMultimemPointer`, with `lsaMultimem = true` in the device communicator requirements |
| `st_multicast` | A store through a multimem pointer: PTX `multimem.st.global`. One store reaches the buffer on every GPU |
| `ld_reduce` | A load through a multimem pointer: PTX `multimem.ld_reduce.global.add`. One load returns the sum of the value on every GPU |
| Plain load, plain store (`ld`, `st`) | A load or store through the `ncclGetLsaPointer` address of one peer GPU |
| Role | The work that one CTA does in a launch: `multicast`, `reduce`, `allreduce`, `plain_load`, `plain_store`, or `idle` |
| Role byte rate | Bytes one role completed per second on one GPU (metric N1) |
| nccl-tests busBW | The number nccl-tests prints, from the formulas below (metric N2). It is computed from bytes and time. It is not a measured link rate |
| Link-model busBW | The data one GPU must move in its busier link direction, computed from the role rates (metric N5). For one collective on n GPUs: algBW × 1 for AllGather or ReduceScatter, algBW × (1 + 1/n) for AllReduce, where algBW is buffer bytes ÷ time |

## Machine and software

| Property | Setting |
|---|---|
| GPU count | 4 of the 8 H100 GPUs on this machine |
| GPU allocation | Physical GPUs 4–7, given by the user on 2026-11-01. All on socket 1. GPUs 0–3 belong to other work; do not use them. GPU 6 has held about 122 MiB at 0% with no known owner since 2026-11-01 21:43 UTC; the user accepted it as not occupying, and the guard limit for GPU 6 is 150 MiB |
| Link | 18 NVLink links per GPU at 26.562 GB/s each, about 478 GB/s per direction (`nvidia-smi nvlink -s`) |
| Link state | Pass-through virtualization; fabric state "Completed"; clique ID 0 on all 8 GPUs; no `/dev/nvidia-nvswitch*` inside this VM |
| Multimem support | Checked by E1.0 on GPUs 4–7; see the [report](multimem_load_store_mix/README.md). Check again if the allocation or driver changes |
| Clocks and power | Unchanged. No clock locking. Profilers use `--clock-control none` |
| NCCL | In-tree `build/lib/libnccl.so.2.32.3` |
| nccl-tests | `/root/nccl-tests/build`, version 2.21.1. It links the system `libnccl.so.2` by default, so set `LD_LIBRARY_PATH` to the in-tree build and record the version it prints |

## Harness

**Note (2026-11-01).** Built as [`mm_mix.cu`](../common/src/mm_mix.cu) and [`run_session.py`](../common/scripts/run_session.py). The table below is the design we wrote before building. These points differ: each run launches 132 CTAs, one per SM, and CTAs without a role exit at once instead of polling; each rank's window has one controller CTA; rates use each role's own first start and last end on that GPU; output is checked only for the words written in that window; and each CTA reserves over half of the SM's shared memory, which NCCL's kernels do not do. Plan 01 ran only N1, N2, N3, N4 within one GPU, A10, A11, B2 and B3; C1, C3, C8 and C9 did not run. N4 cannot compare roles on different GPUs from timestamps, because each GPU clock has its own origin, seconds apart.

The harness is `experiments/h100x4/nvls/common/`, built with `make`. It follows the structure of [`store_congest.cu`](../../../h100x2/thread_store/common/src/store_congest.cu): one process, one host thread per rank, each with its own GPU, NCCL rank, stream and window, and a table that gives each SM its role.

| Design point | Plan | Why |
|---|---|---|
| Launch | Every run launches the **same CTA count, thread count and shared-memory reservation**. Roles not in use are `idle`: one thread polls a stop flag slowly and the rest exit | A one-role run must not gain L1 cache or SMs compared with a two-role run |
| Role placement | Roles are spread evenly over the GPCs, using the SM-to-GPC map method from plan 07 (A12) | One GPC carries only about 700–800 GB/s of stores on H100, so packing roles could create a limit that is not under test |
| Measured window | All roles start after a barrier across all four GPUs and run for a fixed time (default 300 ms) in a loop over their buffer. Each CTA records bytes and time at regular checkpoints. Rates use only the time when all roles were running, per GPU | With a fixed byte count, the faster role finishes first and the slower one then runs alone. That hides the interference this series is testing |
| Posted stores | Stores are counted when issued. All writers fence before the end time. At 300 ms, bytes in flight are a very small part of the total | A timer around posted stores is not remote-store latency |
| Data | `float` sum in 16-byte packs (4 × `f32`) for every role, matching NCCL's 16-byte `ncclSymkBytePerPack` | One width across arms, so the width does not change between comparisons |
| Buffers | Each GPU holds an 8 GiB input and an 8 GiB output region. Each rank uses its own 2 GiB part, so no two ranks store to the same addresses | Matches the "8GB, 4 GPUs" setting in the report's section 2 |
| Correctness | `reduce` compares returned sums against the known pattern. After the window, each GPU checks the last complete pass of the output region. A mismatch stops the run | A fast but wrong run must not be counted |

Role loops, taken from the NCCL symmetric kernels so the work has the same shape:

| Role | One loop step | Modeled on |
|---|---|---|
| `multicast` | Load from local input, then `st_multicast` to the rank's part of output | `bcastMultimem` in [primitives.cuh](../../../../src/device/symmetric/primitives.cuh) |
| `reduce` | `ld_reduce` from the rank's part of input, then store to local output | `reduceMultimem` in [reduce_scatter.cuh](../../../../src/device/symmetric/reduce_scatter.cuh) |
| `allreduce` | `ld_reduce` from input, then `st_multicast` of that same value to output, in the same thread | `allreduceMultimem` in [all_reduce.cuh](../../../../src/device/symmetric/all_reduce.cuh) |
| `plain_load` | Load from peer, then store to local output | Plain-load equivalent of `reduce` |
| `plain_store` | Load from local input, then store to peer | Plain-store equivalent of `multicast` |

## Metric registry

IDs with a letter and number from the [thread-store registry](../../../h100x2/thread_store/observation/00_common_setup.md#metric-registry) keep the same meaning here. `N` IDs are new in this series.

| ID | What it means | How it is obtained | Limits |
|---|---|---|---|
| N1 | Role byte rate per GPU | Bytes the role's CTAs finished, divided by the time all roles were running on that GPU, decimal GB/s | Requested bytes, not link bytes. Issue time for stores |
| N2 | busBW with the nccl-tests formula | See formulas below. For AllReduce, one role reports both operations. For two separate roles, report both readings of the combined number | Not a link rate. The factor differs by collective |
| N3 | CTA progress spread | Per-CTA bytes and checkpoint times | Shows whether a few CTAs starve. It is not a queue measurement |
| N5 | Link-model busBW | From N1 rates with the link model below. Defined only when all 4 GPUs issue every role. Added 2026-11-07 | A model, not a measurement. It counts data only, not request packets. It assumes the issuing GPU's own copy crosses its own link, which B3 confirms on this machine only |
| N4 | Role overlap | Share of each role's active time when the other role was also running, per GPU | Overlap of active time is not continuous request activity |
| A10 | Role and SM placement | Role, SM ID and CTA index per rank | Spatial separation does not mean GPU-wide resources are separate |
| A11 | Correctness | Mismatch count per rank | |
| B2 | Device operating state | Snapshot and sampled clocks, power, temperature, clock-event reasons, per GPU | Whole-run samples |
| B3 | Direction totals of NVLink traffic | `nvidia-smi nvlink -gt d` before and after, read-only | Includes setup and checking traffic. No counter resets. Record errors if not supported |
| C1 | HBM bytes per GPU | DRAM byte families through `ncu`, separate pass | Whole-GPU, not per role |
| C3 | L2 demand per GPU | L2 sector families through `ncu`, separate pass | Whole-GPU, not per role |
| C8 | Profiling validity | Tool version, metric list, overlap and correctness checks, errors | |
| C9 | NVLink bytes per direction | `nvltx__bytes_packet_*` and `nvlrx__bytes_packet_*` through `ncu --replay-mode application`, per GPU. This worked on two GPUs in [plan 08](../../../h100x2/thread_store/observation/peer_store_l2_path/README.md); it is not yet validated on multimem traffic | GPU-wide, not per role. Meaning of protocol bytes is not known |

### busBW formulas

With `n` ranks and `t` the elapsed seconds. These come from nccl-tests.

| Collective | busBW | At n = 4 |
|---|---|---|
| AllGather | `(count × size × n / t) × (n − 1) / n` | 0.75 × total gathered bytes / t |
| ReduceScatter | `(count × size × n / t) × (n − 1) / n` | 0.75 × total input bytes / t |
| AllReduce | `(count × size / t) × 2(n − 1) / n` | 1.5 × bytes per rank / t |

For the role loops, N2 is computed from the bytes all ranks completed in the window, so a fixed-time run gives the same formula as a fixed-byte run.

### Link-model busBW (N5)

**Added 2026-11-07.** The author of the scale-up switch report confirmed that its busBW values are not nccl-tests busBW. nccl-tests uses the same factor for a collective whatever algorithm runs, and its factors are the per-GPU bytes of the ring algorithm. They are not the bytes that multimem traffic puts on the links. N5 counts those bytes instead.

With x the N1 rate of a role (bytes of its own part per GPU per second), and all n = 4 GPUs issuing, each GPU moves:

| Role | Into each GPU | Out of each GPU | N5 for this role alone | Same as, for one collective |
|---|---|---|---|---|
| `st_multicast` | 4x: one copy from every issuer, its own copy included | x | 4x | AllGather: algBW × 1 |
| `ld_reduce` | x: its own sums | 4x: its part, read by every issuer, itself included | 4x | ReduceScatter: algBW × 1 |
| AllReduce (same CTAs) | 5x | 5x | 5x | AllReduce: algBW × (1 + 1/n) = 1.25 × algBW |
| Plain load or plain store, ring | x | x | x | One point-to-point transfer |

When several roles run, N5 is the larger of the summed "into" and summed "out of" values. The issuing GPU counts as one of the n receivers of its own data. On this machine the B3 link counters confirm this within about 1–2%; on another platform it must be checked again.

## Run protocol

- Before each run, hold locks on the four GPUs, check that they are idle, and stop if any is occupied. Never touch another user's process.
- Wrap each run with snapshots and telemetry, as `run_one.py` does for the thread-store harness.
- The unit of comparison is one run's median over its windows. Windows inside a run are not independent repetitions.
- Default: one warm-up window and five kept windows per run, with at least five rounds per arm. Rotate the arm order between rounds.
- Report the absolute values, the paired ratio per round, and `(median_P − median_H) / IQR_H`. If the reference IQR is zero, report that and use absolute differences.
- Profiler passes are separate runs after the timing runs. Check that the interaction is still present in them with N1 and N4.
- Keep raw records out of git. Keep configs, scripts, summaries, figures and reports in git.
