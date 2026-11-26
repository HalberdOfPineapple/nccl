# Common setup for the AllReduce scalability series

**Status (2026-11-16): all plans ran.** The selection check (plan 01) confirmed every setting marked "to check": each CTA knob gave exactly the requested CTA count.

## Words used in this series

| Word | Meaning |
|---|---|
| Arm | One AllReduce implementation under comparison: Ring, NVLS, Symmetric multicast, Symmetric plain load/store |
| GPU count (N) | Number of GPUs in one AllReduce call: 2, 4, 6 or 8. One NCCL rank per GPU |
| Payload (S) | Bytes of the buffer that each GPU contributes to one call. Sizes use binary units (1 GiB = 2^30 bytes) |
| CTA | One thread block of the NCCL kernel. NCCL also calls it a channel |
| Cell | One arm at one GPU count. There are 15 cells: 4 arms × 4 GPU counts, less Symmetric multicast at 2 GPUs |
| Rate | Payload ÷ mean time of one call, in decimal GB/s (metric S1). nccl-tests prints it as `algbw` |
| Multimem | PTX instructions `multimem.ld_reduce` and `multimem.st`: one load returns the sum over all GPUs, one store reaches all GPUs |

## Machine and software

| Property | Setting |
|---|---|
| Machine | 8 × H100 80 GB, all on NVLink. GPUs 0–3 on CPU socket 0, GPUs 4–7 on socket 1 (from the machine guide, `~/workspace/MACHINE-GUIDE.md`) |
| Link | 18 NVLink links per GPU, about 478 GB/s per direction, as recorded in the [4-GPU setup](../../../h100x4/nvls/observation/00_common_setup.md). Check again with `nvidia-smi nvlink -s` on all 8 GPUs |
| GPU allocation | **Not given yet.** The machine is shared. 8-GPU runs need every GPU free |
| Clocks and power | Unchanged. No clock locking |
| NCCL | In-tree `build/lib/libnccl.so.2.32.3`, through `LD_LIBRARY_PATH` |
| nccl-tests | `/root/nccl-tests/build/all_reduce_perf`. Record the NCCL version it prints |

## GPU sets

| GPU count | Main set | Control sets (plan 02) |
|---|---|---|
| 2 | 4, 5 | 0, 1 and 3, 4 (one GPU per socket) |
| 4 | 4, 5, 6, 7 | 0, 1, 2, 3 and 2, 3, 4, 5 (two GPUs per socket) |
| 6 | 1, 2, 3, 4, 5, 6 | 0–5 (4 + 2 per socket) |
| 8 | 0–7 | None |

The main sets for 2 and 4 GPUs match the earlier allocation, so the 4-GPU numbers can be compared with the [4-GPU report](../../../h100x4/nvls/observation/multimem_load_store_mix/README.md). The final sets depend on the allocation.

## How each arm is selected

| Arm | Environment | Buffer registration (`-R`) | CTA count knob (to check) | Proof in the NCCL log |
|---|---|---|---|---|
| Ring | `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple` | 0 | `NCCL_MIN_CTAS` = `NCCL_MAX_CTAS` | Algorithm Ring, protocol Simple |
| NVLS | `NCCL_ALGO=NVLS` | 0 | `NCCL_NVLS_NCHANNELS` | Algorithm NVLS |
| Symmetric multicast | `NCCL_SYM_KERNEL=allreduce:AllReduce_RSxLDMC_AGxSTMC` | 2 (symmetric) | `NCCL_SYM_CTAS` | `-> Kernel AllReduce_RSxLDMC_AGxSTMC`, no other kernel |
| Symmetric plain load/store | `NCCL_SYM_KERNEL=allreduce:AllReduce_RSxLD_AGxST` | 2 (symmetric) | `NCCL_SYM_CTAS` | `-> Kernel AllReduce_RSxLD_AGxST`, no other kernel |
| NVLS, registered (plan 05, added 2026-11-17) | `NCCL_ALGO=NVLS`, `NCCL_PROTO=Simple`, `NCCL_SINGLE_PROC_MEM_REG_ENABLE=1` set before communicator init | 1 (`ncclCommRegister`) | `NCCL_NVLS_NCHANNELS` | Algorithm NVLS, and `NVLS UB registered` on every rank with no `fell back for` |
| Example kernels of `contrib/custom_algos/allreduce` (plan 06, added 2026-11-17) | Own MPI program, not nccl-tests; see the plan | Symmetric windows | `-c`/`-C`, powers of 2, 1024 threads/CTA | Built-in correctness PASS; NCCL column: `-> Kernel AllReduce_RSxLDMC_AGxSTMC` |

Notes:

- Ring is fixed to the Simple protocol because that is the Ring protocol for large payloads; the LL and LL128 protocols are for small ones.
- The two symmetric arms use registered buffers and the other two do not. This difference is part of the arms as NCCL offers them. Plan 01 records whether Ring and NVLS change with `-R 1`. **Note (2026-11-17):** plan 01 did not run `-R 1`. For NVLS this comparison is now plan 05, which also needs `NCCL_SINGLE_PROC_MEM_REG_ENABLE=1`; Ring with `-R 1` is not planned.
- `NCCL_SYM_KERNEL` forcing was confirmed at 4 GPUs for the multicast kernel only. `NCCL_ALGO` does not accept kernel names.

## Base command

```
all_reduce_perf -t 1 -g <N> -b <S> -e <S> -f 2 -n 20 -w 5 -c 1 -R <0|2> -d float -o sum
```

with `CUDA_VISIBLE_DEVICES=<GPU set>`, `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=TUNING`, and one `NCCL_DEBUG_FILE` per run. This is the command of the 4-GPU cross-check with `-g` and the size changed.

| Setting | Value | Why |
|---|---|---|
| Process layout | One process, one host thread, N GPUs (`-t 1 -g N`) | Same as the 4-GPU cross-check. With large payloads the host launch time is small against the call time; plan 01 checks this at 8 GPUs with `-t N -g 1` |
| Data type and operation | `float`, sum | Same as the 4-GPU work |
| Calls per run | 5 warm-up, 20 timed | Same as the 4-GPU work. Plan 04 uses more timed calls |
| Check | 1 check pass (`-c 1`) | A fast but wrong run must not count |
| In-place and out-of-place | Both are printed; the series reports out-of-place and keeps in-place in the summary | Same as the 4-GPU work |

## Metric registry

IDs A11, B2 and B3 keep the meaning they have in the [thread-store registry](../../../h100x2/thread_store/observation/00_common_setup.md#metric-registry) and the [NVLS registry](../../../h100x4/nvls/observation/00_common_setup.md#metric-registry). `S` IDs are new here.

| ID | What it means | How it is obtained | Limits |
|---|---|---|---|
| S1 | Rate: payload bytes per GPU ÷ mean time of one call, GB/s | `algbw` column of nccl-tests, out-of-place; median over rounds | Mean over 20 calls on rank 0's timer. Not a link rate |
| S6 | busBW with the formula of the arm: algBW × 2(N − 1)/N for Ring and Symmetric plain load/store; algBW × (1 + 1/N) for NVLS and Symmetric multicast | Computed from S1 (algBW). See "busBW formulas" below | Not a link rate. Not the `busbw` column of nccl-tests, which uses the Ring factor for every arm |
| S2 | Scaling ratio: S1 at N GPUs ÷ S1 of the same arm at its smallest GPU count | Computed from S1 at matched payload and CTA setting | Symmetric multicast starts at 4 GPUs, so its ratios are not comparable with ratios that start at 2 |
| S3 | What NCCL ran: algorithm or kernel, protocol, CTAs × threads/CTA | NCCL log lines of each run | A run with any other kernel line is rejected |
| S4 | Link bytes per payload byte, per GPU and direction | B3 difference ÷ (payload × calls) | Whole run, warm-up included. Used in plan 04 |
| S5 | Share of link rating in use: S1 × S4 ÷ 478 GB/s | Computed | Uses the rated link speed, not a measured maximum |
| A11 | Correctness | `#wrong` column, both passes | |
| B2 | Device operating state | Clocks, power, temperature, other processes, before and after each run | Snapshots, not continuous |
| B3 | NVLink byte counters per GPU and direction | `nvidia-smi nvlink -gt d` before and after | No resets. Includes all traffic in the interval |

### busBW formulas

**Added 2026-11-16, given by the user.** busBW turns algBW into the bytes one GPU must move on its link per second. The factor depends on the algorithm, so one factor for all arms would be wrong.

| Arm | busBW | N = 2 | N = 4 | N = 6 | N = 8 |
|---|---|---|---|---|---|
| Ring | algBW × 2(N − 1)/N | 1.000 | 1.500 | 1.667 | 1.750 |
| Symmetric plain load/store | algBW × 2(N − 1)/N: each GPU loads (N − 1)/N × S from peers and receives (N − 1)/N × S of peer stores | 1.000 | 1.500 | 1.667 | 1.750 |
| NVLS | algBW × (1 + 1/N) | 1.500 | 1.250 | 1.167 | 1.125 |
| NVLS, registered (plan 05) | algBW × (1 + 1/N) | 1.500 | 1.250 | 1.167 | 1.125 |
| Symmetric multicast | algBW × (1 + 1/N) | not run | 1.250 | 1.167 | 1.125 |
| One-shot example kernel (plan 06) | algBW × N for bytes sent (each GPU answers every GPU's loads); measured 4.000 and 8.000 at 4 and 8 GPUs (plan 06), received 1 | 2 | 4 | 6 | 8 |
| Two-shot example kernels (plan 06) | algBW × (1 + 1/N) | 1.500 | 1.250 | 1.167 | 1.125 |

The nccl-tests `busbw` column always uses 2(N − 1)/N. It is kept in the raw summary as `out_of_place_busbw_nccltests_gbps` and is not reported for the multimem arms.

Not planned in this series: profiler counters (`ncu`, `nsys`), per-CTA progress, and timing inside the kernel. These need a kernel harness, which this series does not build.

## Run protocol

- Before each run: hold the lock of every GPU in the set, check that those GPUs are idle, and stop if one is occupied. Never touch another user's process. Use the lock files and idle guard of [`run_session.py`](../../../h100x4/nvls/common/scripts/run_session.py).
- One run is one `all_reduce_perf` invocation with one config. The runner writes the NCCL log, the program output and the before/after snapshots next to each other.
- One round runs every arm once at one GPU count. The order of arms rotates between rounds. At least 5 rounds per cell for counted results, 3 for calibration.
- GPU counts are run in blocks (all rounds at 2 GPUs, then 4, and so on) and the whole sequence is repeated once in reverse order, so slow drift of the machine does not look like a GPU-count effect.
- The unit of comparison is one run. The 20 calls inside a run are not independent repetitions.
- Report medians, the range over rounds, and ratios between arms per round. There is no healthy and perturbed pair in this series; where a ratio needs a reference, name it in the table.
- Rejected runs (wrong kernel, wrong values, occupied GPU) are kept locally with the reason and run again.
- Raw logs and snapshots stay out of git. Configs, scripts, summaries, figures and reports go in git.
