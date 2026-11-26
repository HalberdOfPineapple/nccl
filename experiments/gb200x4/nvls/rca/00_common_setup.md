# Common setup for the GB200 NVLS root-cause series

**Status (2026-11-26): plan only.** Words, role loops and metric definitions not listed here are those of the H100 series: metrics N1–N5, A10, A11, B3 in the [H100 observation setup](../../../h100x4/nvls/observation/00_common_setup.md); R1–R10 in the [H100 root-cause setup](../../../h100x4/nvls/rca/00_common_setup.md); S1 and S6 in the [AllReduce scalability setup](../../../h100-scale/allreduce/scalability/00_common_setup.md). Only GB200 values and new metrics are listed here.

## Machine

Facts read on node `b3-11-s1-dgx-03-c01` on 2026-11-26 (read again at the start of every session; nodes differ).

| Item | Value |
|---|---|
| Cluster, partition | Empire AI Beta, `beta_test` (4 nodes: `b2-11`, `b2-12` in rack 2; `b3-11`, `b3-12` in rack 3) |
| Node | 1 GB200 compute tray: 2 Grace CPUs (144 cores, aarch64, 64 KiB pages), 4 GB200 GPUs |
| GPU | NVIDIA GB200, 152 SMs, 184 GiB HBM, compute capability 10.0, max SM clock 2062 MHz, power limit 1200 W |
| NVLink | 18 links per GPU × 53.125 GB/s per direction = 956 GB/s per direction per GPU (nominal); all 4 GPUs `NV18` to each other through the NVL72 switch trays |
| Fabric | `nvidia-smi -q`: fabric state Completed, Success; IMEX channel `/dev/nvidia-caps-imex-channels/channel0` present; NCCL reports "NVLS multicast support is available ..., symmetric multimem enabled" |
| Driver | 580.167.08 (CUDA 13.0 driver API) |
| Addressing | ATS (GPU can address Grace memory); not used in this series |
| Clocks | Not locked (no permission). SM clock idles at 120 MHz and boosts under load; recorded per run |

## Software

| Item | Value |
|---|---|
| Container | `nvcr.io/nvidia/pytorch:25.06-py3`, named `dev` (`pyxis_dev`): CUDA 12.9.86, Nsight Systems 2025.3.1, gcc 13.3 |
| How to enter it from a VS Code / SSH shell on the node | `.vscode/in_container.sh` (uses `enroot exec`, resets PATH so the container's tools are used) |
| NCCL | This repository, 2.32.3, built in the container with `make -j 128 src.build BUILDDIR=<repo>/build-gb200 NVCC_GENCODE=-gencode=arch=compute_100,code=sm_100` |
| nccl-tests | 2.22.0 (commit `826b3e6`), `/mnt/home/wli10/nccl-tests/build-gb200/`, same `NVCC_GENCODE`, `NCCL_HOME=<repo>/build-gb200` |
| Process model | One process, 4 GPUs (`-t 1 -g 4`), no MPI |

## GPUs and sessions

| Item | Setting |
|---|---|
| GPUs | Physical 0–3 of the session's node (all GPUs of the node; whole-node allocation) |
| Session | One `beta_test` job: 1 node, `--gres=gpu:4 -c 144 -t 02:00:00`, `--container-name=dev`; started by the user from `tmux` on the login node |
| Usable GPU time | About 90 minutes per session; the rest is reserved for start-up, build checks and copying results to shared storage |
| Results | Written straight to the shared home directory (`results/<date>/<session>/`), never only to node-local `/tmp` |
| Other tenants | None on the node's GPUs (whole node). The NVL72 switch trays are shared with every job in the rack; not observable |

## Run protocol

1. At session start: record the session context (G1). Check `nvidia-smi --query-compute-apps` is empty on all 4 GPUs.
2. Build checks only if the source changed (NCCL build takes about 2 minutes on 128 cores).
3. One sequential runner per session. Arms rotate inside a round; long and short runs of one arm are back to back.
4. Each run writes its config, NCCL log (`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,TUNING,NVLS`, `NCCL_DEBUG_FILE`), stdout, and `nvidia-smi nvlink -gt d` and `-gt r` snapshots before and after (B3, and the raw counter of R9/R10).
5. If the session ends early, completed rounds are kept; incomplete rounds are kept and marked, and are not used in medians.

## Instruments

| Instrument | Use | Permission | Status on GB200 |
|---|---|---|---|
| nccl-tests timing | algBW (S1) per call; busBW (S6) by formula | None | Works (pilot) |
| `mm_mix` harness timing | Role byte rate N1 per GPU, fixed time windows | None | Needs the port in plan 01 E1.5 |
| Driver link counters `nvidia-smi nvlink -gt d` and `-gt r` | Data and raw bytes per link and direction (KiB units, 18 links) | None | Readable (2026-11-26); semantics on NVLink5 checked in plan 01 E1.1 |
| Nsight Systems GPU metrics | Packet-type bytes R1–R4 | Admin (`RmProfilingAdminOnly=0`, or run by an admin) | **Not available**: "insufficient privilege" on all 4 GPUs; metric set for GB200 to be confirmed by an admin run |
| `ncu` | Not used (serializes multi-GPU kernels; also needs admin) | Admin | Not installed in the container |

## Metrics added in this series

| ID | What it means | How it is obtained | Limits |
|---|---|---|---|
| G1 | Session context | Hostname, rack, GPU UUIDs, `nvidia-smi -q` fabric state and clique ID, driver, clocks, temperature and power before and after each run, compute-apps list | Taken from tools, not controlled |
| G2 | Ceiling share | Raw link rate (R10) on a busy direction ÷ 956 GB/s nominal | Nominal = 18 × 53.125 GB/s from `nvidia-smi nvlink -s`; the usable wire rate after encoding is not known |
| G3 | Mixing ratio | NVLS AllReduce busBW ÷ mean of NVLS AllGather busBW and NVLS ReduceScatter busBW, same session, same payload, same CTA count per kernel | busBW by the per-direction factors below, not the nccl-tests column |
| G4 | Gain over Ring | NVLS AllReduce algBW ÷ Ring AllReduce algBW, same session, same payload | Each at its own calibrated CTA count, and also at NCCL defaults |
| G5 | Raw cost of data | Raw ÷ data on one direction, = R9 ÷ R8 | Same as in the H100 series; named here because plan 01 uses it as a gate |

busBW per direction for each collective (data bytes per payload byte on the busier direction; N = 4): NVLS AllGather 1 (nccl-tests prints algBW for the full output buffer); NVLS ReduceScatter 1 (algBW for the full input buffer); NVLS AllReduce 1 + 1/N = 1.25; Ring AllReduce and plain-load/store symmetric AllReduce 2(N − 1)/N = 1.5; Ring AllGather and ReduceScatter (N − 1)/N = 0.75. These are the factors of the H100 report; plan 01 E1.4 checks them against the data counter (R8) on GB200.

## Long-minus-short counter method on nccl-tests

The driver counters cover the whole process (set-up, warm-up, checks). Every counter measurement is a pair of runs in one session: a **long** run (`-n 200 -w 20`) and a **short** run (`-n 40 -w 20`) at the same size and settings, back to back. Long minus short leaves 160 timed calls. Payload bytes = 160 × buffer size. Time = 160 × the mean time per call of the long run. Checking is off in both (`-c 0`) so that no checking traffic differs between them.
