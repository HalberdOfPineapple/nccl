# Common setup for the NVLS root-cause series

**Status (2026-11-18): plan only.** Machine, software, words, role loops and metrics N1–N5, A10, A11, B2, B3, C1, C3, C8, C9 are those of the [observation series setup](../observation/00_common_setup.md). Only additions are listed here.

## GPUs

| Item | Setting |
|---|---|
| Main set | 4 GPUs, physical 4–7 (as the observation series), unless the user allocates others |
| 8-GPU check | Physical 0–7, NCCL kernels through nccl-tests |
| Allocation | **Not given yet** for this series |

## Instrument: Nsight Systems GPU metrics

`ncu` serializes kernel launches, so a kernel that waits for other GPUs cannot be profiled with it in one process without application replay. Nsight Systems samples hardware counters on every GPU at the same time while the kernels run, without serializing them.

| Item | Setting |
|---|---|
| Tool | Nsight Systems 2026.4.1, `nsys profile --gpu-metrics-devices=<list> --gpu-metrics-set=gh100 --gpu-metrics-frequency=10000` |
| Counters | NVLink per direction: `nvl{tx,rx}__bytes_packet_{request,response}_data_{user,protocol}`, 16-byte granularity; L2 sector throughput by source (`srcnode_hub`, `srcnode_gpc`, `srcunit_tex`) |
| Custom set (to check in E1.0) | A `file:` metric set adding `nvl{tx,rx}__cycles_active` and `nvl{tx,rx}__cycles_elapsed`, if Nsight Systems accepts them |
| Window | Samples inside the timed windows of `mm_mix` only (window start and end from its JSON), summed per GPU and direction |
| Permission | Profiling counters; the session runs as root |

## Metrics added in this series

IDs are new; definitions of N1, B3 and C9 are in the observation setup.

| ID | What it means | How it is obtained | Limits |
|---|---|---|---|
| R1 | Request payload bytes per payload byte, per GPU and direction | `nvl{tx,rx}__bytes_packet_request_data_user` ÷ the role's payload bytes in the window | Payload byte = the bytes the role completed (N1 × window) |
| R2 | Request protocol bytes per payload byte | `..._request_data_protocol` ÷ payload bytes | Protocol = headers and control in request packets |
| R3 | Response payload bytes per payload byte | `..._response_data_user` ÷ payload bytes | |
| R4 | Response protocol bytes per payload byte | `..._response_data_protocol` ÷ payload bytes | |
| R5 | Link direction busy share | `nvl{tx,rx}__cycles_active ÷ cycles_elapsed`, if the custom set works | **Not available (2026-11-19):** Nsight Systems 2026.4.1 accepted the set but recorded only the packet counters |
| R6 | Operation latency | Probe warp: `%globaltimer` before and after a batch of one outstanding operation, median and 90th percentile per window | One warp on one SM per GPU; its own traffic is negligible but not zero |
| R7 | Bytes in flight per GPU | N1 × R6 for the role, by Little's law | Holds for a steady state only |

Packet sizes: the counters count in 16-byte units, so the protocol bytes of one packet are a multiple of 16. R2 ÷ (accesses per payload byte) gives protocol bytes per access.

## Instrument: driver link counters, long minus short (added 2026-11-18)

The driver keeps two byte counters per link and direction. Reading them is read-only and needs no profiler.

| Counter | Command | Counts |
|---|---|---|
| Data | `nvidia-smi nvlink -i <list> -gt d` | Payload only |
| Raw | `nvidia-smi nvlink -i <list> -gt r` | Everything on the wire: payload, packet headers, read requests, write responses |

The counters run over the whole process, so they include start-up, set-up and checking traffic. To remove it, every setting runs twice in the same round: a **long** run (1 warm-up and 10 kept windows of 300 ms) and a **short** run (1 warm-up and 2 kept windows). Long minus short leaves the traffic of 8 windows. The payload of those 8 windows is the long run's role bytes minus the short run's, counted over all windows including warm-up, because warm-up windows also move data. Same method as AllReduce plan 04 of the [scalability series](../../../h100-scale/allreduce/scalability/PLANS.md).

| ID | What it means | How it is obtained | Limits |
|---|---|---|---|
| R8 | Data link bytes per payload byte, per GPU and direction | (data counter, long − short) ÷ (payload bytes, long − short) | Must match the byte model (per payload byte: `multimem.st` 1 into each GPU and 1/N out; `ld_reduce` the reverse; AllReduce 1 + 1/N each way); a mismatch of more than 1% means the run or the method is wrong |
| R9 | Raw link bytes per payload byte, per GPU and direction | (raw counter, long − short) ÷ (payload bytes, long − short) | Includes every protocol byte the driver counts; what the raw counter leaves out (framing, idle symbols) is not known |
| R10 | Raw link rate per direction | (raw counter, long − short) ÷ (window time, long − short) | Nominal window time (8 × 300 ms); gaps between windows are not counted in the time |

Raw ÷ data for a direction is R9 ÷ R8, and does not depend on how payload is counted.

**Payload byte in R8 and R9** is a byte of the collective's buffer, as in the nccl-tests tables of the 8-GPU raw-counter check, so that the two compare directly. The harness counts role bytes: bytes of one GPU's own part (metric N1). A buffer holds the parts of all N GPUs, so payload bytes = N × the mean over GPUs of one GPU's role bytes. All GPUs issue the same role, and their rates differ by less than 1% in the observation series. For an arm with two roles on separate CTAs, payload bytes = N × the mean of the two roles' bytes, and the byte model adds each role's share; for this arm only raw ÷ data and R10 are compared with the other arms. The plain arms use the same conversion; only their raw ÷ data and R10 are used.

## Profiler rows that lost link counters (added 2026-11-19)

In some profiled runs one GPU's profiler data bytes (R1 + R3) are 17/18, 16/18, 15/18 or 14/18 of the driver data counter, with no samples missing in time: the profiler lost the counters of 1–4 of the GPU's 18 NVLinks. A GPU-direction row whose profiler data bytes differ from the driver data counter by more than 0.5% is kept in the per-run CSV and left out of every median; summaries count such rows (`excluded_gpu_rows`). In plan 01 this removed 46 of 976 rows.
