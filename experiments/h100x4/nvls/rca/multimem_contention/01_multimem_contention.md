# Plan 01: packets, granularity, arbitration and latency of mixed multimem traffic

**Status (2026-11-19): done. E1.1–E1.8 on GPUs 4–7, E1.9 on GPUs 0–7. Results: [README.md](README.md); E1.9 configs and results in [`h100x8/nvls/rca/multimem_contention/`](../../../../h100x8/nvls/rca/multimem_contention/README.md).**

**Expectations the results contradicted (2026-11-19; the hypotheses below are kept as written):**

| Sub-experiment | Expected | Found |
|---|---|---|
| E1.2 | Raw bytes per payload byte constant on the busy direction | They fall as load rises (1.71 → 1.135 for `multimem.st`, 1.62 → 1.19 for `ld_reduce`) |
| E1.3 | Busy-cycle counters (R5) usable if accepted | The metric set was accepted but the counters were silently left out; R5 not available |
| E1.4 | About one 16-byte packet per 16-byte memory instruction of each thread | The instructions of a warp's 32 threads are merged into 128-byte packets (16 protocol bytes per 128 data bytes); multicast writes get one 16-byte response per packet |
| E1.6 | One of three explanations (combined packets, time-based traffic, smaller packets) | Packets add up in the mix; what vanishes is raw-counter traffic outside every packet type, present only on low-data directions |
| E1.7 | Fair sharing per packet, or priority for read replies | No priority; each operation's share follows its CTA count |


**Revision (2026-11-18), before any run.** Two raw-counter checks were added first (E1.1, E1.2), and the sub-experiments were renumbered in run order (old E1.0–E1.6 are now E1.3–E1.9). The old mix experiment predicted the mixed packets as the sum of each operation's packets alone. The 8-GPU raw counters already contradict that sum: in multicast AllReduce each operation's protocol traffic is a quarter to a third of its value alone ([series index](../PLANS.md#what-is-already-established)). The mix experiment (now E1.6) measures each packet type in the mix instead of testing the sum.

**Matrix cell:** NVLS-based collective × request generation and concurrency, and synchronization and progress. The "problem" is the mix itself: `multimem.st` and `multimem.ld_reduce` issued at the same time by the same GPUs.

**Question:** which NVLink packets does each multimem operation produce per payload byte and per direction, and why do they cost multicast AllReduce 1.35 raw bytes per data byte where every other pattern needs 1.13–1.18? The same quantity explains both problems of the [series](../PLANS.md#the-two-problems-this-series-explains).

Setup, instruments and metrics R1–R10: [00_common_setup.md](../00_common_setup.md). What is already known: [PLANS.md](../PLANS.md#what-is-already-established).

## Harness additions

| Addition | Needed by | Config key | Why |
|---|---|---|---|
| Session runner also saves `nvidia-smi nvlink -gt r` before and after every run (new snapshot key `link_raw`; the key `link` keeps the data counter) | E1.1, E1.2 | — | R9, R10 |
| Analysis script for long-minus-short pairs, data and raw, per GPU and direction | E1.1, E1.2 | — | R8–R10 |
| Instruction width (bytes one thread moves with one memory instruction): 16 B (`.v4.f32`, current), 8 B (`.v2.f32`), 4 B (`.f32`), for multimem and plain roles | E1.5 | `access_bytes` | Tells per-thread requests from coalesced requests |
| Address pattern: warp-contiguous (current: the 32 lanes of one instruction touch 32 × width contiguous bytes) or thread-contiguous (each thread owns 8 × width contiguous bytes, so the lanes of one instruction are 8 × width apart: one thread instruction per 128-byte line at 16 B, per 64 bytes at 8 B, per 32-byte sector at 4 B) | E1.5 | `pattern=contiguous|thread_contiguous` | Same. Both patterns keep 8 memory instructions in flight per thread and move every byte of a chunk once |
| Probe: one extra CTA on the last SM of the role order (no role reaches it), whose thread 0 times single outstanding 16-byte operations with `clock64`, converted to ns with `%globaltimer` over the whole loop (`ld_reduce` followed by a use of the value; or `multimem.st` followed by `fence.acq_rel.sys`), on its own 4 KiB slot per rank | E1.8 | `probe=none|reduce|multicast` | Latency per operation (R6) |
| Nsight Systems wrapper in the session runner (`"nsys": {"metrics_file": …}` in a session plan), a custom metric set with byte sums ([`nvlink_packets.config`](../../common/nsys/nvlink_packets.config)), and a parser of its SQLite export that keeps only samples inside the harness kernel intervals | E1.3 onward | — | R1–R5 |

Change of 2026-11-18 while implementing: the address pattern "one thread instruction per 128-byte line" cannot hold at every width without changing the accesses in flight per thread, so the thread-contiguous pattern above replaces it. Also on that day, the committed harness did not compile (a parameter name `startNsctasInRole` in `runRole`); restoring `ctasInRole` gave GPU code identical, instruction for instruction, to the 2026-11-01 binary that produced the observation results.

Check after the kernel changes: the multimem arms of the observation report (NVLS AllGather, ReduceScatter, AllReduce, 32 CTAs × 512 threads/CTA) rerun with `access_bytes=16`, `pattern=contiguous`, `probe=none` must give the recorded rates within 1%. The runner change of E1.1 adds one read-only snapshot and changes no kernel.

## E1.1 Raw link bytes at 4 GPUs

### Motivation

At 8 GPUs, every pattern runs its busiest direction at about 420 GB/s raw, and multicast AllReduce loses its data rate because it spends 1.35 raw bytes per data byte. The 4-GPU mixing result (0.865 of the mean of NVLS AllGather and ReduceScatter) was measured with the data counter only. If the 8-GPU explanation holds at 4 GPUs, the mixed arms reach the same raw rate as the arms alone, with more raw bytes per data byte. If the mixed arms reach a lower raw rate, part of the loss is idle link time, not protocol traffic.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Arm**, from the observation report |
| Arms (all on 4 GPUs, each GPU issues) | NVLS AllGather: `multimem.st` 32 CTAs × 512 threads/CTA. NVLS ReduceScatter: `ld_reduce` 32 × 512. NVLS AllReduce: both on the same 32 × 512 CTAs. Separate CTAs: `multimem.st` 32 × 512 + `ld_reduce` 32 × 512. Plain store: 32 × 512 to the next GPU. Plain load: 32 × 512 from the next GPU. Plain mix: plain load 32 × 512 + plain store 32 × 512 |
| Fixed | GPUs 4–7, 16-byte packs, warp-contiguous, 2 GiB part per rank, 300 ms windows |
| Long and short run | 1 warm-up + 10 kept windows; 1 warm-up + 2 kept windows |
| Rounds | 3; arms rotated, long and short of one arm back to back |
| Configs | `configs/e1_1/e1_1_<arm>_<long|short>.cfg`, written by the runner from `configs/plans/e1_1.json` |

### Procedure

1. 7 arms × 2 lengths × 3 rounds = 42 runs in one guarded session.
2. Per arm and direction: R8, R9, R10, and raw ÷ data = R9 ÷ R8.
3. Check R8 against the byte model (for example 1.25 per direction for NVLS AllReduce, 1 and 0.25 for NVLS AllGather). A mismatch of more than 1% stops the analysis.
4. Data rate per direction from the windows (N1 × bytes per payload byte), compared with R10 ÷ (raw ÷ data). Both come from the same runs, so this is a consistency check, not a prediction.

### Privileged access

None beyond free GPUs 4–7. The driver counters are read-only.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| R10, busiest direction | About equal for all multimem arms, mixed or not, as at 8 GPUs | Protocol traffic (equal raw rate) from idle link time (lower raw rate in the mix) |
| Raw ÷ data, busiest direction | About 1.13–1.16 for the arms alone and the plain arms; about 1.3–1.35 for NVLS AllReduce and separate CTAs | Whether the 8-GPU accounting holds at 4 GPUs |
| R9 on the low-data direction of NVLS AllGather and ReduceScatter | Near the 8-GPU values (0.89 and 1.04 raw bytes per payload byte), scaled by the 4-GPU byte model | Control cost per operation at this GPU count |
| R9, plain mix against plain arms | Raw ÷ data rises little (the plain mix lost only 3–7% of data rate) | Whether the plain control traffic follows the same rule |

## E1.2 Low-data-direction raw bytes against issue rate

### Motivation

The low-data direction of NVLS AllGather carries 0.89 raw bytes per payload byte at 8 GPUs, almost all of it protocol traffic. That number is used as "the cost of write responses". It is only a cost per request if it scales with the number of requests. If part of it is sent at a fixed rate whatever the traffic (for example flow-control messages), then it overstates the cost per request, and this could be why the protocol traffic of the two operations does not add up in the mix. Lowering the issue rate tells the two apart: per-request traffic keeps R9 constant, while time-based traffic keeps R10 constant and makes R9 rise as the rate falls.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Issue rate**, set by the CTAs × threads/CTA of one role. The achieved rate is measured (N1), not assumed from the setting |
| Settings, `multimem.st` alone (NVLS AllGather) | 1 × 32, 1 × 64, 1 × 128, 1 × 512, 2 × 512, 32 × 512 CTAs × threads/CTA |
| Settings, `ld_reduce` alone (NVLS ReduceScatter) | 1 × 32, 1 × 128, 1 × 512, 2 × 512, 32 × 512 |
| Fixed | All 4 GPUs issue the role; GPUs 4–7; 16-byte packs, warp-contiguous; 300 ms windows |
| Long and short run | As E1.1 |
| Rounds | 3; settings rotated |
| Configs | `configs/e1_2/e1_2_<role>_c<ctas>_t<threads>_<long|short>.cfg`, from `configs/plans/e1_2.json` |

The observation calibration was flat from 4 CTAs × 256 threads/CTA up, so all reduced rates come from 1 and 2 CTAs. If 1 × 32 still reaches more than half of the 32 × 512 rate, the lower end is not reached; record it and add a setting only if the harness allows one (it allows at least 1 CTA and 32 threads).

### Procedure

1. 11 settings × 2 lengths × 3 rounds = 66 runs in one guarded session.
2. Per setting: N1 (achieved rate), R8 and R9 on both directions, R10 on the low-data direction.
3. Plot R9 and R10 of the low-data direction against the achieved data rate.

### Privileged access

None beyond free GPUs 4–7.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| R9, low-data direction | Constant across rates: the protocol traffic is per request | Per-request traffic from time-based traffic |
| R10, low-data direction | Falls in proportion to the data rate if per request; stays high (about 300 GB/s or more) at low data rates if time-based | Same |
| R9, busy direction | Constant (headers of data packets) | Checks that the method gives constant values where they are expected |

If R9 on the low-data direction is constant, the per-request cost stands, and the non-additivity must come from how packets change in the mix (E1.6). If R10 stays high, the low-data-direction value is not a per-request cost, and the "protocol traffic per operation" in the series index must be corrected before E1.6.

## E1.3 Instrument check

### Motivation

The NVLink counters of Nsight Systems have not been used on this machine. Their sums must match the driver counters, and sampling must not slow the kernels.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Nsight Systems sampling on or off** |
| Arms | NVLS AllGather, NVLS ReduceScatter, plain store; each 32 CTAs × 512 threads/CTA on GPUs 4–7 |
| Rounds | 3 |

### Procedure

1. Each arm with and without sampling, interleaved.
2. Compare: user bytes (requests + responses) per direction against the data counter (`-gt d`); user + protocol bytes against the raw counter (`-gt r`); both over the same run, and against the E1.1 values of the same arm.
3. Try the custom metric set with `nvl{tx,rx}__cycles_active`; record whether it is accepted.

### Privileged access

Profiling counters (root). Free GPUs 4–7.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| R1–R4 sums against the data and raw counters | Within 2% | A usable instrument from a misread one |
| N1 with and without sampling | Within 1% | Probe effect |

## E1.4 Packet anatomy of each operation alone

### Motivation

Before explaining the mix, each operation's packets must be known: per payload byte, how many request and response bytes, payload and protocol, on the issuing GPU and on the GPUs that serve it. From the raw counters at 8 GPUs, the low-data direction of `ld_reduce` carries about 0.9 protocol bytes per payload byte and that of `multimem.st` about 0.74. That is near one 16-byte packet per 16-byte memory instruction of each thread, which would mean that the instructions of a warp's 32 threads are not merged. E1.2 says whether these values are per request.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Operation: `multimem.st`, `multimem.ld_reduce`, plain load, plain store**, each issued by all 4 GPUs |
| CTAs × threads/CTA | 32 × 512 (flat part of the observation calibration) |
| Memory instructions | 16 B per thread per instruction, warp-contiguous addresses |
| Rounds | 3 |

### Procedure

1. One run per operation and round, with sampling.
2. Per GPU and direction: R1–R4.
3. Thread instructions per payload byte are known from the code (one per 16 bytes per serving GPU for multimem). Divide R2 and R4 by them: protocol bytes per access.

### Privileged access

As E1.3.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| R2 on the serving GPU's receive direction, `ld_reduce` | About 0.9 if each thread's 16-byte instruction sends its own one-unit (16-byte) read request; 0.5, 0.25 or 0.125 if requests cover 32, 64 or 128 bytes | Per-access requests from merged requests |
| R4 on the receiving GPU's send direction, `multimem.st` | About 0.75–1 if each 16-byte store gets its own response; much less if responses are merged | Same, for writes |
| Same, plain load and plain store | Lower than multimem if plain instructions of a warp are merged into sectors or lines | Whether per-access packets are specific to multimem |

## E1.5 Granularity: instruction width and address pattern

### Motivation

E1.4 gives packet bytes for one access size. Changing the size per thread and the address pattern tells whether packets follow each thread's memory instruction (one packet per instruction) or the memory lines (instructions of a warp merged). NCCL never issues multimem accesses wider than 16 B, so fewer, larger accesses are not available; only smaller ones are.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Instruction width (bytes per thread per memory instruction): 4, 8, 16 B × address pattern: warp-contiguous, thread-contiguous** |
| Operations | `multimem.st` alone, `multimem.ld_reduce` alone; plain load and plain store at 16 B as controls; each 32 CTAs × 512 threads/CTA on all 4 GPUs |
| Rounds | 3 |

### Procedure

1. 2 operations × 3 widths × 2 patterns × 3 rounds = 36 runs, plus 2 controls × 2 patterns × 3 = 12.
2. R1–R4 and N1 per arm.

### Privileged access

As E1.3.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| R2 (`ld_reduce`), R4 (`multimem.st`) against width | One packet per instruction: protocol bytes per payload byte grow as 1/width and do not depend on the pattern. Merged packets: they depend on the pattern (contiguous lower) and hardly on width | How multimem requests are formed |
| N1 against width | Falls with smaller width if one packet per instruction fills the link | Whether packet count limits the rate |

## E1.6 Packet anatomy of the mix

### Motivation

This is the core test. The 8-GPU raw counters show that the mix carries much less protocol traffic than the two operations carry alone, about 0.2 protocol bytes per payload byte per direction against 0.74 and 0.90. Some packets therefore change when the operations are mixed. Three explanations are open:

1. **Combined packets.** Responses and requests travel inside data packets going the same way, so they cost little extra on a direction full of data.
2. **Not per request.** Part of the low-data-direction traffic is sent at a fixed rate, not per request (tested in E1.2).
3. **Different packets.** Requests or responses take a smaller form when the direction is busy.

The per-type counters separate request and response bytes, and payload and protocol bytes, in the mix itself. So they show which packet type shrinks, without any prediction from the arms alone.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Mix: NVLS AllReduce (both operations on the same 32 × 512 CTAs) and separate CTAs (`multimem.st` 32 × 512 + `ld_reduce` 32 × 512)** |
| Reference | NVLS AllGather (`multimem.st` 32 × 512) and NVLS ReduceScatter (`ld_reduce` 32 × 512), run in the same session |
| Memory instructions | 16 B per thread per instruction, warp-contiguous addresses |
| Rounds | 5, interleaved with the arms alone |

### Procedure

1. Interleave mixed and alone arms.
2. Per direction: R1–R4 of each arm, per payload byte of each operation.
3. For each packet type (request payload, request protocol, response payload, response protocol), compare its bytes per payload byte in the mix with its value in the arm alone that produces it.
4. R5 per direction, if available.

### Privileged access

As E1.3.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| R2 and R4 (protocol bytes) in the mix against the arms alone | Lower in the mix under explanations 1 and 3; unchanged per request under explanation 2 if E1.2 found time-based traffic | Which packets shrink |
| R1 and R3 (payload bytes) in the mix against the arms alone | Unchanged; payload is fixed by the byte model | Checks that only protocol traffic changes |
| R5 | Near 1 on both directions if the link is always busy; below 1 with data waiting means flow control or arbitration idles the link | Protocol traffic from idle link time |

## E1.7 Arbitration: share of CTAs per operation

### Motivation

The split mix showed that read replies keep their rate and multicast stores take what is left. Varying how much each operation asks for shows the rule the link or switch applies when both want the same direction.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **`ld_reduce` CTAs: 0, 4, 8, 16, 32, 64 with `multimem.st` fixed at 32 CTAs; and `multimem.st` CTAs: 0, 4, 8, 16, 32, 64 with `ld_reduce` fixed at 32 CTAs** |
| Threads/CTA, memory instructions | 512 for both roles; 16 B per thread per instruction, warp-contiguous addresses |
| Rounds | 3, settings rotated |

### Procedure

1. 11 settings × 3 rounds = 33 runs.
2. Per setting: N1 of each operation, R1–R4 per direction.
3. Plot each operation's rate against its own and the other's demand.

### Privileged access

As E1.3.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| N1 of each operation | Fair sharing per packet: each operation's share follows its share of packets in demand. Priority for responses: `ld_reduce` keeps its alone rate at every `multimem.st` demand, and stores get the remaining link time | Fair arbitration from priority |

## E1.8 Latency and bytes in flight

### Motivation

The peer report proposes a shared limit on requests in flight below the SMs. If such a pool exists, bytes in flight (rate × latency) stay constant when the mix raises latency. If the link queue is the limit, latency rises with the share of the direction taken by other packets, and bytes in flight need not stay constant.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Background traffic: none, `multimem.st` alone, `ld_reduce` alone, NVLS AllReduce, plain store alone**, each 32 CTAs × 512 threads/CTA on all 4 GPUs |
| Probe | `probe=reduce` and `probe=multicast`, one warp on one SM per GPU, single outstanding operation |
| Rounds | 3 |

### Procedure

1. 5 backgrounds × 2 probes × 3 rounds = 30 runs.
2. R6 (median, 90th percentile) per probe and background; R7 for the background roles.

### Privileged access

As E1.3.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| R6, `ld_reduce` probe | Rises most when the direction carrying its requests or replies is shared with store data | Which direction delays reads |
| R6, `multimem.st` probe | Rises most when its data direction is shared with read replies | Which direction delays stores |
| R7 | Constant across backgrounds: a pool limit. Not constant: queueing on the link | In-flight pool from link queueing |

## E1.9 8-GPU check with NCCL kernels

### Motivation

The harness has 4 GPUs. The 8-GPU raw counters come from NCCL's kernels. The packet costs must be checked there, and Ring AllReduce, which needs only 1.16 raw bytes per data byte with both directions full, shows what the packets of a cheap two-way pattern look like.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Kernel: `AllGather_STMC`, `ReduceScatter_LDMC`, `AllReduce_RSxLDMC_AGxSTMC`, Ring AllReduce** |
| Program | nccl-tests, `-t 1 -g 8`, 2 GiB, 200 calls. CTAs × threads/CTA: NCCL's choice, multicast 2 × 512, Ring 24 × 512 |
| GPUs | 0–7 |
| Rounds | 3 |

### Procedure

1. Sampling over the timed calls; R1–R4 per direction, per payload byte.
2. Compare with E1.4 and E1.6, scaled from 4 to 8 GPUs by the byte model.

### Privileged access

All 8 GPUs free; counters.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| R1–R4, multicast kernels | Same costs per thread instruction as in the harness | Harness artefact from real behaviour |
| R2 and R4, Ring AllReduce | Small protocol bytes per payload byte on both directions | What a two-way pattern without remote reads costs |

## Run order and time

| Step | GPUs | Runs | Time (estimate) |
|---|---|---|---|
| Runner change and long-minus-short script | none | — | under 1 hour of code work |
| E1.1 | 4–7 | 42 | about 7 min (about 8–10 s per run) |
| E1.2 | 4–7 | 66 | about 10 min |
| Kernel additions, parser, recheck of recorded arms | 4–7 for the recheck | about 6 | about 3 hours of code work; 5 min GPU |
| E1.3 | 4–7 | 18 | 5 min |
| E1.4 | 4–7 | 12 | 3 min |
| E1.5 | 4–7 | 48 | 12 min |
| E1.6 | 4–7 | 20 | 5 min |
| E1.7 | 4–7 | 33 | 8 min |
| E1.8 | 4–7 | 30 | 8 min |
| E1.9 | 0–7 | 12 | 5 min |

E1.1 and E1.2 decide whether E1.6's question stands as written; review their results before the kernel additions. Not observable from inside this VM: the NVSwitch itself (no switch device or telemetry); everything about the switch is inferred from the GPUs' link counters and timing.
