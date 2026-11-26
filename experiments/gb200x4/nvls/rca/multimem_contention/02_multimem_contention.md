# Plan 02: what takes the link time that NVLS AllReduce's data loses on GB200

**Status (2026-11-26): plan only. Runs only after [plan 01](../degradation_check/01_degradation_check.md) passes its gate; the branch is chosen from plan 01's result.** Series: [PLANS.md](../PLANS.md). Setup and metrics: [00_common_setup.md](../00_common_setup.md). The H100 version of this plan, whose methods are reused: [H100 plan 01](../../../../h100x4/nvls/rca/multimem_contention/01_multimem_contention.md).

**Matrix cell:** NVLS-based collective × request generation and concurrency, and synchronization and progress.

**Question:** when `multimem.st` and `multimem.ld_reduce` run at the same time on GB200, which traffic, idle time or limit takes the time that the data loses, on which link direction?

## Branches

| Plan 01 case | Sub-experiments | Why |
|---|---|---|
| A. Link full, more raw bytes per data byte in the mix | E2.1, E2.2, E2.3, E2.4, E2.5; E2.6 only if the ceiling share (G2) is below 0.85 | Account for the extra raw bytes, as on H100, and test whether the same two packet kinds explain them |
| B. Raw link rate lower in the mix | E2.1, E2.4, E2.5, E2.6 | Find what idles the link or limits issue before it |
| C. No degradation | E2.7 only (optional) | Check whether it appears at 8 GPUs across two nodes |

All sub-experiments use the `mm_mix` harness as ported in plan 01 E1.5, on GPUs 0–3, at plan 01's calibrated CTAs × threads/CTA per role, 16-byte accesses and warp-contiguous addresses unless the table says otherwise. Every comparison runs inside one session.

## E2.1 Low-data-direction traffic against issue rate

### Motivation

On H100 the low-data direction of each operation alone carried much more raw traffic than its packets (0.55–0.66 bytes per payload byte outside every packet type), and that traffic did not carry over into the mix. Whether this traffic is per request or per unit time decides how the raw bytes of the arms alone can be used to predict the mix. On GB200, without packet counters, this test is the main way to tell request traffic from time-based traffic.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Issue rate**, set by CTAs × threads/CTA of one role; the achieved rate (N1) is measured |
| `multimem.st` alone | 1 × 32, 1 × 128, 1 × 512, 2 × 512, 8 × 512, calibrated count × 512 (CTAs × threads/CTA) |
| `ld_reduce` alone | Same settings |
| Fixed | All 4 GPUs issue the role; long and short run pairs |
| Rounds | 3, settings rotated |

### Procedure

1. 2 roles × 6 settings × 2 lengths × 3 rounds = 72 runs.
2. Per setting: N1, R8, R9 on both directions, R10 on the low-data direction.
3. Plot raw bytes per payload byte (R9) and raw link rate (R10) of the low-data direction against the achieved data rate, both axes from zero.

### Privileged access

None.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| R9, low-data direction | Constant across rates if the traffic is per request | Per-request from time-based traffic |
| R10, low-data direction | Falls with the data rate if per request; stays high at low rates if time-based | Same |
| R9, busy direction | Constant (data headers) | Checks the method where a constant is expected |

## E2.2 Packet types (needs counter access)

### Motivation

On H100 the per-type counters (request and response bytes, data and protocol) showed exactly which packets fill the busy directions of the mix: the switch's read requests for `ld_reduce` and the responses to `multimem.st`. On GB200 the same counters need admin rights. This sub-experiment is the direct test of the H100 mechanism; without it, plan 02 can only test it indirectly (E2.3).

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Arm**: `multimem.st` alone, `ld_reduce` alone, both on the same CTAs, both on separate CTAs, plain store alone, plain load alone |
| Instrument | Nsight Systems GPU metrics on all 4 GPUs at 10 kHz, NVLink packet metrics of the GB200 metric set (names to be read from `nsys profile --gpu-metrics-set=help` and the set's metric list in an admin session) |
| Rounds | 3 profiled + 3 unprofiled, interleaved |

### Procedure

1. Ask the cluster admins (sysadmin@empireai.edu) for one of: a session on a node with `RmProfilingAdminOnly=0`, or an admin running a prepared script. State the GPUs, the duration (about 30 minutes) and that the counters are read-only.
2. Instrument check: profiler data bytes (request data + response data) against the driver data counter per GPU and direction, within 2%; N1 with and without profiling, within 1%.
3. Per arm and direction: request data, request protocol, response data, response protocol bytes per payload byte (R1–R4); raw bytes outside the packet counters = R9 − (R1 + R2 + R3 + R4).
4. Compare each packet type in the mix with its value in the arm alone that produces it.

### Privileged access

**Admin.** Fallback if not granted: skip E2.2 and rely on E2.3; state in the report that the packet split was not measured on GB200.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| R2 on the received direction, `ld_reduce` (switch read requests) | Present on the busy direction in the mix (H100: 0.23 per payload byte) | Whether the H100 read-request cost exists on NVLink5 |
| R4 on the sent direction, `multimem.st` (write responses) | One 16-byte response per 128-byte packet (H100: 0.125) | Same, for write responses |
| Raw outside packets | Large on the low-data direction alone, near zero on busy directions (H100) | Whether the H100 accounting holds |

## E2.3 Packet formation: instruction width and address pattern

### Motivation

If the extra raw bytes are per packet, changing how many packets carry the same data changes the raw bytes per data byte in a predictable way. On H100 a warp's 32 threads were merged into 128-byte packets at every width; only scattered (thread-contiguous) addresses made one packet per instruction. On GB200 this test works with the raw counter alone, so it is the indirect test of the packet explanation when E2.2 is not possible.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Instruction width (4, 8, 16 bytes per thread per memory instruction) × address pattern (warp-contiguous, thread-contiguous)** |
| Roles | `multimem.st` alone, `ld_reduce` alone, both on the same CTAs; plain store and plain load at 16 bytes as controls |
| Rounds | 3 |

### Procedure

1. 3 roles × 3 widths × 2 patterns × long/short × 3 rounds = 108 runs, plus 2 controls × 2 patterns × 2 × 3 = 24.
2. Per arm: N1, R8, R9, G5 on both directions.
3. For each role, fit raw bytes per payload byte against packets per payload byte (from the width and pattern: 128-byte packets when warp-contiguous, one packet per instruction when thread-contiguous). The slope is the non-data bytes per packet on that direction.

### Privileged access

None.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| Raw ÷ data (G5) against width, warp-contiguous | Flat if warps are merged into 128-byte packets (H100) | Merged from per-instruction packets |
| G5 against width, thread-contiguous | Rises as 1/width | Same |
| Slope per packet, low-data direction of each role alone | About 16 bytes per packet for `multimem.st` responses (H100) | The size of the per-packet cost that the mix moves to busy directions |
| AllReduce G5 predicted from the roles' slopes against measured | Close under case A | Whether per-packet costs explain the mix |

## E2.4 Arbitration: share of CTAs per operation

### Motivation

When both operations want the same direction, the rule that shares it decides which one loses. On H100 each operation's share followed its CTA count, with no priority for read replies.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **`ld_reduce` CTAs: 0, 4, 8, 16, 32, 64 with `multimem.st` fixed at 32 CTAs; and `multimem.st` CTAs: 0, 4, 8, 16, 32, 64 with `ld_reduce` fixed at 32 CTAs** (all × 512 threads/CTA; total ≤ 152 SMs) |
| Fixed | Separate CTAs for the two roles, all 4 GPUs |
| Rounds | 3, settings rotated |

### Procedure

1. 11 settings × 3 rounds = 33 runs (long only; one long/short pair at the 32 + 32 point).
2. N1 of each role against its own and the other's CTA count.

### Privileged access

None.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| N1 of each role | Share follows CTA share (H100); or read replies keep their rate (priority) | Fair sharing from priority |

## E2.5 Latency and bytes in flight

### Motivation

If a fixed pool of requests in flight limits the mix, bytes in flight (rate × latency) stay constant while latency rises. If queues on the link set the limit, latency rises with the share of the direction taken by other traffic. Under case B this is the main test of where requests wait.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Background traffic: none, `multimem.st` alone, `ld_reduce` alone, both on the same CTAs, plain store alone**, at calibrated CTAs × 512 threads/CTA |
| Probe | One warp on one SM per GPU, one outstanding 16-byte operation: `probe=reduce` and `probe=multicast` (harness probe of the H100 series) |
| Rounds | 3 |

### Procedure

1. 5 backgrounds × 2 probes × 3 rounds = 30 runs.
2. Probe latency (R6, median and 90th percentile); bytes in flight (R7) of the background roles.

### Privileged access

None.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| R6, `ld_reduce` probe | Rises most when its request or reply direction is shared with store data | Which direction delays reads |
| R6, `multimem.st` probe | Rises most when its data direction is shared with read replies | Which direction delays stores |
| R7 | Constant across backgrounds under a pool limit | Pool limit from link queueing |

## E2.6 What limits each direction below nominal

### Motivation

If plan 01 finds every pattern well below the 956 GB/s nominal (ceiling share G2 below 0.85), the H100 picture of "a full link" does not apply as such, and the mix may be losing to a limit before the link (SM issue, outstanding requests, the switch). Comparing the raw ceiling of different ways of putting bytes on the same direction separates a link limit from a per-path limit.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **How the GPU puts bytes on one direction**: plain stores to 1 peer, plain stores to 3 peers, `multimem.st`, `ld_reduce` replies (sent direction), plain loads from 3 peers (received direction), copy engine (`cudaMemcpyPeerAsync` to 3 peers) |
| CTAs × threads/CTA | Each calibrated to its plateau (stopping rule of plan 01 E1.2); copy engine has none |
| Rounds | 3 |

### Procedure

1. 6 paths × calibration (about 8 settings) × 3 rounds ≈ 150 runs, then long/short at each plateau.
2. R10 and G2 per path and direction.

### Privileged access

None.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| R10 at the plateau, per path | All paths reach the same raw rate if the link is the limit; different rates mean per-path limits | Link limit from per-path (issue or switch) limit |
| Copy engine against SM paths | Copy engine higher would point to an SM-side issue limit | SM issue limit from link or switch limit |

## E2.7 8-GPU check across two nodes of one NVL72 rack (optional)

### Motivation

On H100 the loss had the same size at 4 and 8 GPUs, and the gain over Ring grew with N (1.04 at 4 GPUs, 1.33 at 8). On GB200 8 GPUs means two compute trays of the same rack (multi-node NVLink), which `beta_test` allows. This is an 8-GPU experiment: configs and results go in `experiments/gb200x8/nvls/rca/multimem_contention/` with a short report that links back here.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Arm**: AG-MC, RS-MC, AR-MC, AR-Ring (plan 01 names) |
| Program | nccl-tests with MPI, 8 processes (`-N 2 --ntasks-per-node=4 --gres=gpu:4 -c 36 --constraint="[rack2|rack3]"`, both nodes in one rack), 2 GiB, CTAs from a short calibration as in plan 01 E1.2 |
| Rounds | 3, plus long/short counter pairs |

### Procedure

1. Check that NCCL reports NVLS multicast across both nodes (MNNVL) and no network transport.
2. S1, S6, G3, G4, R8–R10 as in plan 01 E1.3 and E1.4.

### Privileged access

A two-node `beta_test` session in one rack (both nodes of rack 2 or of rack 3). MPI inside the container to be checked.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| G3 and G4 at 8 GPUs | G3 similar to 4 GPUs; G4 larger than at 4 GPUs (byte-count bound 1.56) | Whether the GB200 result depends on the GPU count |

## Run order and time

| Session | Sub-experiments (branch A) | GPU time (estimate) |
|---|---|---|
| 1 | E2.1 (72 runs), E2.4 (35 runs) | 45 min |
| 2 | E2.3 (132 runs) | 60 min |
| 3 | E2.5 (30 runs), E2.6 if needed | 30–80 min |
| Admin session | E2.2 | 30 min |
| Two-node session | E2.7 (optional) | 40 min |

Branch B replaces E2.3 by E2.6 in session 2. A finding that contradicts an expectation above is recorded in the report and, if it changes the design, in a dated note at the top of this plan.
