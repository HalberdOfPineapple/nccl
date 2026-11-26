# Plan 01: does NVLS AllReduce lose data rate on GB200, and is the link full when it does

**Status (2026-11-26): E1.1–E1.4 done (E1.1, E1.2 interactive on `b3-11`; E1.3, E1.4 batch job 238369 on `b3-36`). Pending: E1.3 repeat on a second node, E1.5. Results: [README.md](README.md).** Series: [PLANS.md](../PLANS.md). Setup and metrics: [00_common_setup.md](../00_common_setup.md). Harness and how to run: [`../../common/README.md`](../../common/README.md).

**Revision (2026-11-26), while implementing; the expectations below are kept as written:**

| Change | Why |
|---|---|
| Kernel selection fixed: symmetric kernels with `NCCL_SYM_KERNEL=<collective>:<kernel>` and `-R 2`; CTAs with `NCCL_SYM_CTAS` (NCCL's maximum is 64); Ring with `NCCL_ALGO=<collective>:ring NCCL_PROTO=Simple`; non-symmetric NVLS with `NCCL_ALGO=allreduce:nvls` | Checked in the NCCL source and in the logs (`-> Kernel <name> nchannels <CTAs>`, `-> Algo RING ... channel{Lo..Hi}`) |
| AG-MC also sets `NCCL_SYM_CE_THRESHOLD=2^40` | Without it a registered AllGather above 8 MiB runs on the copy engine even when `NCCL_SYM_KERNEL` names `AllGather_STMC`; the first attempt's AG-MC runs were such runs and that attempt is excluded |
| New arm AG-CE: the copy-engine AllGather (NCCL's own choice above 8 MiB), no CTA knob | NCCL uses it for real AllGathers of this size, so it is the practical AllGather reference. It reached 945 GB/s algBW at 2 GiB, but it sends unicast copies to the 3 peers (data counter: 0.75 bytes per payload byte each way), so per direction it is 709 GB/s data and 739 GB/s raw (77% of nominal), not a full link. An earlier note in this session that it filled the link was wrong (busBW factor 1 applied by mistake) |
| E1.2 uses `-n 30 -w 5` instead of `-n 50 -w 10` | To fit the calibration into one 2-hour interactive session; each point is still 3 rounds |
| E1.3, E1.4 and the E1.3 repeat run as batch jobs (`common/jobs/plan01.sbatch`), not in interactive sessions | Interactive sessions are limited to 2 hours and reserved for calibration; each batch job is one session (one node) for the pairing rule |
| Plans are JSON files in [`configs/plans/`](configs/plans/); the calibrated CTA counts the batch jobs use are in `configs/calibrated_ctas.json` | One file per sub-experiment; every run's full configuration is also saved in its `result.json` |

**Matrix cell:** NVLS-based collective × request generation and concurrency. The "problem" is the mix itself: `multimem.st` (the AllGather half) and `multimem.ld_reduce` (the ReduceScatter half) issued at the same time by the same GPUs.

**Question:** on one GB200 node (4 GPUs), does NVLS AllReduce move less data per link direction than NVLS AllGather and NVLS ReduceScatter alone, by how much, and does each busy link direction then run at the same raw rate as in the other patterns?

This plan is a gate for [plan 02](../multimem_contention/02_multimem_contention.md). It measures completed-collective throughput first, against matched references, and uses link counters only to classify the result into the three cases of the [series decision table](../PLANS.md#decision-after-plan-01).

## Arms used throughout

All arms: 4 GPUs (physical 0–3 of the session's node), `float`, `sum` where it applies, out-of-place, nccl-tests `-t 1 -g 4`.

| Arm | Collective | NCCL path | How it is selected (to be verified in E1.1) | Link operations |
|---|---|---|---|---|
| AG-MC | AllGather | Symmetric `AllGather_STMC` | `-R 2` (symmetric registration) + kernel selector | `multimem.st` |
| RS-MC | ReduceScatter | Symmetric `ReduceScatter_LDMC` | `-R 2` + kernel selector | `multimem.ld_reduce` |
| AR-MC | AllReduce | Symmetric `AllReduce_RSxLDMC_AGxSTMC` | `-R 2` + kernel selector | Both, concurrently |
| AR-NVLS | AllReduce | Non-symmetric NVLS (`NCCL_ALGO=NVLS`), the kernel the pilot used | `NCCL_ALGO=NVLS` | Both, through NVLS channels |
| AR-Ring | AllReduce | Ring, Simple protocol | `NCCL_ALGO=Ring NCCL_PROTO=Simple` | Plain stores to the next GPU |
| AR-TMA | AllReduce | Symmetric `AllReduce_RSxTmaLD_AGxTmaST` (GB200 only) | `-R 2` + kernel selector | Plain TMA loads and stores, no multicast |
| AR-LDST | AllReduce | Symmetric `AllReduce_RSxLD_AGxST` | `-R 2` + kernel selector | Plain loads and stores, no multicast |
| AG-Ring, RS-Ring | AllGather, ReduceScatter | Ring, Simple | `NCCL_ALGO=Ring NCCL_PROTO=Simple` | Plain stores |

Kernel selector: this NCCL accepts algorithm names from its registry (`SYMK_RSxLDMC_AGxSTMC`, `SYMK_RSxTmaLD_AGxTmaST`, ...) through `NCCL_ALGO`; the H100 runs used `NCCL_SYM_KERNEL=<collective>:<kernel>`. E1.1 finds the working form; nothing else in the plan depends on which.

## E1.1 Session, kernel selection and counter check

### Motivation

Every later number depends on three things that have not been checked on GB200: that each arm runs the kernel it names, that the driver's data counter counts what the byte model says, and that the raw counter is at least the data counter on every link. A wrong kernel or a misread counter would look like a degradation or hide one.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Arm** (all 9 above) |
| Payload | 2 GiB per buffer (as the H100 series) |
| CTAs × threads/CTA | NCCL defaults (recorded from the log) |
| Counter runs | Long (`-n 200 -w 20`) and short (`-n 40 -w 20`), `-c 0`, back to back |
| Correctness run | One run per arm with `-c 1 -n 5` |
| Rounds | 1 |
| Configs | `configs/e1_1/e1_1_<arm>_<long|short|check>.cfg` |

### Procedure

1. Record the session context (G1).
2. For each arm: correctness run; then long and short runs with link snapshots.
3. From each NCCL log: the algorithm or kernel name, channels or CTAs, threads per CTA. A run whose log does not name the intended kernel is excluded and the selector is fixed before E1.2.
4. Data counter (R8) per GPU and direction against the byte model: AG-MC 1 received and 1/4 sent per payload byte; RS-MC the reverse; AR-MC 1.25 both ways; Ring AllReduce 1.5 both ways. A mismatch over 1% stops the plan until it is explained.
5. Raw counter ≥ data counter on every link and direction; per-link spread of raw and data across the 18 links of each GPU.

### Privileged access

None. Free node (whole-node allocation).

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| Kernel names in the log (C8 in the observation setup) | Each arm runs its named kernel; AR-TMA may need a minimum size per CTA count | A real comparison from a wrong-kernel artefact |
| Data bytes per payload byte (R8) | Equal to the byte model within 1% | A usable data counter from a misread one |
| Raw bytes per data byte (G5) on the busy direction | Above 1 (headers); near the H100 values 1.13–1.19 for AG-MC and RS-MC if packets are formed the same way | Whether the raw counter has the H100 meaning on NVLink5 |
| Per-link spread | Below 2% across the 18 links | Even striping over links (else per-link limits matter) |

## E1.2 Saturation calibration: CTAs per kernel

### Motivation

The pilot ran every arm at NCCL's default CTA count and found busBW of 540–650 GB/s, 56–68% of the 956 GB/s nominal. On H100 every pattern ran within 5% of nominal. If GB200 kernels are limited by how many requests their CTAs issue, a "degradation" between arms could be only a difference in default CTA counts. The comparison is valid only at CTA counts where more CTAs stop helping.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **CTAs per kernel** |
| Symmetric arms (AG-MC, RS-MC, AR-MC, AR-TMA, AR-LDST) | `NCCL_SYM_CTAS` = 1, 2, 4, 8, 16, 24, 32, 48, 64 (each value confirmed in the log; the largest is the largest NCCL accepts) |
| Channel arms (AR-NVLS, AR-Ring, AG-Ring, RS-Ring) | `NCCL_MIN_NCHANNELS = NCCL_MAX_NCHANNELS` = 4, 8, 16, 24, 32, 48, 64; for AR-NVLS also `NCCL_NVLS_NCHANNELS` = 4, 8, 16, 24, 32 |
| Threads/CTA | NCCL default (recorded); a second pass at `NCCL_NTHREADS` = 256 and 1024 for the channel arms only if the CTA curve has not flattened |
| Fixed | 2 GiB, `-n 50 -w 10`, `-c 0` |
| Rounds | 3, CTA values rotated within each round |
| Configs | `configs/e1_2/e1_2_<arm>_c<ctas>[_t<threads>].cfg` |

### Procedure

1. 9 arms × about 8 CTA values × 3 rounds ≈ 220 runs over one session.
2. Per arm: algBW (S1) and busBW (S6) against CTAs × threads/CTA, all points shown.
3. **Stopping rule (stated before the runs):** the plateau starts at the first CTA count after which two successive increases each add less than 1% of busBW, with the spread between rounds below 1%. The calibrated count is the smallest count on the plateau. If an arm still rises at the largest accepted count, it is "not saturated" and that is reported; it is not given a plateau.
4. A downturn at high CTA counts is reported as a downturn, not smoothed.
5. One long/short counter pair at the calibrated count of each arm (raw link rate R10), to see whether the plateau is a link ceiling.

### Privileged access

None.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| busBW (S6) against CTAs | Flattens for every arm. If AR-MC flattens below AG-MC and RS-MC, the gap is not a CTA-count artefact | A real degradation from a default-configuration artefact |
| Raw link rate (R10) at the plateau | Near the same value for all arms if the link is the limit (H100 case); different values if each kernel stops at its own issue limit | Link ceiling from issue ceiling |
| Ceiling share (G2) | Unknown on GB200; H100 was 0.95 | Whether anything other than the link limits every pattern |

## E1.3 Throughput against payload size

### Motivation

This is the primary measurement: completed-collective throughput of NVLS AllReduce against its two halves and against Ring, at the calibrated CTA counts and at NCCL's defaults, over the sizes where a real workload uses AllReduce. It answers "is there a degradation on GB200, and how large" without any counter.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Arm × payload size** |
| Arms | All 9, each at its calibrated CTA count from E1.2 (configuration label: arm, CTAs × threads/CTA) and at NCCL defaults (label: arm, "NCCL default", with the CTAs × threads/CTA read from the log) |
| Payload | 64 MiB, 128 MiB, 256 MiB, 512 MiB, 1 GiB, 2 GiB, 4 GiB, 8 GiB per buffer |
| Fixed | `-n 50 -w 10` at ≥ 1 GiB, `-n 100 -w 20` below; `-c 0`; one size per process (`-b = -e`) so that sizes rotate |
| Rounds | 5, arms and sizes rotated, all in one session; a second session repeats the 1 GiB and 2 GiB points of all arms |
| Configs | `configs/e1_3/e1_3_<arm>_<calibrated|default>_<size>.cfg` |

### Procedure

1. 9 arms × 2 CTA settings × 8 sizes × 5 rounds = 720 short runs is too many for one session. Run the calibrated setting for all arms and sizes (360 runs, about 50 minutes), and the default setting at 256 MiB, 1 GiB, 4 GiB only (135 runs, about 20 minutes).
2. Per arm and size: median algBW (S1) and busBW (S6) over rounds, with interquartile range.
3. Mixing ratio (G3) = AR-MC busBW ÷ mean(AG-MC, RS-MC busBW), and the same with AR-NVLS; gain over Ring (G4) = AR-MC algBW ÷ AR-Ring algBW; same for AR-NVLS and AR-TMA.
4. Record which algorithm NCCL's own tuner picks at each size (default runs), as a separate column.
5. Second session: repeat 1 GiB and 2 GiB of all arms (calibrated), 5 rounds; report the spread between sessions.

### Privileged access

None.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| Mixing ratio (G3), AR-MC | Below 0.97 if the H100 degradation carries over (H100: 0.865; the pilot at NCCL defaults gave about 0.89 with AR-NVLS). At or above 0.97 = no degradation beyond the plain-mix loss | Case C (no degradation) from A or B |
| Gain over Ring (G4) | Below the byte-count bound 2(N − 1)/N ÷ (1 + 1/N) = 1.20 at 4 GPUs. The pilot gave 0.99 with AR-NVLS | Whether NVLS AllReduce has any use over Ring on GB200 at 4 GPUs |
| AR-TMA and AR-LDST against AR-Ring | Close to Ring: they move the same bytes without multicast | Whether the loss is specific to multicast, or to any concurrent load/store AllReduce |
| Size dependence | Ratios settle above about 256 MiB | The size range where the gate is applied |

## E1.4 Link-counter accounting at the gate size

### Motivation

E1.3 says whether data rate is lost. It cannot say where the time goes. The single most informative number is the raw link rate of the busy directions: if AR-MC runs them at the same raw rate as AG-MC and RS-MC, the link is full and the loss is non-data bytes (case A, as on H100); if lower, the link has idle time (case B). This needs only the driver counters.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Arm**: AG-MC, RS-MC, AR-MC, AR-NVLS, AR-Ring, AR-TMA |
| CTAs × threads/CTA | Calibrated from E1.2 |
| Payload | 2 GiB (and 1 GiB if E1.3 shows the ratios differ between them by more than 2%) |
| Counter runs | Long and short pair per arm (method in the common setup), `-c 0` |
| Rounds | 5, arms rotated, pairs back to back |
| Configs | `configs/e1_4/e1_4_<arm>_<long|short>.cfg` |

### Procedure

1. 6 arms × 2 × 5 rounds = 60 runs, about 15 minutes.
2. Per GPU and direction: data bytes per payload byte (R8, check against the byte model), raw bytes per payload byte (R9), raw link rate (R10), raw ÷ data (G5), ceiling share (G2).
3. Classify into case A, B or C of the [decision table](../PLANS.md#decision-after-plan-01). Case A needs R10 of AR-MC's busy directions within 2% of the mean of AG-MC's received and RS-MC's sent R10. Case B needs it lower by more than 2%.
4. Low-data direction of AG-MC and RS-MC: raw bytes per payload byte, for comparison with H100 (0.956 and 1.118 at 4 GPUs).

### Privileged access

None.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| Raw link rate (R10), busy directions | Equal for AG-MC, RS-MC and AR-MC under case A; lower for AR-MC under case B | Non-data bytes on a full link from idle link time |
| Raw bytes per data byte (G5) | AR-MC above AG-MC and RS-MC under case A (H100: 1.34 against 1.135 and 1.189); about equal under case B | Same |
| Ceiling share (G2) | Equal for all arms if the link is the common limit | A link limit from a per-kernel limit |
| AR-Ring and AR-TMA raw ÷ data | Near AG-MC and RS-MC (no read requests from the switch, few write responses) | Whether non-multicast AllReduce pays the same extra cost |

## E1.5 Reproduction in the role-loop harness

### Motivation

NCCL's kernels differ in more than the mix: channel counts, chunking, synchronization. The H100 explanation was built on the `mm_mix` harness, which runs `multimem.st` and `ld_reduce` role loops shaped like NCCL's symmetric kernels for a fixed time window, on CTAs pinned to chosen SMs. Reproducing the GB200 result in that harness shows that the effect belongs to the mix, not to NCCL's scheduling, and gives plan 02 its instrument.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Arm** (harness roles): `multimem.st` alone (NVLS AllGather); `ld_reduce` alone (NVLS ReduceScatter); both on the same CTAs (NVLS AllReduce); both on separate CTAs; plain store to the next GPU; plain load from the next GPU; plain load + plain store |
| CTAs × threads/CTA | Calibrated first: `multimem.st` alone and `ld_reduce` alone at 1, 2, 4, 8, 16, 32, 64, 76 CTAs × 512 threads/CTA, then 128, 256, 1024 threads/CTA at the CTA count where the 512-thread curve flattens. Same stopping rule as E1.2. The mixed arms use the calibrated count per role; separate CTAs use it for each role (2 × count ≤ 152 SMs) |
| Fixed | 4 GPUs (0–3), 16-byte accesses, warp-contiguous addresses, 2 GiB part per rank, 300 ms windows, 1 warm-up + 10 kept windows (long) and 1 + 2 (short) |
| Rounds | 3 for calibration; 5 for the arms, rotated |
| Configs | `configs/e1_5/e1_5_<arm>_c<ctas>_t<threads>_<long|short>.cfg` |

### Harness port (before E1.5)

| Change | Why |
|---|---|
| GPU list from the config (default stays 4–7 for H100) | GB200 uses 0–3 |
| SM limit from the device (`maxSms` 132 → device property, bounded by a compile-time 256) | GB200 has 152 SMs |
| Target architecture as a Makefile variable (`ARCH=sm_90` default; `sm_100a` for GB200) and binaries in `bin-<arch>/` | One harness, two GPU types, shared home |
| SM-to-GPC maps for the GB200 node, from the existing cluster-placement probe (`h100x2/thread_store/common/scripts/run_gpc_map.py`), per session node | Role placement by GPC; maps differ between GPU types and possibly nodes |
| Runner: GPU list, session context (G1), results on shared storage | Sessions are 2 hours on varying nodes |

Behaviour check after the port: compile `mm_mix` for `sm_90` before and after and compare the SASS (`cuobjdump -sass`); they must be identical, since H100 GPUs are not available here to rerun the recorded arms. On GB200, the correctness count (A11) must be zero for every arm.

### Procedure

1. Port and check as above (code work; GPU time only for a correctness run).
2. Calibration: 2 roles × about 11 settings × 3 rounds ≈ 66 runs (one session with step 3).
3. Arms: 7 arms × long and short × 5 rounds = 70 runs.
4. Per arm: role byte rate (N1), busBW by the link model (N5), R8, R9, R10, G5, and the mixing ratio G3 computed from the harness arms.
5. Compare the harness's case (A, B or C) with E1.4's.

### Privileged access

None.

### Metrics

| Metric | Expected (hypotheses) | Separates |
|---|---|---|
| Mixing ratio from N1 (harness) | Same case as E1.4; ratio within 3% of E1.3's AR-MC | An effect of the mix itself from an effect of NCCL's kernels |
| Plain-mix ratio | Near 1 (H100: 0.97) | Whether any concurrent load/store loses, or only multicast |
| Same CTAs against separate CTAs | Equal (H100: equal) | Whether the loss is inside a CTA (scheduling) or on the link and switch |
| R10 and G5 | As in E1.4 | Same as E1.4, with the harness's exact payload count |

## Run order and time

| Session | Steps | GPU time (estimate) |
|---|---|---|
| 1 | Session context; E1.1 (about 30 runs); E1.2 (about 220 runs) | 70 min |
| 2 | E1.3 calibrated (360 runs) and default (135 runs) | 75 min |
| 3 | E1.4 (60 runs); E1.3 repeat at 1 and 2 GiB (90 runs) | 40 min |
| Code work between sessions | Harness port, SASS check, runner | about 3 hours, no GPU |
| 4 | E1.5 calibration and arms (about 140 runs) | 60 min |

Sessions 2 and 3 should land on different nodes where possible, so that the 1 and 2 GiB values exist from two nodes. Results of every session are reviewed before the next one; a failed check in E1.1 stops the plan.

## What is not observable

The NVL72 switch trays and their load from other jobs in the rack; the packet types on the link (needs admin, see plan 02); the GPU clocks during a call (only before and after each run).
