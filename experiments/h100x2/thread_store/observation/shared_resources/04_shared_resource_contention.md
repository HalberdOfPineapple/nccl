# 04 · Shared-resource contention

The communication operation is unchanged: the **source GPU** reads a local array and pushes its elements into an array on the **destination GPU** using thread-issued peer stores. However, the same operation may perform differently when other work is active. The purpose of these experiments is to test whether controlled pressure degrades communication and determine **which observations expose the interference and how far they help locate its origin**.

This corresponds to the problem-category matrix cell **thread-issued peer stores × shared-resource contention**. The broader category includes competition for issuing-SM resources, source-side memory resources, destination-side memory resources, and shared fabric capacity. The configurations below focus on the agreed cases: communication alone, compute on separate source SMs, source-side memory traffic, and destination-side memory traffic.

These cases distinguish three questions:

- **Compute elsewhere on the source GPU:** does communication change even when its issuing SMs remain dedicated? This establishes whether separate-SM compute produces interference that warrants investigating GPU-wide effects.
- **Memory traffic on the source GPU:** does local memory activity reduce communication progress? The communication kernel must read the source array, so interference can affect data supply as well as other shared source-side resources.
- **Memory traffic on the destination GPU:** do incoming writes slow down when the destination memory system is also serving local work, even though the source's issuing configuration is unchanged?

The intended outcome is a comparison of communication behavior under controlled pressure: overall completed-transfer throughput, per-CTA progress and timing, and relevant memory/device observations. We want to determine whether different pressure locations produce distinguishable signatures. A throughput reduction establishes degradation; additional observations are needed to explain where it arises.

**Scope clarification:** the chosen fused-kernel design puts compute and communication on separate SMs. It therefore does not directly test competition for the communication SMs' registers or issue slots. Such direct competition would require sharing SMs, which is outside the four configurations specified here. Similarly, fabric competition from additional flows remains part of the broader motivation, but is not added as another experiment in this note.

## Common experiment configuration

### GPU role mapping

| Physical device | Semantic role                                                |
| --------------- | ------------------------------------------------------------ |
| GPU 4 (original session); GPU 6 (follow-up) | Source GPU: owns the source array and issues peer stores     |
| GPU 5 (original session); GPU 7 (follow-up) | Destination GPU: owns the destination array and receives the writes |

These are physical device IDs; NCCL ranks are 0 (source) and 1 (destination). Check the allocated devices are idle before execution. All descriptions below use these semantic roles. Keep the same device pair and interconnect topology within each comparison. The new follow-up uses GPUs 6 → 7; do not pair those measurements with the earlier GPU 4 → 5 session.


### Communication kernel

| Field                         | Default configuration                                        | Configuration sweep        |
| ----------------------------- | ------------------------------------------------------------ | -------------------------- |
| Communication interface       | NCCL LSA peer pointer, accessed by ordinary thread-level loads/stores | Fixed                      |
| Direction                     | Source GPU → destination GPU                                 | Fixed                      |
| Array data type               | `int32_t`, 4 bytes per element                               | Fixed                      |
| Array size                    | 256 MiB = 268,435,456 bytes                                  | Fixed initially            |
| Element count                 | 67,108,864                                                   | Fixed initially            |
| Communication CTAs            | 32                                                           | 4, 8, 16, 32, 64           |
| **Threads per communication CTA** | 256                                                          | 64, 128, 256, 512          |
| Per-thread operation          | Read one local element, store it to the matching remote element, then advance | Fixed                      |
| Default traversal             | Split into contiguous CTA regions; stride within each region equals threads per CTA | Compare with global stride |
| Initial instruction policy    | Scalar element operations, without deliberate vectorization or aggressive unrolling | Fixed initially            |

Use device-local source and destination arrays. Initialize them before measurement and verify the destination afterward. Confirm that the selected pair uses the intended NVLink path; NCCL LSA accessibility alone does not identify the physical interconnect.

**Reasoning for the defaults.** A 256 MiB array provides a substantial streaming transfer. Its relation to L2 capacity must be checked on the actual GPU. The 32-CTA, 256-thread configuration is a starting point with eight warps per CTA, not an assertion of optimal bandwidth or fabric saturation. The separate CTA-count and thread-count sweeps show how much the baseline depends on issuing concurrency.

### How threads traverse the array

**Default: CTA-local stride.** Split the array equally into contiguous regions, one per communication CTA. Within its region, thread t accesses offsets `t, t+T, t+2T, ...`, where T is the threads per CTA. With the default configuration, each CTA owns 8 MiB and each thread moves 8,192 elements per pass. This makes CTA ownership and per-CTA timing easy to interpret.

**Comparison: global stride.** Communication thread `(b,t)` starts at `b*T+t` and advances by `B*T`, where B is the communication CTA count. This stride uses communication threads only, excluding co-runner threads. Both schemes access contiguous elements across adjacent warp lanes; the difference is how each CTA advances through the array.

Use CTA-local stride for the interference experiments first. Compare traversal schemes in the communication-only setup before changing the interference configuration.

### Fused co-running arrangement

On a GPU hosting both communication/control work and a co-runner, use a fused kernel with different CTAs assigned different roles. Communication CTAs occupy dedicated SMs; co-runner CTAs occupy other SMs. Use one resident CTA per SM to realize this separation, and verify actual placement. CTA indices do not designate physical SM IDs.

For the source-side co-running experiments, keep 32 communication CTAs and 256 threads per CTA fixed. Use M co-runner CTAs, each with 256 threads, on the remaining SMs. Sweep `M = 0, 8, 16, 32, ...`, including the exact available maximum and excluding counts that do not fit.

A fused launch has one block size. Experiments 2 and 3 retain 256 threads for all CTA roles. The destination follow-up in Experiment 4 tests 256 and 512 threads as separate groups. The current harness also uses the same thread-count setting on both GPUs, so a change from 256 to 512 changes communication, receiver/control, and memory CTA sizes together; it is not a communication-only change. Keep the fused resource allocation and receiver/control setup identical between each co-running condition and its no-co-runner baseline.


### Shared acquisition and comparison protocol

The tables describe intended measurements and hypotheses, not established signatures; implementation and acquisition status are recorded separately in the experiment report. Read the [series design rules](../PLANS.md#rules-for-exploratory-observation-plans). The implemented metric subset is defined in [00_common_setup.md](../00_common_setup.md); observations not yet validated remain candidates. The observation tables below supplement the standard Metrics heading with acquisition details and interpretation limits.

1. **Validate the operation and placement.** Record rank/device UUID, topology, CTA role and physical SM ID, launch resource usage, and useful payload. A one-CTA-per-SM arrangement requires a verified residency constraint; a block count alone does not ensure separation or simultaneous residency. The M=0 fused baseline must retain the same resource reservation as its perturbed partner. Do not add a resident spin-wait scheme that can deadlock when blocks cannot all reside.
2. **Define progress and completion.** In a sparse instrumented build, collect per-communication-CTA start, end of the load/store loop, end of the required ordering operation, and notification publication. Measure receiver polling/observation and acknowledgement separately if the protocol includes them. Ordering must cover writes from all issuing threads before completion is published; one thread's fence alone must not be assumed to order every thread's writes. State the actual NCCL/CUDA ordering and signaling protocol. Use an issuer-local acknowledged interval for completed-transfer timing if available; otherwise name the measured completion boundary precisely. Keep validation reads outside timing.
3. **Measure progress without changing the workload.** Optionally timestamp fixed byte checkpoints per CTA in a diagnostic build, with no extra per-checkpoint fences, to expose gaps and tails. These record software issue progress, not remote arrival. Buffer records locally and export after timing; compare against an uninstrumented build. Report per-CTA median, spread, last-finisher tail, and phase durations. Use device timer units explicitly; do not subtract raw timestamps across GPUs. Adding per-chunk completion acknowledgements would change the protocol and must be a separate diagnostic variant.
4. **Run matched comparisons.** After warm-up, collect at least five interleaved no-pressure/pressure pairs at each level. Calibrate mild, moderate, and strong pressure from achieved co-runner activity; do not assume larger M always causes a larger slowdown. Include co-runner-alone runs with the same resource reservation, and a return-to-baseline run. Save per-run samples, not just averages. Record both GPUs' clocks, power, temperature, and supported clock-limiting reasons before/during/after a repeated-transfer window; keep clocks at the same policy unless a separate intervention is authorized.
5. **Collect in separate passes.** Establish primary timing without profilers, then use sparse device instrumentation, an Nsight Systems timeline with named host ranges for warm-up/measurement/validation, and selected Nsight Compute counters/source views. A fused kernel appears as one kernel in a timeline; prove CTA-role overlap using device start/end records and co-runner work counters. Host ranges do not by themselves delimit device counter collection. Select a repeatable steady transfer window and document whether each counter covers it or the whole fused launch, including the co-runner tail.
6. **Preserve overlap during counter collection.** Query available metrics/sections on the installed GPU/tool version (for example `ncu --query-metrics` and `ncu --list-sections`); save the resolved names, units, scope, replay mode, and cache/clock settings. Memory Workload Analysis, Scheduler Statistics, Warp State Statistics, and source/PC sampling are candidate views where supported. Kernel replay may alter cache state or serialize dependent work. Evaluate range/application-range or supported multi-process profiling for the actual launch structure, and verify peer progress and overlap for every accepted pass. Do not assume a replay mode supports this multi-GPU protocol merely because it supports concurrent kernels. If collection changes or hangs the interaction, omit that pass and record the limitation. Profile each endpoint in separate matched runs if simultaneous capture is unsupported.
7. **Respect aggregation boundaries.** Fused-kernel SM/memory counters mix communication and co-runner roles. Use source/SASS PC attribution for communication loads, peer stores, and ordering instructions when available, keeping those paths distinct in generated code. Device L2/DRAM and supported NVLink byte/rate counters are endpoint/window evidence, not per-operation attribution or direct queue occupancy. Counter deltas include background work; record idle/background windows and link direction. Normalize bytes to useful communication payload and rates to the documented counter window. Missing access is unavailable evidence, not a zero.

**Access and fallback.** Ordinary-user device timers, work counters, and correctness checks are the minimum path. GPU allocations are required before execution; device telemetry, NVLink counters, and GPU performance counters depend on platform permissions and tool support. If unavailable, retain phase/progress observations and achieved co-runner rates and state which endpoint/mechanism distinctions remain unresolved. Clock locking is optional, requires authorization, and is not a prerequisite for the plan.

**Interpretation boundary.** An issue-loop timer includes source loads, instruction execution, and backpressure. An ordering interval is a protocol cost, not a direct measurement of wire latency or HBM service time; notification/polling can add further delay. A longer issue interval and a longer ordering interval are both possible under endpoint pressure. No single stall reason or utilization percentage uniquely identifies contention.

Tool semantics: [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) describes replay and counter-attribution limits; [CUDA memory-fence documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html) distinguishes ordering from visibility/synchronization. Check installed-version support before selecting concrete passes.

## Experiment 1: communication alone (healthy baseline)

### Motivation

Establish the reference performance of the peer-store copy before introducing contention. This is necessary because communication can already be limited by insufficient issuing concurrency: a small CTA or thread count may not generate enough outstanding work. Without a baseline sweep, a later slowdown or lack of slowdown is difficult to interpret.

The experiment also establishes normal per-CTA timing variation and the performance difference between the two array traversal schemes.

### Configuration

Use the common communication workload with no active compute or memory co-runner. Start with 32 CTAs, 256 threads per CTA, a 256 MiB array, and CTA-local stride.

| **Independent variable (one sweep at a time)** | Complete geometry / traversal                   | Values held fixed                      |
| ----------------------------- | ------------------------------- | -------------------------------------- |
| **Communication CTA count**       | **4** × 256; **8** × 256; **16** × 256; **32** × 256; **64** × 256 (CTAs × threads/CTA) | 256 threads, 256 MiB, CTA-local stride |
| **Threads per communication CTA** | 32 × **64**; 32 × **128**; 32 × **256**; 32 × **512** (CTAs × threads/CTA) | 32 CTAs, 256 MiB, CTA-local stride     |
| **Communication CTA count, repeated at 512 threads/CTA** | **16** × 512; **32** × 512; **64** × 512 (CTAs × threads/CTA) | 512 threads, 256 MiB, CTA-local stride |
| **Traversal**                     | 32 × 256 (CTAs × threads/CTA) with CTA-local stride; 32 × 256 with global stride | 32 CTAs, 256 threads, 256 MiB          |

### Procedure

1. Allocate and initialize the arrays, obtain the peer pointer, and run untimed warm-up transfers.
2. Measure repeated transfers with the default configuration and verify the destination after timing.
3. Sweep CTA count, then thread count, changing one parameter at a time. Also test 16, 32, and 64 communication CTAs at 512 threads/CTA. Use the same compiled program that contains the vector-read co-runner, with the co-runner disabled. Repeat each setting in five rounds, with three warm-ups and twelve retained transfers per invocation. Recheck SM placement and one-CTA-per-SM occupancy when launch resources change. Save the binary hash and compiler resource report.
4. Compare traversal schemes with the default CTA/thread counts.
5. Collect the shared sparse-instrumentation, timeline, and selected counter passes for representative low-concurrency and default cases; establish the observations below as reference distributions.

### Privileged access

Use the shared access/fallback policy above; no clock changes are required. Counter-dependent observations are optional if permissions are absent.

### Observations and interpretation

| Observation beyond completed-transfer throughput | Method, endpoint, and window | Question / expected signature (hypothesis) | Interpretation limit |
|---|---|---|---|
| Per-CTA issue, ordering, and notification timing; completion tail | Sparse device timestamps on source, receiver-local wait duration on destination, over each transfer; retain bytes and SM IDs | Does increasing issuing concurrency shorten the issue phase or change which phase dominates? Does a small number of CTAs finish late? Establish the reference distribution for all later cases. | The fastest tested configuration is a reference, not proof of fabric saturation. A short final ordering interval can coexist with backpressure during issue. |
| Instruction supply and communication memory dependencies | Source Nsight Compute scheduler/warp-state views; inspect samples at source-load, peer-store, and ordering PCs separately | Does the low-concurrency baseline lack eligible issuing warps, or show memory-related waiting? Check generated access width and actual loads/stores against the intended scalar/coalesced workload. | Eligible-warp and stall metrics alone cannot identify a physical bottleneck; no peer-store latency is directly measured. |
| Source/destination memory demand and link activity | Supported endpoint L2/DRAM bytes/rates, cache behavior, and directional NVLink deltas over a repeated-transfer window, with background snapshots | How do payload-normalized transactions and rates vary with geometry/traversal? Establish whether repeat transfers have different cache behavior despite identical useful bytes. | Incoming writes need not correspond one-to-one with immediate destination HBM writes; aggregate counters include other traffic. |
| Measurement validity and normal operating state | Correctness check after timing; SM/CTA records; Nsight Systems for launch/control gaps; both endpoints' telemetry | Are traversal comparisons changing work assignment rather than bytes or completion semantics? Are outliers associated with launch gaps, clocks, or placement? | Traces and telemetry can miss sub-transfer events; determine their sampling resolution before making a time-local claim. |

Check whether throughput stops improving as issuing threads increase, using repeated-run variation to judge small differences. A flat result is a limit of the tested program, not proof of HBM or NVLink saturation. If throughput still rises at 64 CTAs × 512 threads/CTA, report that the current harness limits were reached without finding a flat region; do not label that setting saturated. Experiments 2 and 3 retain their original geometry. Experiment 4 uses the explicitly listed geometries below, each with its own no-pressure reference. Save both unprofiled and instrumented references so a probe-induced effect cannot be mistaken for interference. Plot phase distributions and per-CTA timelines against geometry/traversal, not throughput alone.


## Required correction: finish communication calibration before more contention runs

**Revision, 2026-10-27; [execution and selected reference](results/2026-10-27-saturation-gpu6-7/README.md).** The preceding follow-up stopped at a software limit while communication throughput was still increasing. Completing that grid did not satisfy the intended calibration. Do not start another destination-pressure grid until communication-alone measurements establish where more issuing threads stop improving throughput. Retain the earlier results as measurements of the tested configurations, not a saturated reference.

| Knob | Role | Planned setting |
|---|---|---|
| Devices | Constant | Physical GPU 6 → GPU 7; inspect availability before each invocation |
| **Communication CTA count C** | **Independent variable at fixed T** | **64, 96, 128, 132**, subject to actual one-CTA-per-SM occupancy |
| **Threads per CTA T** | **Independent variable at fixed C** | **512, 768, 896, 1024**; prioritize the 128-CTA comparison to keep CTA region boundaries unchanged |
| Payload, traversal, store width | Constant for the initial calibration | 256 MiB, CTA-local stride, scalar 4-byte loads/stores |
| Destination reads | Constant | Disabled: 0 CTAs × T threads/CTA |
| Destination control | Constant within each thread group | 1 CTA × T threads/CTA; one polling lane |
| Completion and validation | Constant | Existing all-writer fence, publication, and acknowledgement protocol; full-buffer verification after each transfer |
| Repetitions | Calibration then confirmation | Initial screen, followed by at least five repeated rounds around the candidate limit; rotate order and retain individual transfers |
| Program | Constant within a comparison | Newly built binary with expanded bounds; fresh references; preserve the previous binary/source locally |

Use the screen to choose confirmation settings, not to declare saturation from one run. At fixed 128 CTAs, increase threads from 512 to 768 to 1024; also compare CTA counts at fixed thread count. As an operational criterion, seek two successive increases in issuing threads with less than 1% median throughput gain, then report paired changes and run variation. The 1% criterion is a practical calibration tolerance, not a statistical significance test or proof of a specific physical bottleneck. If throughput falls rather than flattening, report the downturn and investigate it instead of calling it a flat plateau.

The transfer-size follow-up changes **payload size** at fixed communication geometry: **256, 512, 1024, 2048 MiB**, with **4096 MiB** available for a further confirmation point. Destination reads remain disabled. At the selected longer size, repeat the thread-count comparison rather than assuming a longer transfer creates more requests at once. The `payload_mib` setting changes host-side configuration only; check that the generated GPU instructions match the preceding concurrency build, and record a separate binary hash and fresh controls.

The next calibration check varies **independent scalar requests per thread (`copy_batch`) = 1, 2, 4, 8** at fixed CTA count, thread count, and payload. Each thread loads that many separate source elements into registers, then issues the matching 4-byte stores; no vector stores or extra fences are added. This changes request supply and compiler output, so build a separate revision and remeasure every compared setting in that binary. Inspect registers/spills and verify the full destination. At 128 CTAs, confirm both the 512/768/1024-thread comparison with batch 8 and the 1/2/4/8-batch comparison with 1024 threads. Use a 2048 MiB payload and a separate 4096 MiB check for remaining short-transfer effects.

The 132-CTA setting tests the available SM limit. Its CTA regions are not aligned in exactly the same way as the power-of-two CTA settings; use the fixed-128-CTA thread comparison as the cleaner confirmation. Inspect source issue/checkpoint, ordering, acknowledgement, and tail times alongside throughput. Keep the per-role block-size and compiler register/shared-memory changes explicit.

If the curve is still rising at the expanded limits, investigate remaining request-generation constraints and longer transfer windows before proceeding. A larger transfer-size check, when added, must keep CTA/thread geometry fixed and be treated as a separate comparison. Only after this calibration is resolved should destination read pressure be calibrated and combined with the selected communication setting. All conclusions and execution status belong in the dated report.

## Experiment 2: compute on the issuing GPU

### Motivation

Determine whether compute activity elsewhere on the source GPU changes communication performance while the communication CTAs retain their dedicated SMs. This models a fused workload with spatially separated communication and compute roles.

The compute workload should generate substantial arithmetic activity with minimal bulk memory traffic. That makes it easier to distinguish this case from the source-memory-pressure experiment. Any slowdown here should be investigated as a GPU-wide effect or unexpected resource sharing; it cannot be attributed directly to sharing the communication SMs' issue slots or registers.

### Configuration

| Field                        | Choice                                                       |
| ---------------------------- | ------------------------------------------------------------ |
| Co-runner placement          | Source GPU, on SMs separate from communication               |
| Kernel organization          | One fused kernel with communication and compute CTA roles    |
| Communication                | 32 CTAs × 256 threads; default array and traversal           |
| Compute operation            | Repeated FP32 fused multiply-add operations                  |
| Per-thread state             | Eight independent register accumulators                      |
| **Compute CTA count (independent variable M)** | **M** co-runner CTAs × 256 threads/CTA; M swept over the available remaining SMs |
| Threads per compute CTA      | 256                                                          |
| Compute-loop memory activity | No intentional bulk loads/stores; initialize before and write checksum after |

Eight independent accumulators provide arithmetic independence rather than a single dependency chain. Inspect generated code to ensure the arithmetic remains and that register spilling does not turn the intended compute workload into a memory workload. All other table entries and shared settings are held constant while M varies. The achieved compute rate is a measured response, not a controlled knob, and must be measured rather than assumed to be peak.

### Procedure

1. Run the fused configuration with M=0 to establish its matched communication baseline.
2. Enable a small number of compute CTAs, keeping the communication geometry and array unchanged.
3. Start compute before the measured communication phase and keep it active throughout the transfer.
4. Increase M through the agreed sweep, measuring communication and achieved compute throughput at each point.
5. Repeat the no-co-runner reference between conditions to identify changes in the GPU's operating state.
6. After primary paired timing, collect the observation passes below for M=0 and calibrated mild/moderate/strong achieved compute activity, verifying role overlap and placement in each pass.

### Privileged access

Use the shared access/fallback policy. Counter permission is needed for arithmetic/source-PC profiling; absent it, use checksum/work counters, phase timers, and available telemetry. Do not lock clocks without separate authorization.

### Observations and interpretation

| Observation beyond completed-transfer throughput | Method, endpoint, and window | Question / expected signature (hypothesis) | Interpretation limit |
|---|---|---|---|
| Verified compute activity and spatial/temporal separation | Source per-role SM IDs and active intervals; co-runner FMA iteration counts/checksum over the measured overlap; SASS and spill inspection | Is arithmetic active throughout communication on distinct SMs, and does achieved arithmetic work increase with M? Reject a nominal compute-only case if spills create material memory traffic. | Whole-fused-kernel arithmetic utilization cannot establish activity on communication SMs or overlap by itself. |
| Communication issue, ordering, and completion shape | Shared per-CTA phase/checkpoint probes and receiver wait observations, paired against identical M=0 launch resources | Is any slowdown broad across communication CTAs or concentrated in a tail? Does issue slow while destination observation/notification adds little extra delay? No change is a valid outcome. | A phase signature may resemble source memory or endpoint backpressure; it is evidence to combine, not a compute-contention label. |
| GPU operating-state changes | Source and destination clocks, power, temperature, supported limiting reasons during repeated windows, paired with phase durations in time units and optionally cycles | Does source compute correlate with a different clock/power regime? A broad issue slowdown with clock change motivates investigating a GPU-wide operating-state effect. | Separate SMs exclude intended direct SM sharing, not GPU-wide effects. Telemetry association does not prove power limiting caused the slowdown. |
| Unexpected memory pressure and communication-PC stalls | Source profiler memory view and source-PC samples, plus destination aggregate memory/link observations; compare M=0, compute alone, and mixed runs | Does a nominal arithmetic workload add L2/DRAM demand or change waiting at communication load/store PCs? Contrast with the source-memory experiment. | Aggregate demand combines roles; memory traffic may come from initialization/checksum or spills outside the transfer window. Do not assign aggregate stall percentages to communication CTAs. |

Plot communication phase distributions and achieved compute rate versus M, with operating-state observations in separate aligned plots. If communication is unchanged, report the tested achieved compute range and noise floor. If it degrades, report whether observations separate a clock-associated change from unexpected memory traffic; otherwise keep the mechanism unresolved. A separate authorized fixed-clock test could examine that hypothesis later.


## Experiment 3: memory traffic on the issuing GPU

### Motivation

Determine whether local memory activity on the source GPU slows the push despite communication having dedicated SMs. The two roles still access shared GPU memory resources, and the communication kernel needs local source loads before issuing its remote stores.

Use a synthetic streaming workload because its working set, access pattern, concurrency, and pacing are explicit. Attention decoding would introduce additional computation and dependencies that make the generated memory pressure harder to control.

### Configuration

> **Revision 2026-10-28:** new runs use the [rate-controlled memory co-runner](#memory-pressure-with-the-rate-controlled-co-runner-revision-2026-10-28): 24 pressure CTAs, target read rate 20/40/60/80% of HBM peak. The text below is kept as the record of the earlier plan (its calibration phases ran on 2026-10-27 to 2026-10-28; see the revision's last subsection).

**Revised source-memory plan; not yet executed.** First increase the read workload's achieved bandwidth, then sweep those calibrated configurations with communication. The earlier 8/32/100-CTA sampling was not a bandwidth calibration and must not define the new pressure ladder. Include 64 explicitly. More configured CTAs or threads are not themselves evidence of stronger memory traffic.

The fused kernel reserves one SM per CTA. On the 132-SM devices, source communication C and source reads M must satisfy **C + M ≤ 132**. Calibrate the read workload across the available range, then determine the strongest feasible pressure for each fixed communication budget. A configuration with 128 communication CTAs cannot coexist with more than four source read CTAs under this policy.

| Knob | Role | Planned setting |
|---|---|---|
| Devices and direction | Constant | Physical GPU 6 issues stores to GPU 7; inspect allocation/availability before execution |
| **Source read CTAs M** | **Independent variable at fixed read threads and fixed communication geometry** | **1, 2, 4, 8, 16, 32, 64, 96, 100, 128, 131**, subject to the row's source-SM budget; include exact remaining-SM endpoints where useful |
| **Read threads per CTA Tm** | **Independent variable at fixed M and communication geometry** | **128, 256, 512, 768, 1024**; report each setting as M × Tm |
| Communication payload / accesses | Constant in contention comparisons | 2048 MiB; batch 8 scalar 4-byte source loads and peer stores; CTA-local stride; existing all-writer fence/publication/acknowledgement protocol |
| **Communication threads per CTA Tc** | **Independent variable only during communication calibration** | **256, 512, 768, 1024**, at each fixed C; select and freeze Tc before that row's read-pressure sweep |
| Communication CTA budget C | Separate comparison groups | 1, 32, 64, 128 CTAs; calibrate communication separately within each group; do not pool their throughput or interpret changes between groups as read-only effects |
| Source read implementation | Initial constant | Coalesced 16-byte vector loads; four checksum chains, four-way unrolling; 256 KiB logical bytes per burst; pacing 0 |
| Read working set | Constant | Disjoint 1 GiB array, partitioned into contiguous read-CTA regions; verify generated loads/checksums and that the working set exceeds L2 capacity |
| Destination | Constant within a communication group | One receiver/control CTA × Tc threads/CTA; no destination co-runner |
| Standalone read controls | Separate calibration arm | Communication disabled; initially source timer control 1 × Tc plus M × Tm reads, then repeat with the exact C × Tc controls of each feasible contention group |
| Sampling | Constant for confirmation | At least five rotated invocations per retained calibration setting; standalone calibration starts with two warm-ups and five retained 20 ms windows; contention uses five adjacent copy-only / copy-with-reads pairs, three warm-ups and twelve retained transfers per invocation |

**Implementation prerequisite:** the current fused kernel uses one shared `threads` setting for both roles. Independent Tc and Tm are not implemented. Add independently configurable role thread counts before this plan is executed, preserving the semantics of each workload and explicitly checking occupancy, synchronization, checksums, and full-payload correctness. If implemented through a common block size with inactive lanes, record both physical block size and active role threads; inactive lanes still reserve resources. Do not present a shared-thread sweep as changing only the memory workload. Any changed GPU binary requires fresh communication and read calibration, including a behavior check against the existing harness.

| Fixed communication group | **Source read CTA values to sweep**, each at the calibrated Tm values | Source residency limit | Purpose |
|---|---|---|---|
| 1 × selected Tc | **1, 2, 4, 8, 16, 32, 64, 96, 100, 128, 131 × Tm** | 1 + M ≤ 132 | Carry the strongest read configurations into a concurrent test; communication may be request-limited |
| 32 × selected Tc | **1, 2, 4, 8, 16, 32, 64, 96, 100 × Tm** | 32 + M ≤ 132 | Increase source read bandwidth while holding a larger communication allocation fixed |
| 64 × selected Tc | **1, 2, 4, 8, 16, 32, 64, 68 × Tm** | 64 + M ≤ 132 | Test stronger communication demand with the maximum reads that fit this allocation |
| 128 × selected Tc | **1, 2, 4 × Tm** | 128 + M ≤ 132 | Check near the previously selected communication geometry; explicitly limited read-SM budget |

Every group also has its own **M = 0** copy-only reference with identical C, Tc, payload, batch, binary, and reservation policy. “Selected Tc” is a calibration output, not a missing constant to choose during a pressure comparison. Save the complete values in configs and result tables before execution of the paired sweep.

### Procedure

1. **Calibrate read bandwidth before choosing pressure levels.** With communication disabled, sweep M at each fixed Tm and Tm at each fixed M using the table above. Record logical read bytes, actual duration, per-CTA work and placement, and repeat each retained point. Plot achieved read GB/s from zero against the complete M × Tm configurations; keep separate curves for fixed Tm. The objective is to progressively raise achieved bandwidth, not merely to finish a predefined CTA list.
2. **Continue until the read workload's limit is characterized.** If achieved rate is still rising at the last feasible point, investigate request supply rather than declaring saturation. After the CTA/thread sweeps, test stronger independent-load supply (for example read-loop unroll 4, 8, 16 at fixed M × Tm, working set and burst size), checking generated loads, registers/spills and occupancy. These variants require implementation and fresh same-binary controls. Confirm candidate flat regions over five rounds; use less than 1% gain over two successive feasible request-supply increases, with the three medians spanning less than 1%, as an operational tolerance. A downturn requires investigation. If no plateau is established before a real resource limit, report that limit and the strongest achieved rate; do not call it memory saturation. Supported, validated physical DRAM-byte counters are needed to claim HBM bandwidth rather than logical kernel bandwidth.
3. **Calibrate communication within each fixed source-SM budget.** At C = 1, 32, 64, 128 with reads disabled, increase Tc through the listed values using the new binary. Preserve payload/batch/traversal and plot the full curves. Select Tc per group using the repeated measurements, retaining a request-limited label wherever the curve is still rising. The previously measured 128 × 512 result cannot be assumed to apply to a changed binary or a smaller C. If both maximal source read demand and the communication reference cannot fit simultaneously, state that experimental constraint; do not silently reduce one and call both saturated.
4. **Recheck reads with the exact control geometry.** For every feasible C × Tc group, repeat the retained read configurations with communication disabled but the same source controls, physical block size and reservations used by the combined kernel. Preserve low, intermediate and highest achieved bandwidth points, including **64 × Tm wherever feasible**. Do not discard a flat or nonmonotonic point. Only compare M effects at fixed Tm; changing M and Tm together is a complete-configuration comparison, not a one-knob sweep.
5. **Sweep the calibrated read configurations with communication.** Run at least five adjacent copy-only / copy-with-reads pairs per retained configuration, rotate configuration order, and finish each group with a return-to-reference run. Keep C × Tc fixed within each group. Start reads before communication and stop after completion; verify every read CTA covers the source communication interval and completes bursts. Report communication throughput/duration, paired changes and run variation first, plotted against both configured M × Tm and independently calibrated read rate. Also report the actual concurrent read rate: communication can change the pressure achieved.
6. **Measure the reciprocal effect with duration-matched controls.** For the highest achieved pressure in each group and any lower setting showing communication degradation, run fresh reads-only / reads-with-copy pairs with the same geometry and reservations. Set the solo timer from the observed concurrent active duration and report actual durations. These controls measure read slowdown, not a communication-throughput reference.
7. **Collect observations after primary timing.** Check minimal/detailed probe overhead and separate endpoint timelines at no reads, an intermediate setting including M = 64 where feasible, and highest achieved pressure. Attempt source memory/PC counters only through a validated concurrent capture method; reject captures that remove overlap. Optional pacing is a later, separate reduction of already-calibrated pressure, not the method for finding maximum bandwidth.

### Privileged access

Use the shared access/fallback policy. Source memory/source-PC counters require supported performance-counter access; achieved read work and phase timing remain available without them.

### Observations and interpretation

| Observation beyond completed-transfer throughput | Method, endpoint, and window | Question / expected signature (hypothesis) | Interpretation limit |
|---|---|---|---|
| Achieved source-memory pressure and overlap | Source completed bursts × 256 KiB divided by union active duration, checksum, SM-role records, plus separately validated L2/DRAM traffic; compare co-runner alone and mixed runs | Do M and pacing produce distinct actual read rates? Does communication also reduce co-runner progress, indicating mutual interference? | Logical read bytes are not HBM bytes: cache reuse and compiler transformations matter. M and P are injection controls, not evidence of the achieved pressure. |
| Supply-side instruction waiting | Source-PC sampling around communication source loads and their dependent instructions; relevant cache/DRAM observations on source | Does increased source memory demand coincide with increased waiting in the communication load path? This supports a data-supply explanation more than throughput alone. | Sampled long-scoreboard or related memory waits are not a direct source-HBM queue metric, and mixed role aggregates cannot identify the affected path. |
| Issue versus store/ordering backpressure | Per-CTA issue/checkpoint/ordering durations on source; receiver-local polling/observation; samples at peer-store and ordering PCs where supported | Does pressure lengthen issue progress, completion-protocol cost, or both? Compare the destination-memory case at a similar communication slowdown. | Both endpoints can induce store-path backpressure during issue; an unchanged final ordering interval does not rule it out. |
| Endpoint asymmetry and operating state | L2/DRAM demand on both GPUs, directional link observations, and both endpoints' clocks/power over matched windows | Does source-local demand rise while destination demand attributable to the copy follows reduced incoming throughput? Does pacing change the operating state as well as request rate? | Pacing arithmetic itself adds compute. Lower link throughput can be the consequence of source starvation; it is not evidence that the link degraded. |

Plot communication phase changes against achieved source read rate, retaining M and P as configuration annotations. Compare source-load-PC evidence with destination-memory pressure and separate-SM compute at overlapping slowdown ranges where attainable; do not force matching by changing the communication geometry. Without role-specific stall evidence, classify source-memory pressure as known injection context while leaving the precise shared resource unresolved. Co-runner-alone counter subtraction is only an approximate comparison, not an isolation of communication traffic.

## Experiment 4: memory traffic on the destination GPU

### Motivation

Determine whether local memory activity on the destination GPU slows incoming peer stores while the source GPU's communication configuration remains unchanged. This tests interference at the receiving endpoint without intentionally adding source-side pressure.

Use the same memory workload as Experiment 3 so its placement is the main conceptual difference. The destination's local reads and the incoming writes access shared endpoint memory resources, although the precise bottleneck must be established from observations rather than assumed.

### Configuration

> **Revision 2026-10-28:** new runs use the [rate-controlled memory co-runner](#memory-pressure-with-the-rate-controlled-co-runner-revision-2026-10-28): 24 pressure CTAs, target read rate 20/40/60/80% of HBM peak. The text below is kept as the record of the earlier plan and its executed runs.

**Follow-up plan, 2026-10-27 — executed on physical GPUs 6 → 7; see the [dated report](results/2026-10-27-gpu6-7/README.md).** First measure communication alone, then destination reads alone, then run both together. The main question is whether increasing the number of issuing threads makes communication more sensitive to destination reads. Earlier measurements remain in [the report](README.md); do not reuse an earlier compiled program's measurements as new matched controls.

| Knob | Role | Planned setting |
|---|---|---|
| Device pair and direction | Constant | Physical GPU 6 (source, rank 0) → GPU 7 (destination, rank 1); check availability before running |
| Communication payload and accesses | Constant | 256 MiB; scalar 4-byte source loads and peer stores; CTA-local stride; same completion/acknowledgement protocol as the common setup |
| **Communication CTA count C** | **Independent variable when testing communication demand** | **16, 32, 64**; keep threads/CTA and destination read configuration fixed within each comparison |
| **Threads per CTA T** | **Separate test groups** | **256 or 512** for every CTA on both GPUs; compare C and M within each group first |
| **Destination read CTA count M** | **Independent variable when testing memory pressure** | **0, 8, 32, 100, 131**; keep communication C × T fixed within each comparison |
| Destination read operation | Constant | Coalesced 16-byte vector reads; four checksum chains; four-way loop unrolling; 256 KiB per CTA burst; `vector_reads=1` |
| Destination read buffer | Constant | Separate 1 GiB allocation; one contiguous slice per read CTA; wrap within the slice; no overlap with incoming payload |
| Read pacing | Constant | `pacing=0`; no added arithmetic rounds between bursts |
| Source co-runner | Constant | None |
| Destination control | Constant within each T group | One receiver/control CTA × T threads; one polling lane; same reservation with and without reads |
| Placement | Constant policy | One resident CTA per SM, verified for each geometry; at most 64 source CTAs and 132 destination CTAs |
| Program and instrumentation | Constant | Same compiled program and `detailed=1` for primary comparisons; record its hash and resource use; profile in separate runs |
| Primary run sampling | Constant | Five interleaved reference/pressure pairs per condition; three warm-ups and twelve retained transfers per invocation |
| Standalone read window | Calibration setting | Initially 20 ms; two warm-ups and five retained windows per invocation; repeat five invocations per read geometry |

The destination read workload is read-only apart from control/checksum output. A write or mixed read/write workload would need new harness code and its own calibration; it is not part of this first follow-up. The planned settings fit the current parser's limits of 64 communication CTAs and 512 threads/CTA. Independent source and destination thread counts are not currently supported.

### Procedure

#### Step 1: communication alone — increase issuing threads

Run the expanded Experiment 1 with the current compiled program. The following six settings are then carried into the combined test; transfer size remains 256 MiB.

| Communication CTAs × threads/CTA | Destination reads | Destination control CTAs × threads/CTA | Comparison |
|---|---|---|---|
| 16 × 256 | 0 × 256 (inactive) | 1 × 256 | Vary C within the 256-thread group |
| 32 × 256 | 0 × 256 (inactive) | 1 × 256 | Vary C within the 256-thread group |
| 64 × 256 | 0 × 256 (inactive) | 1 × 256 | Vary C within the 256-thread group |
| 16 × 512 | 0 × 512 (inactive) | 1 × 512 | Vary C within the 512-thread group |
| 32 × 512 | 0 × 512 (inactive) | 1 × 512 | Vary C within the 512-thread group |
| 64 × 512 | 0 × 512 (inactive) | 1 × 512 | Vary C within the 512-thread group |

Check whether adding issuing threads still improves completed-transfer throughput. Keep all six settings for Step 3, including slower settings: they show whether sensitivity to destination work changes with communication demand. Do not increase payload size or change store width during this comparison.

#### Step 2: destination reads alone — measure the pressure the workload can generate

Disable communication (`communication=0`, `kind=3`). Keep the destination control CTA and resource reservation. The source's configured control CTAs execute the standalone timer, not a copy; record their geometry and report no communication throughput for this arm.

| Group | Source control CTAs × threads/CTA (copy disabled) | **Destination read geometries tested separately** | Destination control CTAs × threads/CTA |
|---|---|---|---|
| 256 threads/CTA | 32 × 256 | **8 × 256; 32 × 256; 100 × 256; 131 × 256** | 1 × 256 |
| 512 threads/CTA | 32 × 512 | **8 × 512; 32 × 512; 100 × 512; 131 × 512** | 1 × 512 |

For each setting, measure completed read bursts, actual active duration, and logical read GB/s. Check whether increasing M continues to increase the rate. Compare the two thread groups separately; using every available SM does not prove that memory bandwidth is fully used. Inspect generated loads and register spills. If the rate still rises at the largest setting, record that the tested range has not established a limit. If it stops rising, keep open both memory-service and instruction/loop-overhead explanations until counters distinguish them.

#### Step 3: communication and destination reads together

For each row below, test every listed destination geometry. Within a row, **M is the independent variable**; communication geometry stays fixed. Between rows with the same T and M, **C is the independent variable**. Comparing the two T groups changes both workloads' thread counts and must be described that way.

| Fixed communication CTAs × threads/CTA | Matched copy-only destination reads | **Destination read geometries, each a separate pressure condition** |
|---|---|---|
| 16 × 256 | 0 × 256 (inactive) | **8 × 256; 32 × 256; 100 × 256; 131 × 256** |
| 32 × 256 | 0 × 256 (inactive) | **8 × 256; 32 × 256; 100 × 256; 131 × 256** |
| 64 × 256 | 0 × 256 (inactive) | **8 × 256; 32 × 256; 100 × 256; 131 × 256** |
| 16 × 512 | 0 × 512 (inactive) | **8 × 512; 32 × 512; 100 × 512; 131 × 512** |
| 32 × 512 | 0 × 512 (inactive) | **8 × 512; 32 × 512; 100 × 512; 131 × 512** |
| 64 × 512 | 0 × 512 (inactive) | **8 × 512; 32 × 512; 100 × 512; 131 × 512** |

1. Run five interleaved copy-only / copy-with-reads pairs for each of the 24 pressure conditions. Each reference uses the same C, T, compiled program, payload, instrumentation, and reservations as its pressure partner. Step 1 is calibration, not a substitute for these nearby references. Rotate condition order between rounds and finish with a return-to-reference run.
2. Start destination reads before the transfer and stop them after receiver acknowledgement. Verify each read CTA's active interval covers the destination control interval; retain work counts as well as interval coverage. Coverage alone does not prove uniform request activity.
3. Present communication throughput and completed duration first, including per-invocation medians, paired changes, and variability. Also inspect individual-transfer distributions and slow transfers; do not let medians hide occasional delays.
4. Present destination read performance second. For the strongest read setting in each row, and any lower setting showing a communication slowdown, run fresh interleaved reads-only / reads-with-copy pairs. Match source control geometry, destination geometry, and reservations. Set the reads-only duration to that condition's measured median concurrent read-active duration, then compare actual durations. Use five pairs with three warm-ups and twelve retained windows per invocation. A copy-disabled arm has no communication-throughput baseline.
5. Collect the separate observation passes below for a low-thread reference, the fastest tested communication setting, and any setting with a repeatable slowdown. Preserve overlap and correctness during profiling. If counters remain unavailable, report that physical memory traffic and its limiting resource are unresolved.

A slowdown should be judged against variation across the matched invocations, with absolute and percentage changes reported together. Do not label a small change meaningful solely because the reference IQR is tiny. If no communication setting slows, report the tested configurations and achieved read rates; do not claim that destination memory traffic cannot interfere with communication.

### Privileged access

Use the shared access/fallback policy. Destination L2/DRAM/link profiling requires supported access; without it, retain destination co-runner work counts, receiver timing, and source progress, and state the reduced localization confidence.

### Observations and interpretation

Read the observations in this order. These are questions to test, not predicted results.

| What to check | How to measure it | What the comparison can tell us |
|---|---|---|
| Communication throughput first | Useful payload / acknowledged source duration; compare each C × T setting with its own no-read reference | Does destination work slow the transfer, and does that change when more threads issue stores? |
| Where communication time changes | Existing source CTA start, issue-end, fence-end, publish, and acknowledgement timestamps; eight issue checkpoints in detailed mode | Does extra time appear during copying, final ordering, or waiting for acknowledgement? Preserve per-CTA spread; phase medians do not add up to the full transfer duration. |
| Whether reads really make progress | Destination completed burst counts × 256 KiB / recorded active window; retain per-CTA counts and start/end times | Is pressure actually generated? Does communication reduce destination read progress? The current rate includes startup and tail; it is not a read rate measured only during the transfer. |
| Whether the workloads overlap | Existing destination-local active intervals and SM IDs, plus a separate timeline capture | Are read CTAs active throughout receiver waiting on separate SMs? This does not show constant memory traffic within that interval. |
| Whether physical memory traffic explains the change | Separate, validated destination L2/DRAM read/write counter pass; source comparison where supported; document the counter window | Does the logical read rate correspond to physical memory traffic? If the profiler interrupts peer progress or changes overlap, reject the pass. |

Existing device probes are collected per transfer without profiler replay; their overhead must be checked with minimal/detailed pairs at representative low and high thread counts. Hardware counters and timelines cost separate runs. A finer record of read work during the transfer would require additional probes and an overhead check; the existing active-window rate must not be relabeled as that measurement.

| Observation beyond completed-transfer throughput | Method, endpoint, and window | Question / expected signature (hypothesis) | Interpretation limit |
|---|---|---|---|
| Achieved destination pressure and overlap | Destination read-byte/work counters, checksum, per-role SM IDs and active intervals; destination L2/DRAM observations over the transfer window | Is the same streaming workload active at the receiving endpoint throughout communication, and does incoming communication reduce its progress relative to co-runner alone? | SM separation from receiver/control work does not isolate shared endpoint memory resources. Logical read rate is not physical HBM rate. |
| Reflected issuer backpressure | Source per-CTA issue/checkpoint/ordering durations and communication-PC sampling; source co-runner absent | Can destination pressure be exposed from source progress alone? Longer peer-store-related waiting, issue time, or ordering time is a candidate signature, even though issuing geometry is fixed. | Such symptoms may also occur under source-memory pressure; sender-only timing cannot uniquely locate the contention. |
| Receiving memory state versus source data supply | Matched source/destination L2/DRAM traffic/cache views and supported directional link counters, compared with the source-memory experiment | Does added local demand appear at destination while communication source-load behavior remains closer to baseline? Together with timing this could distinguish the endpoints. | Destination aggregate traffic mixes incoming writes, local reads, and control. HBM saturation is not implied by high utilization; do not infer exact transaction routing from these counters. |
| Data-path delay versus receiver/control delay | Destination-local polling start/end and observation records, source notification/acknowledgement intervals, verified control placement and both GPUs' telemetry | Is the extra completed-transfer time primarily before source publication or in receiver observation/acknowledgement? Does destination pressure delay polling/control despite separate SMs? | A late receiver timestamp alone is not slow incoming writes. Cross-device raw timestamps cannot establish one-way delay without calibration. |

Present communication throughput and duration first, with one clearly labeled panel per complete communication geometry and comparable zero-based axes. Show destination read rate second. Then plot phase distributions and receiver wait against achieved destination read rate. Compare with source-pressure cases only where communication geometry, read implementation, and compiled program match. The strongest candidate endpoint signature combines communication progress with independently measured pressure location and instruction evidence; the configured co-runner location alone is not a diagnostic observation. If source-only measurements look alike in both cases, retain that ambiguity and assess whether destination counters resolve it. If no counter separates them, record that outcome rather than claiming endpoint RCA.

## Destination reads with the calibrated communication kernel

> **Revision 2026-10-28:** new runs use the [rate-controlled memory co-runner](#memory-pressure-with-the-rate-controlled-co-runner-revision-2026-10-28): 24 pressure CTAs, target read rate 20/40/60/80% of HBM peak. The text below is kept as the record of the earlier plan and its executed runs.

This follow-up replaces the earlier six-geometry grid for testing destination pressure at the communication plateau. Use the unchanged binary selected in [communication calibration](results/2026-10-27-saturation-gpu6-7/README.md). Execution records belong in [the dated contention report](results/2026-10-27-saturated-contention-gpu6-7/README.md).

### Motivation

Does destination read activity reduce completed communication throughput after the communication request supply has been calibrated? If communication stays unchanged, determine whether the destination reads slow and whether both roles actually overlap.

### Configuration

| Knob | Role | Setting |
|---|---|---|
| Devices and direction | Constant | Physical GPU 6 issues stores to GPU 7 |
| Communication | Constant | 128 CTAs × 512 threads/CTA; 2048 MiB; CTA-local stride; batch 8 scalar 4-byte operations |
| **Destination read CTA count** | **Independent variable** | **0 (copy-only reference), 8, 32, 100, 131 CTAs × 512 threads/CTA** |
| Destination control | Constant | 1 CTA × 512 threads/CTA, separate SM from read CTAs |
| Read implementation | Constant | Separate 1 GiB buffer; vector reads, four independent checksum chains; 256 KiB bursts; no pacing |
| Residency and completion | Constant | One CTA per SM; all-writer system fence, publication, receiver acknowledgement; full-buffer validation |
| Primary repetitions | Constant | Five adjacent copy-only / copy-with-reads pairs per level; rotate levels; three warm-ups and twelve retained transfers per invocation |
| Read calibration | Separate control | Communication disabled; source control 128 × 512; destination reads M × 512 plus control 1 × 512; five invocations of five retained 20 ms windows per level |
| Read performance comparison | Separate paired control | Five reads-only / reads-with-copy pairs per level; reads-only timer matched to observed concurrent read-active duration; three warm-ups and twelve retained windows |
| Probe overhead | Separate control | Five minimal/detailed pairs at destination reads 131 × 512, with communication fixed |

### Procedure

1. Check GPU availability and confirm binary/source hashes match the calibrated reference. Run a fresh communication-only reference.
2. Measure achieved destination read rates at all four settings before combining workloads. Report whether the rate is still rising at the maximum available read-CTA count; reaching that count does not prove HBM saturation.
3. Run the primary pairs sequentially. Check full-payload correctness, per-role SM placement, active intervals, and completed read bursts. Present communication throughput, duration, variation, and slow transfers first.
4. Use separately paired, duration-matched controls to measure the destination read change. Preserve the distinction between this comparison and the primary copy-only reference.
5. Run probe-overhead checks and a return-to-reference invocation, then separate Nsight Systems passes for communication alone and the strongest destination pressure. Check that profiling retains actual read progress and overlap. Retain unavailable hardware-counter evidence as a limitation.

### Privileged access

Use the already allocated GPUs 6–7 and read-only device telemetry. Do not alter clocks or power settings. Hardware counters require a validated concurrent profiling method; earlier unsuccessful replay attempts are not counter evidence.

### Observations and interpretation

| Observation | Instrument, endpoint, window | Question and limit |
|---|---|---|
| Completed throughput and duration (A1) | Source timestamps from earliest start to latest acknowledgement, each transfer | Does communication degrade against its adjacent reference? This includes the completion protocol. |
| Issue, ordering, acknowledgement and CTA spread (A2–A7) | Source per-CTA timers and checkpoints, each transfer | Where does extra time appear? Phase medians do not sum to transfer duration; source timing alone cannot locate a physical resource. |
| Read progress and overlap (A8–A10) | Destination burst counts, SM IDs and local active intervals, each transfer | Did every read CTA cover receiver activity and make progress? Logical GB/s includes startup/tail and is not physical HBM bandwidth. |
| Kernel overlap (B1) | Separate Nsight Systems measurement-range capture on both GPUs | Are endpoint kernels concurrent with the intended geometry? Device probes are still needed to distinguish roles within each kernel. |
| Operating state (B2–B3) | Both GPUs, invocation-wide snapshots and sampled telemetry | Check environmental drift; these samples cannot establish clocks during each short transfer. |

## Memory pressure with the rate-controlled co-runner (revision 2026-10-28)

**Status: plan only, not yet run.** This section replaces the memory co-runner of Experiment 3 and Experiment 4 for all new runs. The text of those experiments above stays as the record of what was planned and run before.

### Why this revision

Earlier memory pressure came from plain 16-byte loads. The pressure level was set by changing read CTAs and threads per CTA. This had two problems:

1. **Too many SMs.** Plain loads needed about 128 SMs × 1024 threads to come near the HBM peak. On the source GPU this left no room for the calibrated communication setting (128 CTAs). The strongest reads and the strongest communication could not run together.
2. **Two things changed at once.** A change in read CTAs or threads changed both the SMs the reads held and the traffic they made.

The new co-runner fixes both. It is described and calibrated in the [HBM read-rate calibration](../../../../h100x1/memory_co_runner/calibration/hbm_read_rate/README.md). In short:

| Item | Detail |
|---|---|
| How it reads | Thread 0 of each pressure CTA starts Hopper bulk copies (TMA) of 64 KiB chunks from a 1 GiB buffer into shared memory; thread 32 watches the stop flag; the other threads are idle |
| Setting used here | **24 pressure CTAs × 3 chunks of 64 KiB in flight per CTA** |
| The knob | **Target read rate**, as a percent of HBM peak (3352 GB/s) |
| Measured alone (GPU 6, separate harness) | Achieved = target within ±0.02 percentage points from 5% to 90%; ceiling with no limit 91.3% |

Now only the target rate changes between pressure levels. The pressure CTAs, their SMs and their shared memory stay the same.

### Shared changes for both experiments

These changes come before any run. Each needs a behaviour check and fresh references, as for every earlier binary change.

| Change | Detail | Why |
|---|---|---|
| Add the co-runner to the fused kernel | Include `hbm_read_pressure.cuh`; add the pressure role on the source (Experiment 3) or the destination (Experiment 4); keep the existing plain-load roles for old configs | One build for both experiments |
| New config keys | Pressure kind for source / destination bulk-copy reads; `chunk_kib=64`; `chunks_in_flight=3`; `target_gbps`; a mode where pressure CTAs start, report ready, and then only wait for the stop signal ("idle pressure CTAs") | The idle mode gives a reference with the same launch as every pressure level |
| Shared memory per CTA | **196,864 bytes for every CTA, on both GPUs, in every run of these two experiments**, including references and reads-alone controls | Every CTA of a fused launch gets the same dynamic shared memory. Changing it between arms would change more than the traffic. |
| Physical block | 1024 threads for every CTA, as in the 2026-10-27 source-memory build | Same CTA shape for all arms |
| Output | Record target rate, chunks read per pressure CTA, start/end per pressure CTA, SM IDs | Achieved rate (A9), active interval (A8), placement (A10) |

Things to check after the change:

1. The pressure role in the fused kernel, alone (communication disabled), gives achieved ≈ target at 20/40/60/80%. The earlier calibration ran in a separate harness, so this is a fresh check, not a transfer.
2. Communication with idle pressure CTAs versus communication with no pressure CTAs, five adjacent pairs. If they differ, report it and keep the idle-CTA reference as the primary one.
3. The larger shared-memory reservation leaves less L1 cache for the communication CTAs. Earlier communication references (114 KiB reservation) cannot be reused.

### Experiment 3 revision: rate-controlled reads on the issuing GPU

#### Motivation

The copy reads its source array from source-GPU memory. This tests whether known, rising read traffic on separate source SMs slows the copy. Earlier tests reached only about 30% of HBM peak next to the copy. Now the reads can go up to 80% of peak with only 24 SMs.

#### Configuration

| Knob | Role | Setting |
|---|---|---|
| Devices and direction | Constant | Physical GPU 6 (source) → GPU 7 (destination) |
| **Target read rate on the source** | **Independent variable** | **20%, 40%, 60%, 80% of peak** = 670, 1341, 2011, 2682 GB/s; reference: idle pressure CTAs (no reads) |
| Source pressure CTAs | Constant | 24 CTAs × 3 × 64 KiB in flight; 1024 threads/CTA, 2 working |
| Source communication | Constant within the comparison; chosen by calibration step 1 | C CTAs × Tc threads/CTA with **C + 24 ≤ 132**, so C ≤ 108; 2048 MiB; 8 loads per thread; CTA-local stride |
| Destination | Constant | 1 control CTA × 1024 threads/CTA; no destination pressure |
| Shared memory, block | Constant | 196,864 bytes per CTA; 1024 threads per CTA; both GPUs |
| Completion and checks | Constant | All-writer fence, publication, acknowledgement; full-buffer check after every transfer |
| Repeats | Constant | 5 adjacent reference / pressure pairs per level, level order rotated; 3 warm-ups and 12 transfers per run |

**Calibration of the communication setting (step 1).** The calibrated plateau, 128 × 512, does not fit next to 24 pressure CTAs. Measure, with idle pressure CTAs present:

| Communication CTAs × threads/CTA | Purpose |
|---|---|
| 64 × 512, 64 × 768, 64 × 1024 | Earlier the 64-CTA curve was nearly flat at 1024 threads (+1.0% from 768) |
| 96 × 512, 96 × 768, 96 × 1024 | Middle point |
| 108 × 512, 108 × 768, 108 × 1024 | Largest that fits next to 24 pressure CTAs |

Five rounds each, rotated order. Choose the setting with the highest median throughput. If its curve still rises by 1% or more at the last step, label it "still rising", not saturated. Use that one setting for the whole sweep.

#### Procedure

1. Build; run the three checks in "Shared changes". Record build hashes.
2. Calibrate communication (table above). Select and freeze C × Tc.
3. Reads alone: communication disabled, the same C control CTAs as timers, 24 pressure CTAs at each target. Five runs per level. Record achieved versus target.
4. Main pairs: for each target, five adjacent runs of copy with idle pressure CTAs, then copy with reads at that target. Rotate level order. Run one reference at the start and one at the end.
5. Read side, second: at 80%, and at any level where the copy slows or reads fall below target, run five pairs of reads-only versus reads-with-copy. Set the reads-only window to the measured active time of the concurrent reads.
6. Probe overhead: five pairs of minimal versus detailed probes at 80%.
7. Timelines (Nsight Systems), separate runs: reference and 80%. Check that pressure CTAs are active for the whole copy.

#### Privileged access

None beyond the allocated, idle GPUs 6–7. No clock or power changes. Hardware counters are optional and need a validated method.

#### Observations and interpretation

| Observation | How it is measured | Question it answers | Limit |
|---|---|---|---|
| Copy throughput and duration (A1) | Source start to last acknowledgement, each transfer | **Primary:** does the copy slow as the target rises? | Includes the completion protocol |
| Source phase times (A2–A7) | Per-CTA issue, ordering, acknowledgement times | Where does any extra time appear? | Phase medians do not add up; not remote-store latency |
| Achieved read rate next to the copy versus target (A9) | Chunks read × 64 KiB / active time, per transfer | Did the memory system serve the reads fully while the copy ran? A gap shows the reads gave way. | Requested bytes, not counted HBM bytes |
| Read activity covers the copy (A8, A10) | Per pressure CTA start/end and SM ID | Were reads active the whole time, on separate SMs? | Coverage is not constant activity |

Expected (hypothesis, not a result): at 80% the reads ask for about 2682 GB/s, and the copy reads about 370 GB/s or less from the same memory. Together that is about 91% of peak, close to the measured ceiling of about 95%. So if the copy slows at all, 80% is the most likely level; at 20–60% total demand stays well below the ceiling. Another explanation for no slowdown: the copy keeps more loads in flight than it needs, so a longer memory delay may not lower its throughput. These two can be told apart only with a memory-delay probe or counter data; report the outcome either way.

### Experiment 4 revision: rate-controlled reads on the destination GPU

#### Motivation

Incoming peer stores land in destination memory. The 2026-10-27 test with plain loads showed almost no copy change (at most −0.12%) while the destination reads slowed by about 21%. That test changed read CTAs, not a controlled rate. This revision asks the same question with known, rising read traffic.

#### Configuration

| Knob | Role | Setting |
|---|---|---|
| Devices and direction | Constant | Physical GPU 6 (source) → GPU 7 (destination) |
| **Target read rate on the destination** | **Independent variable** | **20%, 40%, 60%, 80% of peak** = 670, 1341, 2011, 2682 GB/s; reference: idle pressure CTAs (no reads) |
| Destination pressure CTAs | Constant | 24 CTAs × 3 × 64 KiB in flight; 1024 threads/CTA, 2 working |
| Destination control | Constant | 1 CTA × 1024 threads/CTA, separate SM |
| Source communication | Constant | 128 CTAs × 512 threads/CTA; 2048 MiB; 8 loads per thread; CTA-local stride; confirmed again in step 1 |
| Shared memory, block | Constant | 196,864 bytes per CTA; 1024 threads per CTA; both GPUs |
| Completion, checks, repeats | Constant | As in the Experiment 3 revision |

#### Procedure

1. Use the build from the Experiment 3 revision; run its checks if not done.
2. Confirm the communication reference under the new reservation: copy-only 128 × 512 and 128 × 1024, five rounds each, with idle destination pressure CTAs. If they differ by less than 1%, keep 128 × 512. If not, report it and choose again by the plateau rule.
3. Reads alone on the destination: copy disabled, source control 128 × 512 as timers, 24 destination pressure CTAs at each target. Five runs per level.
4. Main pairs: five adjacent copy-with-idle-pressure / copy-with-reads pairs per target, rotated; one reference at the start and one at the end.
5. Read side, second: at 80%, and at any level with a copy slowdown or reads below target, five duration-matched reads-only versus reads-with-copy pairs.
6. Probe overhead at 80%; separate Nsight Systems timelines for reference and 80%.

#### Privileged access

None beyond the allocated, idle GPUs 6–7. No clock or power changes.

#### Observations and interpretation

The same table as the Experiment 3 revision, with achieved read rate and coverage measured on the destination, and the receiver wait time (A6) added.

Expected (hypothesis): at 80% the destination memory serves about 2682 GB/s of reads plus about 370 GB/s of incoming writes, about 91% of peak. The earlier result suggests the reads, not the copy, may give way. If so, the achieved read rate falls below the 80% target while copy throughput stays close to its reference. That would show the interaction in a measured, controlled form. It would not identify the arbitration rule inside the memory system.

### Deviations found while running (2026-10-28)

Results are in the [report](README.md#rate-controlled-memory-reads-shared-method). These notes change the plan for later runs:

1. **Pairs inside one run.** With a 64-CTA copy, the copy CTAs land on a different set of SMs in each run, and throughput follows that set by up to 8%. Separate-run pairs would mix this into every comparison. Each run now alternates idle and reading launches (6 + 6 measured transfers); the run's pair is the median of each half. Use this for any copy that does not fill almost all SMs.
2. **Source copy setting.** The calibration chose 64 × 512 (370.4 GB/s median). 96 and 108 CTAs gave about 270 and 257 GB/s because their CTA-local regions are uneven.
3. **The 80% target was not reached on the source GPU,** even with the copy off (about 71% of peak), because the 24 pressure CTAs landed on neighbouring SMs next to the 64 copy CTAs. Report achieved rates. To reach 80% there, spread the pressure CTAs (for example by placing them first) or add pressure CTAs, and calibrate again.
4. **The idle-versus-no-pressure check and other separate-run comparisons** (duration-matched reads, probes) are confounded by placement on the source GPU; only the destination versions are usable.
5. **Not done:** Nsight Systems timelines and hardware counters.
6. **Added level: 95% of HBM peak.** 24 pressure CTAs stop at about 91%, so this level uses **48 pressure CTAs × 3 × 64 KiB** with a 95% target (3185 GB/s), on both GPUs (source: 64 copy CTAs + 48 = 112 SMs). Its reference is the same 48 CTAs, idle, in the same run. A screen of 32, 40 and 48 CTAs (reads alone) found about 94% as the reachable rate in the fused kernel; exactly 95% was not reached.

### What stays from the earlier work

- **2026-10-27 to 2026-10-28 source-memory calibration** (plain loads, separate role thread counts): validation, read grid, loads-per-lane check and communication calibration finished under `results/2026-10-27-source-memory-gpu6-7/`; read-only controls, pairs and report did not run. They remain evidence about the plain-load reader. Their communication calibration used a 114 KiB reservation and does not transfer to this revision.
- **2026-10-27 destination-pressure results** stay as measured. They used a different reader and reservation; do not pair them with runs from this revision.

## Implementation refinement recorded 2026-10-27

The initial scalar read co-runner is retained as a calibration variant. A second variant uses coalesced 16-byte vector loads, four independent checksum chains, and four-way unrolling to expose more memory requests while keeping the communication copy scalar and unchanged. Both endpoints use the same variant and disjoint 1 GiB working set; matched no-pressure references are remeasured with the revised binary. The destination sweep additionally includes its exact available maximum of 131 memory CTAs after reserving one receiver/control CTA. Achieved rates, run status, and comparisons belong in [the report](README.md); this refinement does not assert a particular bottleneck or expected slowdown.

The earlier workload-alone comparison also has a matched-duration follow-up (distinct from the new three-step plan above): use a 3.3 ms standalone window and interleave it with the concurrent copy at 100 and 131 destination read CTAs, retaining the same co-runner, launch reservations, and binary. Compare actual active intervals as well as logical read rates, so a long standalone calibration window cannot be confused with a communication effect. Results remain in the report.
