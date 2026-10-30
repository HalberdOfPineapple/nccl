# 07 · Data-store congestion on the issuing GPU

This plan covers the problem-matrix cell **thread-issued peer stores × shared-resource contention**. It narrows plan [04](../shared_resources/04_shared_resource_contention.md) to one question about the issuing (source) GPU:

**Common question.** When some SMs of the source GPU store data to the peer GPU over NVLink, and other SMs of the same GPU store data to local memory, do the two store flows slow each other down? Does it matter whether they share a GPC?

Words used in this plan:

| Word | Meaning |
|---|---|
| Copy | The communication kernel: CTAs on the source GPU store data into the destination GPU's memory through an NCCL LSA peer pointer |
| Writer | The pressure workload: CTAs on the source GPU store data into a local 1 GiB buffer |
| GPC | A group of SMs that shares one connection into the GPU's on-chip network (crossbar). On H100 there are 8 on the full chip |
| Store flow | Data moving from SMs into the crossbar. Peer stores then go to the NVLink unit; local stores go to L2 and HBM |

## Why this plan

Plan 04 measured the copy against **read** traffic. Two parts of that design hide the competition we want to observe:

1. **The copy also reads.** Each copy thread loads source elements from local HBM before storing them. On the source GPU this adds about 370 GB/s of HBM reads, so read pressure partly competes with the copy's loads, not with its stores.
2. **Reads and stores move data in opposite directions.** Store data leaves the SM (SM → crossbar). Read data comes back to the SM (L2 → crossbar → SM). If the on-chip network carries requests and replies separately (common in GPU designs, not documented for H100), read data and store data hardly share it.

This plan changes both:

| Change | Effect |
|---|---|
| **Store-only copy**: each value is computed from its index in registers; no source loads | The copy uses no source-GPU memory. The issue loop then measures store issue only, so store backpressure shows directly in it |
| **Writer instead of reader**: other SMs store to local memory | Both flows leave SMs in the same direction, into the crossbar. They split only after it: NVLink unit or L2 |
| **Placement as a knob**: copy and writer CTAs in the same GPCs or in different GPCs | Tests whether competition happens at the GPC's connection into the crossbar, not only inside the crossbar or memory |

The plan-04 results stay valid; they answer a different question (a copy with loads, against reads). The read co-runner is reused here as a contrast (E7.5).

## Common experiment configuration

### GPUs and roles

| Item | Setting |
|---|---|
| GPUs | One allocated pair from physical GPUs 0, 1, 6, 7; each experiment stays on one pair; the pair is recorded in every raw record, not in configs |
| Source GPU | Runs the copy CTAs, the writer CTAs, and idle CTAs |
| Destination GPU | One receiver/control CTA; no pressure on the destination |
| Clocks and power | Unchanged; observed, not controlled |

### Store-only copy

| Field | Setting | Note |
|---|---|---|
| Interface | NCCL LSA peer pointer, ordinary thread stores | As in plan 04 |
| Store width | 4 bytes per store (scalar), as before; 16-byte stores are a later variant | Keep the access type comparable with plan 04 |
| Values | `value = hash(element index, transfer number)`, computed in registers | The destination check still catches missing **and** misplaced writes. A constant would catch only missing writes |
| Stores in flight per thread | Independent stores per loop step: 1, 2, 4, 8 (calibrated) | Replaces the load batch of plan 04 |
| Payload | 2048 MiB | Long enough for stable timing |
| Traversal | CTA-local stride | As in plan 04 |
| Completion | All writers fence, each CTA publishes "done", the receiver acknowledges | Unchanged, so timers mean the same as before |
| Correctness | Destination is filled with 0xFF before each transfer and fully checked after it | Unchanged |

### Writer

| Field | Setting |
|---|---|
| Operation | Thread-level stores to a separate local 1 GiB buffer; each writer CTA writes its own contiguous slice, in order, and repeats |
| Store width | 16-byte stores (`st.global.v4`), coalesced; values computed from the index |
| **Knob: target write rate** | Total bytes per second for all writer CTAs. Each CTA writes one 64 KiB block with all its active threads, then waits until the block's scheduled start time, as in the read co-runner |
| Ceiling | Measured in E7.2. HBM write and read ceilings may differ; levels are set as a percent of the HBM peak (3352 GB/s) and reported with the achieved rate |
| Proof of work | Blocks written per CTA; a sample of the buffer is checked after each window |
| Variant (E7.6) | TMA bulk stores (shared memory → global, `cp.async.bulk.global.shared`): few SMs, different request mechanism |

### Placement: roles chosen by SM, not by CTA index

Plan 04 found that the GPU puts CTAs on a different set of SMs in each run, and that copy throughput follows that set by up to 8%. This plan controls placement instead of only recording it:

1. Launch one CTA per SM on all 132 SMs (cooperative launch, one CTA per SM, as now).
2. Each CTA reads its SM ID (`%smid`) at the start and looks up its GPC in the map from E7.0.
3. A role table in the config says which GPCs (or which SMs) host copy CTAs, writer CTAs, or nothing. CTAs on unused SMs exit at once.
4. Copy CTAs take their part of the array in the order they register (an atomic counter), so every part is covered whatever SMs are used.

The same role table then gives the same SMs in every run. Pairs inside one run (alternating idle and writing transfers, as in plan 04) are still used, because they also share buffers and operating state.

### Shared memory and block size

| Setting | Value |
|---|---|
| Physical block | 1024 threads for every CTA; inactive lanes join barriers only |
| Shared memory | Default reservation (just over half an SM) for thread-store roles; 196,864 bytes only in runs that include the TMA co-runners (E7.5 read contrast, E7.6), with their own references |

### Observations recorded in every run

Metric ids are from [00_common_setup.md](../00_common_setup.md).

| Observation | Id | Endpoint | Note |
|---|---|---|---|
| Copy duration and throughput | A1 | Source timer | Primary outcome |
| CTA start, issue loop, ordering, publication, acknowledgement wait | A2–A6 | Source timer | Issue loop is now pure store issue |
| Issue progress checkpoints | A7 | Source timer | Store progress, not arrival |
| Writer active interval and coverage of the copy | A8 | Source timer | Every writer CTA must cover the whole copy in writing transfers |
| Writer achieved rate and blocks written | A9 | Source | Requested bytes, not counted HBM bytes |
| CTA role and SM ID | A10 | Both | Must match the role table |
| SM-to-GPC map | A12 | Source | From E7.0; recorded with each run's role table |
| Destination correctness | A11 | Destination | Every element of every transfer |
| Operating state, NVLink counters | B2, B3 | Both | Whole-run samples |

## E7.0 · Map SMs to GPCs

### Motivation

Placement is a knob in this plan, so we must know which SM IDs share a GPC. NVIDIA does not document the mapping. CUDA guarantees that the CTAs of one thread-block cluster run in the same GPC, so the mapping can be measured.

### Configuration

| Knob | Setting |
|---|---|
| GPU | Each GPU of the allocated pair, measured separately |
| **Cluster size** | **2, 4, 8, 16** (16 needs the non-portable cluster-size attribute); grids that fill all 132 SMs |
| Repeats | 20 launches per cluster size |

### Procedure

1. Launch clusters; each CTA records its cluster ID and SM ID.
2. Build a graph: two SMs are joined if they were ever in the same cluster. Connected groups are GPCs.
3. Check that every SM lands in exactly one group, and that group sizes add up to 132.
4. Save the map as a small JSON file next to the configs; use it in every later step.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A12 SM-to-GPC map | 7 or 8 groups of 16–18 SMs (hypothesis) | Not a perturbation; setup evidence |

## E7.1 · Store-only copy alone

### Motivation

The store-only copy is a new workload, so it needs its own reference: the setting where more issuing threads no longer raise throughput. Without loads it may reach the plateau with fewer threads or behave differently from the plan-04 copy.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Copy CTAs × threads/CTA** | **Independent variable** | **CTAs: 16, 32, 48, 64, 96, 128 at 512 and 1024 threads/CTA; threads/CTA: 256, 512, 768, 1024 at 64 and 128 CTAs** |
| **Stores in flight per thread** | **Independent variable at the top of the curve** | **1, 2, 4, 8** |
| Placement | Constant within a sweep | Copy CTAs spread evenly over all GPCs; a second sweep packs them into as few GPCs as possible |
| Writer | Constant | Absent |
| Repeats | Constant | 5 runs per setting, rotated order; 3 warm-ups and 12 transfers per run |

### Procedure

1. Run the sweep; record achieved throughput as table with row being number of CTAs and column indicating number threads per CTA. Each number of store in flight can have a separate table.
2. Stop rule: less than 1% gain over two successive increases, confirmed in 5 runs.
3. Compare even and packed placement at the selected setting.
4. Select the copy setting for each later budget: copy CTAs must leave room for writer CTAs (copy + writer ≤ 132).

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1 | Rises with issuing threads, then flattens near the plan-04 plateau (about 370 GB/s), possibly with fewer threads (hypothesis) | Reference only |
| A3, A7 | Issue loop close to the full transfer time | Reference only |
| A10, A12 | Packed and even placement may differ (hypothesis) | Placement effect without pressure |

## E7.2 · Writer alone

### Motivation

The pressure must be known before it is applied. This measures how fast the writer can store to local HBM, how many SMs it needs, and whether the rate knob holds, as was done for the read co-runner.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Writer CTAs** | **Independent variable** | **4, 8, 16, 24, 32, 48, 64, 96, 128** at 1024 threads/CTA |
| **Writer threads/CTA** | **Independent variable** | **256, 512, 1024** at 32 and 64 writer CTAs |
| **Target write rate** | **Independent variable at a fixed writer setting** | **20, 40, 60, 80% of HBM peak, plus no limit** |
| Copy | Constant | Off; copy CTAs replaced by idle CTAs on the same SMs as in the later pairs |
| Repeats | Constant | 5 runs per setting; 2 warm-ups and 5 windows of 20 ms |

### Procedure

1. Sweep writer CTAs and threads with no limit; find the ceiling and the fewest SMs that reach it (95% rule, as in the read calibration).
2. Sweep the target rate at the chosen writer setting; check achieved against target.
3. Repeat with the writer packed into few GPCs and spread over many, since E7.3 uses both.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A9 | Achieved = target below the ceiling; ceiling may be lower than the 95% read ceiling (hypothesis) | Calibration only |
| A10, A12 | Packed writers may reach a lower ceiling (hypothesis) | GPC-connection limit for writes |

## E7.3 · Local writes next to the copy, in different GPCs

### Motivation

This is the main test. Copy CTAs and writer CTAs run in **separate GPCs**, so they do not share a GPC's connection into the crossbar. Any slowdown must come from the crossbar itself, from L2, or from later parts of the store paths.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Target write rate** | **Independent variable** | **0 (idle writers), 20, 40, 60, 80% of HBM peak, and the writer's no-limit ceiling** |
| Copy | Constant | Setting from E7.1, in its own GPCs |
| Writer | Constant | Setting from E7.2, in the other GPCs |
| Pairs | Constant | 5 runs per level; each run alternates idle and writing transfers (6 + 6 measured) |

### Procedure

1. Run the opening reference (writers idle).
2. For each level, 5 alternating runs; rotate level order between rounds.
3. Run the closing reference.
4. Writer side, second: at the highest level, 5 pairs of writer-alone versus writer-with-copy runs, with the writer-alone window matched to the measured active time.
5. Probe cost: 5 minimal-versus-detailed pairs at the highest level.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1 | Small or no change if the crossbar has spare capacity (hypothesis); a change that grows with the write rate points to a shared part after the GPC | From E7.4: same rate, different placement |
| A3 | If the store path pushes back, the issue loop grows (now pure store issue) | From plan 04, where only completion grew |
| A4, A6 | Growth here without issue-loop growth points to completion, not issuing | Cannot separate crossbar from L2 by itself |
| A9 | Writer achieved rate falls when the copy runs, if they compete | Mutual effect |

## E7.4 · Local writes next to the copy, in the same GPCs

### Motivation

Same as E7.3, but copy CTAs and writer CTAs **share GPCs**. If the GPC's connection into the crossbar is a shared bottleneck, this test should show a larger slowdown than E7.3 at the same write rate.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Placement** | **Independent variable (compared with E7.3)** | **Each used GPC hosts both copy and writer CTAs, in the same proportion as the whole** |
| **Target write rate** | **Independent variable** | **Same levels as E7.3** |
| Copy, writer, pairs | Constant | Same settings and SM counts as E7.3; only which GPCs they share changes |

Variant **E7.4b**: one GPC fully used by writer CTAs plus a few copy CTAs, to put the strongest local pressure on a small part of the copy. The copy then waits for its slowest CTAs; per-CTA issue times show which CTAs were slowed.

### Procedure

As E7.3, interleaved with E7.3 by round (E7.3 run, E7.4 run, same level, same round), so both placements see the same machine state.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1 | Larger slowdown than E7.3 at the same achieved write rate if the GPC connection is shared (hypothesis) | Placement: GPC connection versus crossbar/L2 |
| A2–A7 per CTA | Copy CTAs in GPCs with writers slow more than the others (E7.4b) | Local versus global effect |
| A9 | Writer rate may also fall more when sharing GPCs | Mutual effect |

## E7.5 · Direction contrast: local reads next to the store-only copy

### Motivation

To check that direction matters, repeat E7.3 with the TMA read co-runner (from plan 04) instead of the writer, at the same achieved byte rate. If writes slow the copy more than reads, the shared part is on the store direction.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Traffic direction** | **Independent variable** | **Local writes (E7.3 runs) versus local reads (this test)** |
| Read co-runner | Constant | 24 pressure CTAs × 3 × 64 KiB; 196,864 bytes shared memory; its own references with the same shared memory |
| Levels | Constant | Read levels matched to the achieved write rates of E7.3 |
| Placement | Constant | Separate GPCs, as in E7.3 |

### Procedure

As E7.3, with alternating runs.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1 | Less change with reads than with writes at the same byte rate (hypothesis) | Direction of traffic |
| A3 | Little issue-loop change with reads | Direction of traffic |

## E7.6 · Writer variant: TMA bulk stores (optional)

### Motivation

A fused kernel on Hopper may write its results with TMA bulk stores. This variant reaches high write rates with few SMs and uses a different request mechanism than thread stores.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Writer mechanism** | **Independent variable** | **Thread stores (E7.3) versus TMA bulk stores** |
| TMA writer | Constant | Shared memory → global, 64 KiB chunks, chunks in flight calibrated as in the read co-runner; rate knob |
| Levels, placement | Constant | As E7.3 |

Calibrate it alone first (as E7.2), then run as E7.3.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1, A3 | Similar to thread stores at the same rate if only the byte rate matters (hypothesis) | Request mechanism |

## Order of work

| Step | Work | GPU time (estimate) |
|---|---|---|
| 1 | Implement: store-only copy kind, role-by-SM table, thread-store writer with rate knob, cluster probe for E7.0; behaviour check against the current build | None |
| 2 | E7.0 GPC map on both GPUs of the pair | Minutes |
| 3 | E7.1 copy calibration, E7.2 writer calibration | About 30 min |
| 4 | E7.3 and E7.4, interleaved | About 30 min |
| 5 | E7.5 direction contrast | About 15 min |
| 6 | E7.6 if requested | About 30 min with calibration |
| 7 | Analysis and report in `data_store_congest/README.md` | None |

## Interpretation limits

- **Requested bytes are not counted HBM bytes.** Writes to a 1 GiB buffer must reach HBM eventually, but L2 may hold and combine them. Hardware counters were not validated in earlier sessions.
- **The crossbar's structure is not documented.** A slowdown in E7.4 but not in E7.3 points to the GPC connection; a slowdown in both points to a shared part after it (crossbar, L2, or the store-completion path). It does not name a physical queue.
- **The GPC map is measured, not documented.** If cluster launches cannot place clusters on every SM, the map may be incomplete; record any SM that never appears.
- **Completion timers are not store latency.** The issue loop covers store issue and backpressure; ordering and acknowledgement wait cover the completion protocol, as in plan 04.
- **No slowdown is a result.** It means the tested store flows did not interfere at these rates and placements, not that they cannot interfere.

## Implementation notes (2026-10-29)

Added after implementing the harness and running E7.0. The design above is unchanged except for these points.

1. **GPUs.** The user allocated physical GPUs 4 (source) and 5 (destination) for E7.0–E7.2.
2. **Unmapped SMs.** On both GPUs, SMs 124–131 never ran a CTA of a cluster with 4 or more CTAs, so cluster launches cannot place them in a GPC. They are recorded as unmapped. Roles use the 124 mapped SMs by default. Settings with more than 124 role CTAs (the 128-CTA points of E7.1 and E7.2) also use unmapped SMs. Later steps (E7.3 onwards) must keep copy + writer ≤ 124.
3. **Copy part boundaries** are rounded down to 128 bytes. Without this, CTA counts such as 48 and 96 split every warp store over two 128-byte lines. The first sweep attempt lost about 30% at those counts; its records are kept locally under `results/2026-10-29/rejected/`.
4. **Writer-alone runs (E7.2)** use one copy-role CTA as the window timer (`copy_ctas=1`, copy off). Every other SM holds an idle CTA, so the idle copy CTAs of the later pairs are present either way.
5. **Implementation.** A separate binary, `common/bin/store_congest`, implements the plan; `push_harness` is unchanged. See the [harness README](../../common/README.md#store-congestion-harness-plan-07).
6. **GPUs 0 and 1 (2026-10-30).** At the user's request the work moved to GPUs 0 → 1. The cluster probe gave the same SM-to-GPC map on GPUs 0, 1, 4 and 5, and the user asked to treat both pairs as the same hardware; results stay in one dated folder and each raw record names its pair.
7. **Copy floor grid (E7.1, 2026-10-30).** The planned copy grid was already on the plateau at its smallest setting (16 CTAs), so 1, 2, 4, 8 and 16 CTAs × 128, 256, 512 and 1024 threads/CTA were added (`copy_floor`) to find where the plateau starts.
8. **3 runs per setting (2026-10-30).** At the user's request, phases run from then on use 3 runs per setting instead of 5. Analyses use every run present and report the count.
9. **"Stores in flight per thread" removed (2026-10-30).** In the store-only copy this knob only set how many stores each thread writes per loop step: the addresses and their order were the same for 1, 2, 4 and 8, and a store does not wait for a reply, so it could not change how many stores are in flight. The compiler kept one bounds-check branch per store, so values above 1 added instructions; they never raised throughput by more than 0.3% and lowered it by up to 33% at 16 CTAs and up to 41% at 2 CTAs. At the user's request the knob was removed from the harness; the copy issues one store per loop step. The E7.1 copy grid was remeasured with the new build, because the same loop compiled in a different order ran 7–8% slower per SM. The earlier records are kept as the evidence for this change. The E7.1 table row "Stores in flight per thread" no longer applies.
10. **E7.3–E7.6 (2026-10-30).** GPUs 0 → 1, 3 runs per level (user's request), each run 6 idle + 6 active transfers. E7.3: copy packed into G6, writer spread over the other GPCs. E7.4: both spread over all GPCs (about 2 copy and 8 writer SMs per GPC). E7.4b: copy spread, writer packed into G6, G7, G1, G2 (plus 4 SMs of G3), so 8 copy CTAs sit in full writer GPCs. The writer side uses writer-alone runs on the same SMs with windows matched to the measured writer active time, at every level rather than only the highest.
11. **E7.5 reader** uses 48 CTAs × 3 × 64 KiB at every level (24 CTAs stop at about 91% of peak in plan 04); read targets are the write rates achieved in E7.3. The E7.3 write runs were repeated in the same session, back to back with the reads.
12. **E7.5/E7.6 build.** The TMA read co-runner (from plan 04) and a new TMA bulk-store writer are compiled into `bin/store_congest_tma` (`-DSTORE_CONGEST_TMA`, config key `writer_kind` 1 or 2, `chunks_in_flight`); `bin/store_congest` keeps the instructions of the E7.1–E7.4 build. The TMA writer stores one 64 KiB shared-memory chunk repeatedly with `cp.async.bulk`, bounding chunks whose writes are incomplete; it was calibrated alone (64 CTAs × 1 chunk selected) before the contrast.
