# 01 · TMA push next to TMA memory co-runners

This plan covers the problem-matrix cell **TMA-based transfer × shared-resource contention**. It repeats the design of thread-store [plan 07](../../../thread_store/observation/data_store_congest/07_data_store_congest.md) with a TMA push as the communication and TMA kernels as the memory co-runners.

**Common question.** When some SMs of the source GPU push data to the peer GPU with TMA bulk stores while TMA co-runners store to (or read from) local memory, does the push slow down? Does it matter whether they share an SM (and so its TMA unit), or run on different GPUs?

**Revision 2026-11-10 (user's choice, before any calibration).** GPC placement is not a question of this plan. All runs, calibration included, use one layout: copy packed into G6, co-runners spread over the other 7 GPCs. The shared-GPC and crowded-GPC tests (E1.4, E1.4b) and the copy placement step of E1.1 are dropped. The SM-to-GPC maps are reused from plan 07; no mapping runs. The same-SM test (E1.5) stays: it shares an SM's TMA unit, not a GPC.

Words used in this plan:

| Word | Meaning |
|---|---|
| TMA | The Hopper bulk-copy unit, one per SM. One thread starts a copy of a whole chunk; the unit moves the bytes |
| Copy | The communication: copy CTAs on the source GPU push chunks from shared memory into the destination GPU's memory through an NCCL LSA peer pointer, with TMA bulk stores |
| Staging slot | A chunk-sized part of a copy CTA's shared memory. Copy threads fill it; then the TMA unit reads it out and sends it |
| Co-runner | The pressure workload. Main one: the **TMA store writer**, CTAs that store 64 KiB chunks from shared memory into a local 1 GiB buffer with TMA. Contrasts: the **TMA read co-runner** (bulk reads into shared memory) and the plan-07 **thread-store writer** |
| GPC | A group of SMs that shares one connection into the GPU's crossbar (map from plan 07 E7.0) |

## Why this plan

Plan 07 found no copy slowdown from local stores, local reads or local TMA stores on other SMs, at any rate up to the HBM ceiling. In all of those tests the copy itself was made of thread stores. A TMA push is a different request path, and it adds one shared part that thread stores do not use:

| Difference from plan 07 | Why it may change the answer |
|---|---|
| **The copy issues bulk stores through the SM's TMA unit** | One request is a whole chunk, not 4 bytes per thread. The unit, not the warps, holds the outstanding work, so back-pressure appears as staging slots that are not freed (a new, direct observation), not as a slower issue loop |
| **The co-runner also uses TMA** | Both flows now leave their SMs through TMA units and enter the crossbar as bulk requests |
| **A same-SM placement becomes possible** | A copy CTA can host a TMA co-runner in two spare warps, so both flows share **one** TMA unit. Plan 07 could not test this: one CTA per SM, one role per CTA |
| **The copy reads shared memory, not registers** | The fill of staging slots and the TMA unit's read compete for shared-memory bandwidth inside the SM, which a co-runner on the same SM also uses |

## Common experiment configuration

### GPUs and roles

| Item | Setting |
|---|---|
| GPUs | One allocated NVLink pair; each experiment stays on one pair; the pair is recorded in every raw record, not in configs. Both GPUs need a plan-07 SM-to-GPC map; if they lack one, run plan 07 E7.0 on them first |
| Source GPU (`I`) | Copy CTAs, co-runner CTAs (except in E1.7), idle CTAs |
| Destination GPU (`D`) | One receiver CTA; the co-runner only in E1.7; idle CTAs elsewhere. Both GPUs always launch one CTA per SM |
| Clocks and power | Unchanged; observed, not controlled |

### TMA push

| Field | Setting | Note |
|---|---|---|
| Interface | NCCL LSA peer pointer; `cp.async.bulk.global.shared::cta` started by thread 0 of each copy CTA | The same window and signals as plan 07 |
| Values | `value = hash(word index, transfer number)`, written into the staging slot by the copy threads before each chunk is sent | Same function as plan 07, so the destination check catches missing and misplaced chunks |
| **Copy CTAs** | Calibrated in E1.1 | |
| **Staging slots per CTA (chunks in flight)** | 1, 2, 3 (calibrated) | Slot *j mod K* is refilled only after the TMA unit finished reading chunk *j − K* (`cp.async.bulk.wait_group.read`) |
| **Chunk size** | 16, 32, 64 KiB (calibrated; 64 KiB default) | |
| **Fill threads per CTA** | 128, 256, 512, 1024 (calibrated; 512 default) | Threads that write the slot; thread 0 is one of them |
| Fill contrast | Slots filled once per launch (`copy_generate=0`) | Removes the fill from the loop; the destination check then catches only missing writes |
| Payload | 2048 MiB | As plan 07 |
| Traversal | Each copy CTA owns a contiguous run of whole chunks, in order | |
| Completion | Thread 0 waits for bulk-store completion, fences (async proxy, then system scope), publishes "done"; the receiver acknowledges | See [common setup](../00_common_setup.md) |

### Co-runners

| Field | TMA store writer (main) | TMA read co-runner (E1.6) | Thread-store writer (E1.8) |
|---|---|---|---|
| Operation | Thread 0 stores one 64 KiB shared-memory chunk again and again into its slice of a local 1 GiB buffer | Thread 0 reads 64 KiB chunks of a 1 GiB buffer into shared memory | All threads store 16-byte values, 64 KiB per block |
| Requests in flight | **Chunks in flight per CTA: 1, 2** (calibrated; plan 07 chose 1) | 3 chunks (plan 04, plan 07) | Set by hardware |
| CTAs | Calibrated in E1.2 (plan 07: 64 reach 97% of HBM peak) | 48 | 64 × 1024 threads |
| **Knob: target rate** | Total bytes per second of all co-runner CTAs; chunk *k* of a CTA may start only at start + *k* × interval | Same | Same, per 64 KiB block |
| Levels | 20, 40, 60, 80% of HBM peak (3352 GB/s) and no limit | Matched to the store rates achieved in E1.3 | Same as TMA store writer |
| Proof of work | Chunks per CTA; every word of the chunks written is checked | Checksum per CTA | Every vector written is checked |

### Placement

Roles are chosen by SM, from the plan-07 SM-to-GPC map, as in plan 07. **One GPC layout for every run, calibration included:** copy packed into the largest GPCs (G6, 18 SMs, first); co-runner CTAs spread over the 7 GPCs other than G6. Where the co-runner runs is the only placement knob:

| Placement | Copy | Co-runner | Shares |
|---|---|---|---|
| Other SMs, source GPU (E1.3, E1.6, E1.8) | Packed into G6 | Spread over the other 7 GPCs | Crossbar, L2, HBM |
| **Same SM (E1.5)** | As in E1.3 | **One co-runner in the last two warps of each copy CTA** | + the SM's TMA unit, shared-memory bandwidth, L1 |
| Destination GPU (E1.7) | As in E1.3 | On `D`, the 7 GPCs other than G6 | `D`'s crossbar, L2 and HBM, which also take the incoming push |

### Shared memory and block size

| Setting | Value |
|---|---|
| Physical block | 1024 threads for every CTA on both GPUs; threads without a role return at once |
| Shared memory | **196,864 bytes for every CTA in every run**, including copy-alone runs and references. It fits the read co-runner (3 × 64 KiB + barriers), a copy of up to 3 × 64 KiB slots, and a copy of 2 × 64 KiB slots plus a colocated 64 KiB TMA store writer. If E1.1 selects a copy that needs more, every run uses the larger reservation and the co-runners are calibrated again under it |

### Observations recorded in every run

Metric ids are from [00_common_setup.md](../00_common_setup.md) (A15–A17) and the [thread-store common setup](../../../thread_store/observation/00_common_setup.md) (the others).

| Observation | Id | Endpoint | Note |
|---|---|---|---|
| Copy duration and throughput | A1 | `I` timer | Primary outcome |
| CTA start, issue loop, ordering, publication, acknowledgement wait | A2–A6 | `I` timer | The issue loop ends when the last bulk store is started |
| Bulk completion wait | A17 | `I` timer | Last bulk store started → all of this CTA's bulk stores completed |
| Copy slot wait and fill time | A15, A16 | `I`, SM cycles | Fractions of thread 0's issue loop; the direct sign of TMA back-pressure |
| Issue progress checkpoints | A7 | `I` timer | Chunks started, not arrived |
| Co-runner active interval and coverage | A8 | Co-runner's GPU | Every co-runner must cover the whole copy in active transfers |
| Co-runner achieved rate and chunks | A9 | Co-runner's GPU | Requested bytes, not counted HBM bytes |
| CTA role and SM ID (both GPUs) | A10 | Both | Must match both role tables; colocated co-runners appear as a second record on the copy's SM |
| SM-to-GPC map | A12 | Both | From plan 07, checked against each GPU's UUID |
| Destination correctness, co-runner checks | A11 | `D`, co-runner GPU | Every word of every transfer |
| Operating state, NVLink counters | B2, B3 | Both | Whole-run samples |

Not collected in the primary runs: profiler timelines (B1) and counters (C1–C9). A counter pass is a separate, later step (see Procedure of E1.3).

## E1.0 · SM-to-GPC maps

### Motivation

Placement is a knob, so both GPUs of the pair need a map. Plan 07 measured identical maps on GPUs 0, 1, 4, 5, 6 and 7.

### Configuration

| Knob | Setting |
|---|---|
| Map | Reused from plan 07 (`thread_store/.../data_store_congest/configs/gpc_map/gpu<N>.json`, GPUs 0, 1, 4, 5 used here); the harness refuses a map whose UUID is not the GPU's. No new mapping runs |

### Procedure

1. Run the `smoke` phase once: every role combination with a 256 MiB payload, to check the build. Its records are not used in any result.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A12 | Same map as plan 07 (hypothesis) | Setup evidence |
| A11 | Every smoke run verifies | Build check |

## E1.1 · TMA push alone

### Motivation

The TMA push is a new workload, so it needs its own reference: the setting where more CTAs, more slots or larger chunks no longer raise throughput. One TMA CTA may send much more than one thread-store CTA (plan 07: about 35 GB/s per copy SM), so the push may reach its plateau with very few SMs. The fill of staging slots could limit a CTA before the TMA unit does; the fill contrast shows which.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Copy CTAs** | **Independent variable** | **1, 2, 4, 8, 12, 16, 24, 32, 48** |
| **Staging slots per CTA** | **Independent variable** | **1, 2, 3** (64 KiB chunks, 512 fill threads) |
| **Chunk size × fill threads/CTA** | **Independent variable (second step, at the selected CTA count and slots)** | **16, 32, 64 KiB × 128, 256, 512, 1024 threads**, plus slots filled once at each chunk size |
| Placement | Constant | Packed into the largest GPCs, G6 first (the layout of every later test) |
| Co-runner | Constant | Absent; all other SMs hold idle CTAs |
| Shared memory | Constant | 196,864 bytes |
| Repeats | Constant | 3 runs per setting (as plan 07 from 2026-10-30), rotated order; 3 warm-ups and 12 transfers per run |

### Procedure

1. Run the CTA × slots grid (`copy_sweep`). Table: rows = copy CTAs, columns = slots, cells = median throughput.
2. Stop rule (plan 07): a setting is on the plateau when its median is within 1% of the plateau value, and two successive increases add less than 1%. If the largest setting still rises, extend the grid before going on.
3. At the knee, run chunk size × fill threads and the fill-once contrast (`copy_chunk`). If filling once is clearly faster, the fill limits the CTA: record it, and choose fill threads that remove the limit.
4. Select the copy for later steps: the fewest CTAs on the plateau, which must fit in G6 (18 SMs) for the separate placement and leave room for the co-runner (copy + co-runner ≤ 124 mapped SMs). Write it to `selection.json`.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1 | Rises with CTAs and slots, then flattens near the NVLink limit seen by thread stores (about 370 GB/s), with fewer SMs than plan 07 (hypothesis) | Reference only |
| A15 | Small on the rising part; grows once the push is on its plateau, because slots wait for the store path (hypothesis) | Shows where the limit is behind the TMA unit, not in the fill |
| A16 | Large if the fill limits a CTA; falls with more fill threads (hypothesis) | Fill limit vs TMA/NVLink limit |
| A17 | A few chunks' worth of time on the plateau (hypothesis) | Reference only |

## E1.2 · Co-runners alone

### Motivation

The pressure must be known before it is applied, in this binary and under this shared-memory reservation. Plan 07 calibrated the TMA store writer (64 CTAs × 1 chunk, 97% of peak) and the reader (48 × 3) with a different binary; the colocated writer is new and is limited to the copy's few SMs, so its ceiling is much lower and must be measured.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **TMA store writer CTAs × chunks in flight** | **Independent variable** | **4, 8, 16, 24, 32, 48, 64 × 1, 2**, no limit, spread |
| **Target rate** | **Independent variable (selected writer; reader)** | **20, 40, 60, 80% of HBM peak, no limit** (reader: targets only) |
| **Colocated writer** | **Independent variable** | One TMA store writer in each copy CTA of the selected copy (same SMs), copy off, no limit |
| Copy | Constant | Off: copy CTA 0 times the windows (6 × 20 ms); the copy's SMs keep their copy role |
| Shared memory, binary | Constant | 196,864 bytes, `bin/tma_congest` |
| Repeats | Constant | 3 runs per setting; 2 warm-ups and 6 windows |

### Procedure

1. `writer_sweep`; select the fewest CTAs within 95% of the ceiling with even CTAs (slowest ÷ fastest ≥ 0.95), as in plan 07.
2. `writer_rate` and `reader_rate`: achieved against target at each level.
3. `colocated_rate`: ceiling of a writer that shares the copy's SMs. Its levels in E1.5 are 25, 50, 75% of this ceiling and no limit. Write both choices to `selection.json`.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A9 | Achieved = target below the ceiling; ceiling near plan 07's 3257 GB/s (hypothesis) | Calibration only |
| A9 (colocated) | About 50–64 GB/s per SM, as one TMA writer CTA in plan 07 (hypothesis) | Calibration only |

## E1.3 · TMA writes next to the push, in different GPCs

### Motivation

This is the main test. A fused kernel may write its results with TMA while other SMs push data to the peer GPU with TMA. With the copy in G6 and the writer in the other GPCs, the flows share only the crossbar, L2 and HBM path of the source GPU. Plan 07 found no copy slowdown here with a thread-store copy; this repeats the test with both flows made of bulk requests.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Target write rate** | **Independent variable** | **Idle writers (reference half), 20, 40, 60, 80% of HBM peak, no limit** |
| Copy | Constant | Setting from E1.1, packed into G6 |
| TMA store writer | Constant | Setting from E1.2, spread over the other 7 GPCs |
| Pairs | Constant | 3 runs per level; each run alternates idle and writing transfers (6 + 6 measured, after 3 warm-ups); odd rounds start idle |

### Procedure

1. Opening reference (writers idle).
2. At each level, one mixed run, rotating the level order between rounds (`contention`).
3. After each mixed run, the writer alone on the same SMs with the copy off, window matched to the writer's measured active time.
4. Closing references.
5. Probe cost: 3 pairs of detailed vs minimal timing (`detailed` 1 vs 0) at no limit.
6. Analysis: copy change per run = median(active) ÷ median(idle) − 1; idle IQR from all reference halves; writer rate with copy vs alone.
7. Counter pass (separate session, after the timing runs): if a slowdown appears, collect source DRAM and L2 counters (C1, C3) and NVLink bytes (C9) for one level with `ncu` application replay, as in plan 07's counter check.

### Privileged access

None for the timing runs. The counter pass needs profiler counter permission, as in plan 07.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1 | No change, as in plan 07, if the crossbar has spare capacity (hypothesis) | From E1.5: same rate, different placement |
| A15 | Grows if the store path pushes back on the copy's TMA unit (hypothesis) | Back-pressure vs completion delay (A17) |
| A17, A4, A6 | Grow first if stores are accepted but complete late (hypothesis) | Cannot separate crossbar from L2 by itself |
| A9 | Writer rate falls during the copy, if they compete (hypothesis) | Mutual effect |

## E1.4 · Shared and crowded GPCs (dropped 2026-11-10)

Dropped before any run at the user's request: GPC placement is not a question of this plan. The number is kept so E1.5–E1.8 keep their names.

## E1.5 · TMA writes from the same SMs as the push

### Motivation

This is the test plan 07 could not run. Each copy CTA hosts one TMA store writer in its last two warps, so the copy and the writer share the SM's TMA unit, its shared-memory bandwidth and its connection out of the SM. A fused kernel that pushes data and writes results from the same CTA would do this. If the TMA unit serves requests in order, a writer's 64 KiB chunks could delay the copy's chunks even at low byte rates, which the earlier placements cannot show.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Placement of the writer** | **Independent variable** | **Same SMs (one writer per copy CTA)** vs **other SMs** (the same number of writer CTAs, spread over the 7 GPCs other than G6) |
| **Target write rate** | **Independent variable** | **Idle (reference half), 25, 50, 75% of the colocated writer's ceiling from E1.2, no limit** |
| Copy | Constant | Setting from E1.1, packed into G6; fill threads must leave the last 64 threads free (at most 960) |
| Writer | Constant | TMA store writer, chunks in flight from E1.2, one 64 KiB chunk in shared memory after the copy's slots |
| Pairs | Constant | 3 runs per level and placement, alternating idle and writing transfers |

### Procedure

1. Opening references for both placements.
2. At each level, the two placements back to back, rotated between rounds (`colocated`); after each mixed run, the writer alone on the same SMs with a matched window.
3. Closing references.
4. Analysis as in E1.3, plus per-CTA A15 and A16 against the writer's chunks per CTA.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1 | A slowdown with the writer on the same SMs that is absent with the writer on other SMs at the same rate (hypothesis) | Shared TMA unit / SM vs shared crossbar |
| A15 | Grows on the same SMs: the copy's slots wait behind the writer's chunks (hypothesis) | TMA-unit queueing vs fill slowdown (A16) |
| A16 | Grows if the writer's shared-memory reads slow the fill (hypothesis) | Shared-memory bandwidth |
| A9 | Writer rate on the same SMs falls during the copy (hypothesis) | Mutual effect |

## E1.6 · Direction contrast: TMA reads instead of TMA writes

### Motivation

Plan 07 found that local reads did not slow a thread-store copy. With a TMA copy, reads and stores pass through the same TMA units only when they share an SM; on other SMs they meet only in the crossbar and memory. Repeating E1.3 with the reader shows whether the direction of local traffic matters for a TMA push.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Traffic direction** | **Independent variable** | **TMA store writer (repeated in this session) vs TMA read co-runner** |
| **Target rate** | **Independent variable** | Store levels of E1.3; read targets = store rates achieved during the copy in E1.3 |
| Read co-runner | Constant | 48 CTAs × 3 × 64 KiB, the 7 GPCs other than G6 |
| Copy, pairs | Constant | As E1.3 |

### Procedure

As E1.3 (`read_contrast`), with both directions back to back at each level.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1, A15 | Less change with reads than with writes at the same byte rate (hypothesis) | Direction of local traffic |

## E1.7 · TMA writes on the destination GPU

### Motivation

On the source GPU the push uses no source memory. On the destination GPU, incoming peer stores and local TMA stores are written into the same L2 and HBM. Plan 07 found that a thread-store writer on the destination slowed itself (−10% at no limit), not the copy. A TMA push may keep more data in flight across NVLink, so back-pressure from the destination may reach it differently.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **GPU that runs the TMA store writer** | **Independent variable** | **Destination vs source (E1.3 separate, repeated back to back)** |
| **Target write rate** | **Independent variable** | **Same levels as E1.3** |
| Receiver | Constant | 1 CTA in G6 of `D`, sharing no GPC with the writer |
| Writer | Constant | Setting from E1.2, on the same SM IDs as in E1.3 (the 7 GPCs other than G6), on `D` |
| Copy, pairs | Constant | As E1.3 |

### Procedure

1. Opening references (writer on `D` idle).
2. At each level, writer on `D` and writer on `I` back to back, rotated (`destination`). Writer on `D` is ordered around the transfer with signals: it starts before "destination ready" and stops after the acknowledgement.
3. Writer alone on `D`: the receiver CTA times the window and stops it.
4. Closing references.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1 | Little change until copy + writer approach the HBM write ceiling of `D` (hypothesis) | From E1.3: same writer and rate, different GPU |
| A17, A6 | Grow first if bulk stores are accepted but complete late at `D` (hypothesis) | Destination completion vs source back-pressure (A15) |
| A15 | Grows only if back-pressure reaches the source's TMA unit across NVLink (hypothesis) | From A17 |
| A8, A9 | Read on `D`; writer must cover the window; its rate may fall during the copy | Validity, mutual effect |

## E1.8 · Writer mechanism: thread stores next to the TMA push (optional)

### Motivation

Plan 07 E7.6 compared thread-store and TMA writers next to a thread-store copy. This completes the comparison for a TMA copy, so all four combinations of copy mechanism and writer mechanism are measured.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Writer mechanism** | **Independent variable** | **TMA store writer vs thread-store writer (64 CTAs × 1024 threads/CTA)** |
| Levels, placement, copy, pairs | Constant | As E1.3 (separate GPCs) |

### Procedure

As E1.3 (`thread_writer`), both writers back to back at each level.

### Privileged access

None.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1, A15 | Same as with the TMA writer if only the byte rate matters (hypothesis) | Request mechanism of the co-runner |

## Order of work

Status 2026-11-10 (details in the [report](README.md)):

- All steps done, including E1.8.
- Added after E1.5, not planned above: same-SM writers at 1–25% of ceiling, as TMA bulk stores and as thread stores.
- The same-SM loss grows with the writer's bytes and does not need the TMA unit.
- E1.5's expected signature ("slots wait behind the writer's chunks") holds only for the TMA writer.
- E1.7b, added 2026-11-13 at the user's request: E1.7 at no limit with more requests in flight (destination writer 64 × 1, 64 × 2, 64 × 4, 105 × 2; copy 16 × 2, 16 × 3, 48 × 2 CTAs × slots). The split of the destination's full HBM did not change.

| Step | Work | GPU time (estimate) |
|---|---|---|
| 1 | Harness `bin/tma_congest`, runner, this plan (done 2026-11-10) | None |
| 2 | E1.0 smoke run (maps reused from plan 07) | Minutes |
| 3 | E1.1 copy calibration, E1.2 co-runner calibration, on GPUs 0 → 1 and 4 → 5 at the same time | About 40 min |
| 4 | E1.3 | About 20 min |
| 5 | E1.5 same SM | About 20 min |
| 6 | E1.6 direction, E1.7 destination | About 40 min |
| 7 | E1.8 if wanted | About 20 min |
| 8 | Analysis and report in `data_store_congest/README.md` | None |

## Interpretation limits

- **Timers on thread 0 are not bulk-transfer latency.** Slot wait (A15) is the time until the TMA unit finished reading a slot; bulk completion (A17) is the unit's report for all chunks of a CTA. Neither is the time one chunk spent on NVLink.
- **Requested bytes are not counted bytes.** Plan 07 confirmed requested bytes against DRAM counters only for its two store writers alone, with a different binary.
- **SM cycles change with clocks.** A15 and A16 are reported as fractions of the same thread's loop cycles, so a clock change during a run scales both parts alike.
- **The same-SM test cannot name the shared part.** A slowdown there points to the TMA unit, shared-memory bandwidth or the SM's connection out, together. Separating them would need a writer that uses shared memory without TMA (a later variant).
- **No slowdown is a result.** It means these flows did not interfere at these rates and placements, not that they cannot.
