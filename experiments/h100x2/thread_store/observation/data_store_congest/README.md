# Data-store congestion on the issuing GPU

**Status (2026-10-30): E7.0–E7.2 done (SM-to-GPC map, copy alone, writer alone). E7.3 onwards (copy and writer together) not yet run.**

[Plan](07_data_store_congest.md) · [Common setup and metric ids](../00_common_setup.md) · [Related plan 04 and its results](../shared_resources/README.md) · [Harness](../../common/README.md#store-congestion-harness-plan-07) · [Run configurations](configs/)

**Question.** When some SMs of the source GPU store data to the peer GPU over NVLink while other SMs store data to local memory, do the two store flows slow each other down, and does it matter whether they share a GPC?

**Answer so far.** The two flows have not run together yet. This report sets up the parts that test needs. (1) The SM-to-GPC map: 124 of the 132 SMs fall into 8 GPCs of 8, 16 (five GPCs) and 18 (two GPCs) SMs; the same map was found on GPUs 0, 1, 4 and 5. (2) The store-only copy reaches its throughput plateau of about **371 GB/s** with **16 CTAs**, and not with fewer: one SM sends at most about 35 GB/s of 4-byte peer stores, and more threads per CTA do not change that. The selected copy reference is **16 CTAs × 512 threads/CTA: 367.8 GB/s**. It keeps that throughput when packed into one GPC (368.2 GB/s), so it fits the "copy and writer in different GPCs" test. A planned knob, "stores in flight per thread", turned out to control only the stores per loop step, never helped, and was removed. (3) The writer (local 16-byte stores) gives about **50 GB/s per SM** and saturates HBM writes at about **3277 GB/s (98% of peak)** from 64 SMs. With **64 CTAs × 1024 threads/CTA** spread over all GPCs, the rate control holds exactly at 20–80% of peak. When the same 64 writer CTAs are packed into 4 GPCs, they reach only **2819 GB/s (84% of peak)**: each full GPC delivers about 690–810 GB/s, less than its SMs could issue. A GPC therefore has a limit on store traffic, but it is about twice what the copy needs.

| Word used in this report | Metric id ([common setup](../00_common_setup.md)) |
|---|---|
| Copy throughput: payload bytes ÷ time from first copy CTA start to the acknowledgement | A1 |
| Issue-loop time, ordering interval, acknowledgement wait (per copy CTA, medians) | A3, A4, A6 |
| Writer active interval and coverage of the copy window | A8 |
| Achieved write rate: 64 KiB blocks written ÷ writer active time | A9 |
| Role and physical SM of each CTA | A10 |
| Full destination check after every transfer; writer region check after every launch | A11 |
| SM-to-GPC map | A12 |

## Words and setup shared by all sections

| Item | Setting |
|---|---|
| GPUs | Source → destination: **4 → 5** (2026-10-29) and **0 → 1** (2026-10-30). All four GPUs have the identical SM-to-GPC map (E7.0); at the user's request the two pairs are treated as the same hardware. Each raw record names its pair |
| Copy | Store-only: each copy thread computes a value from the element index and stores it (4 bytes) into the destination GPU through the NCCL LSA peer pointer. No source loads. One store per thread per loop step. Payload 2048 MiB, CTA-local stride, each CTA's part aligned to 128 bytes. Completion: all writing threads fence, each CTA publishes "done", the destination acknowledges |
| Writer | CTAs on the source GPU store 16-byte values into a local 1 GiB buffer, 64 KiB blocks, each CTA in its own slice. Optional rate limit: block *k* of a CTA may start only at start + *k* × interval |
| Placement | One CTA per SM (cooperative launch, 132 CTAs). The host chooses each CTA's role by its SM: **spread** = one SM per GPC in turn; **packed** = all SMs of the largest GPC first. Other SMs hold idle CTAs. The 8 unmapped SMs are used only when a setting needs more than 124 role CTAs |
| Repeats | Copy: 3 warm-ups + 12 measured transfers per run, run value = median of the 12. Writer: 2 warm-ups + 5 windows of 20 ms, run value = median of the 5. Settings in rotated order. 5 runs per setting until 2026-10-30, 3 after (user's request); tables give the run count |
| Not controlled | Clocks and power unchanged (device snapshots before and after every run). Machine shared with other users; GPUs were idle when each run started (checked under a per-device lock) |
| Not collected | Profiler passes (Tier B timeline, Tier C counters) were not run for E7.0–E7.2. Writer rates are logical store bytes, not measured DRAM bytes |

## SM-to-GPC map

### Motivation

The later tests put copy CTAs and writer CTAs in the same GPC or in different GPCs, so we need to know which SMs share a GPC. CUDA has no instruction that tells a CTA its GPC; a CTA can only read its SM ID.

### Setup

GPUs 4, 5 (2026-10-29) and 0, 1 (2026-10-30), one GPU at a time. Program `common/bin/gpc_probe`. Thread-block clusters of 2, 4, 8 and 16 CTAs; shared memory per CTA above half an SM, so each SM runs at most one CTA. 20 launches per cluster size, each launch 8 waves of the maximum number of clusters that fit at once; each CTA stays a random time (10–90 µs) and does not wait for its cluster.

### Method

CUDA runs all CTAs of a cluster inside one GPC. Each CTA records its cluster number and its SM ID. Two SMs that ever ran CTAs of the same cluster are put in the same group (union-find). Clusters of 2 always landed on SM pairs 2t, 2t+1 (one TPC), so they do not link TPCs; an SM that never ran a CTA of a cluster with 4 or more CTAs is marked unmapped. GPC numbers follow each group's smallest SM ID.

### Results

![SM-to-GPC map on GPUs 0, 1, 4 and 5](results/2026-10-29/figures/gpc_map.png)

*Each box is one SM (SM ID and GPC). Grey: unmapped. All four GPUs give the same map.*

| GPC | SMs | SM IDs |
|---|---|---|
| G0 | 8 | 0, 1, 16, 17, 32, 33, 48, 49 |
| G1 | 16 | 2, 3, 18, 19, 34, 35, 50, 51, 64, 65, 78, 79, 92, 93, 106, 107 |
| G2 | 16 | 4, 5, 20, 21, 36, 37, 52, 53, 66, 67, 80, 81, 94, 95, 108, 109 |
| G3 | 16 | 6, 7, 22, 23, 38, 39, 54, 55, 68, 69, 82, 83, 96, 97, 110, 111 |
| G4 | 16 | 8, 9, 24, 25, 40, 41, 56, 57, 70, 71, 84, 85, 98, 99, 112, 113 |
| G5 | 16 | 10, 11, 26, 27, 42, 43, 58, 59, 72, 73, 86, 87, 100, 101, 114, 115 |
| G6 | 18 | 12, 13, 28, 29, 44, 45, 60, 61, 74, 75, 88, 89, 102, 103, 116, 117, 120, 121 |
| G7 | 18 | 14, 15, 30, 31, 46, 47, 62, 63, 76, 77, 90, 91, 104, 105, 118, 119, 122, 123 |
| unmapped | 8 | 124–131 |

| Cluster size | Clusters that fit at once (CUDA occupancy query) | SMs used | CTAs that ran on SMs 124–131 |
|---|---|---|---|
| 2 | 66 | 132 | about as many as on other SMs |
| 4 | 30 | 120 | 0 |
| 8 | 15 | 120 | 0 |
| 16 | 7 | 112 | 0 |

### Interpretation

- **The map is a clean round-robin.** TPC *t* (SMs 2t, 2t+1) goes to GPC *t* mod 8 for SMs 0–119; SMs 120–123 add one TPC each to G6 and G7.
- **SMs 124–131 cannot be placed.** The GPU never runs a CTA of a cluster of 4 or more on them, also not with random CTA lifetimes. The occupancy counts above fit exactly a layout where these 8 SMs are left out of larger clusters, so this is how the CUDA runtime treats them, not chance. Which GPC they belong to is unknown (one guess: G0, which would then have 16). They are labelled unmapped; copy + writer must stay ≤ 124 CTAs when placement matters.

## Store-only copy alone

### Motivation

The copy is the communication under test, so it needs a reference: the setting where more issuing CTAs or threads no longer raise throughput. The plan expected a plateau near the plan-04 value (about 370 GB/s), possibly with fewer threads, because this copy does no source loads. The copy must also leave SMs for the writer: with one CTA per SM, copy CTAs + writer CTAs ≤ 124 mapped SMs.

### Setup

| Knob | Role | Values |
|---|---|---|
| **Copy CTAs × threads/CTA** | **Independent variable** | **Plan grid:** CTAs 16, 32, 48, 64, 96, 128 at 512 and 1024 threads/CTA; threads/CTA 256, 512, 768, 1024 at 64 and 128 CTAs. **Added floor grid:** CTAs 1, 2, 4, 8, 16 × threads/CTA 128, 256, 512, 1024 |
| **Placement** | **Independent variable (second step)** | **Spread** over all 8 GPCs vs **packed** into the fewest GPCs, at 16 × 512, 16 × 256 and 32 × 512 |
| Copy loop | Constant | One 4-byte peer store per thread per loop step |
| Writer | Constant | Absent (all other SMs hold idle CTAs) |
| Payload, stride, completion | Constant | 2048 MiB, CTA-local, fence + publish + acknowledgement |

Configs: [configs/copy_alone/](configs/copy_alone/), named `copy_c<CTAs>_t<threads/CTA>_<placement>_R<run>.cfg`.

### Method

1. All results in this section come from the final harness build (2026-10-30, GPUs 0 → 1, 3 runs per setting).
2. Stopping rule, as in plan 04: a setting is on the plateau when its median is within 1% of the plateau value (median of all settings with 32 or more CTAs).
3. Placement: packed and spread runs of the same settings, same build.
4. **Earlier measurements, not joined with these.** The first E7.1 runs (2026-10-29/30, GPUs 4 → 5 and 0 → 1, 3–5 runs) used a build with a fourth knob, "stores per loop step" (1, 2, 4, 8). It was removed (see [Removed knob](#removed-knob-stores-per-loop-step)), which changed the compiled copy loop, so the copy grid was remeasured. The earlier build reached the same plateau and the same 16-CTA knee; its per-SM rates below the plateau differ (table below).
5. A first plan-grid attempt lost about 30% at 48 and 96 CTAs because copy parts were not 128-byte aligned; it was fixed and rerun. The old records are kept under `results/2026-10-29/rejected/misaligned_copy_parts/`.

### Results

![Copy throughput against CTAs and threads per CTA](results/2026-10-29/figures/copy_sweep.png)

*Store-only copy alone, spread over all GPCs, 2048 MiB, one 4-byte store per loop step. Left: CTA scaling at fixed threads/CTA (log x). Right: thread scaling at 16, 64 and 128 CTAs. Points: median of 3 runs; bars: min–max (smaller than the marker). Lines join only points of the same threads/CTA (left) or CTA count (right).*

Copy throughput in GB/s (median of 3 runs; — = not measured). Rows: copy CTAs; columns: threads/CTA.

| Copy CTAs | 128 threads/CTA | 256 threads/CTA | 512 threads/CTA | 768 threads/CTA | 1024 threads/CTA |
|---|---|---|---|---|---|
| 1 | 19.6 | 28.8 | 34.9 | — | 34.8 |
| 2 | 39.3 | 57.5 | 69.8 | — | 69.5 |
| 4 | 78.4 | 114.9 | 138.7 | — | 138.8 |
| 8 | 156.8 | 229.6 | 276.6 | — | 277.4 |
| 16 | 313.2 | **370.5** | **367.8** | — | 367.4 |
| 32 | — | — | 370.7 | — | 372.0 |
| 48 | — | — | 371.1 | — | 371.6 |
| 64 | — | 371.3 | 370.3 | 368.6 | 372.1 |
| 96 | — | — | 370.9 | — | 371.2 |
| 128 | — | 371.1 | 370.8 | 370.2 | 371.4 |

Plateau (median of all settings with 32 or more CTAs): **371.1 GB/s**; within 1%: ≥ 367.4 GB/s.

Time phases near the plateau (medians over CTAs and runs; phases do not add up to the transfer time):

| Copy CTAs × threads/CTA | Runs | Copy throughput GB/s (min–max) | Transfer µs | Issue loop µs | Ordering µs | Acknowledgement wait µs |
|---|---|---|---|---|---|---|
| 8 × 1024 | 3 | 277.4 (277.4–277.5) | 7742 | 7721 | 4.1 | 14.7 |
| 16 × 128 | 3 | 313.2 (313.2–313.3) | 6856 | 6834 | 4.1 | 15.7 |
| 16 × 256 | 3 | 370.5 (370.1–370.6) | 5796 | 5720 | 4.9 | 69.3 |
| **16 × 512** | 3 | **367.8 (367.7–368.1)** | 5839 | 5779 | 5.6 | 52.0 |
| 16 × 1024 | 3 | 367.4 (367.4–367.7) | 5845 | 5792 | 6.1 | 43.9 |
| 32 × 512 | 3 | 370.7 (370.5–371.4) | 5793 | 5743 | 7.0 | 37.4 |
| 64 × 512 | 3 | 370.3 (370.0–370.5) | 5799 | 5042 | 8.7 | 746.0 |
| 128 × 512 | 3 | 370.8 (370.7–370.9) | 5792 | 5527 | 10.4 | 254.6 |
| 128 × 1024 | 3 | 371.4 (371.4–371.6) | 5782 | 5509 | 10.4 | 258.2 |

Spread versus packed placement:

![Copy throughput, spread over all GPCs vs packed into the fewest GPCs](results/2026-10-29/figures/copy_placement.png)

*Bars: median of 3 runs; whiskers: min–max. Packed GPCs named under each pair.*

| Copy CTAs × threads/CTA | Placement | GPCs used | Runs | Copy throughput GB/s (min–max) | Issue loop µs | Acknowledgement wait µs |
|---|---|---|---|---|---|---|
| 16 × 512 | spread | all 8 | 3 | 367.8 (367.7–368.1) | 5779 | 52.0 |
| 16 × 512 | packed | G6 | 3 | 368.2 (368.0–368.3) | 5809 | 16.1 |
| 16 × 256 | spread | all 8 | 3 | 370.5 (370.1–370.6) | 5720 | 69.3 |
| 16 × 256 | packed | G6 | 3 | 367.6 (367.5–367.6) | 5806 | 24.7 |
| 32 × 512 | spread | all 8 | 3 | 370.7 (370.5–371.4) | 5743 | 37.4 |
| 32 × 512 | packed | G6, G7 | 3 | 370.4 (370.3–370.6) | 5763 | 22.7 |

Full tables: [copy_sweep_summary.csv](results/2026-10-29/copy_sweep_summary.csv), [copy_plateau_summary.csv](results/2026-10-29/copy_plateau_summary.csv), [copy_placement_summary.csv](results/2026-10-29/copy_placement_summary.csv).

### Interpretation

- **The plateau is about 371 GB/s, the same as the plan-04 copy with loads.** Every setting with 32 or more CTAs lies between 368.6 and 372.1 GB/s. Removing the source loads did not raise the ceiling; it lowered the cost of reaching it: plan 04 needed 128 CTAs × 512 threads/CTA with 8 loads in flight per thread.
- **The CTA count, not the thread count, decides whether the copy reaches the plateau.** Below 16 CTAs, throughput grows in proportion to the CTA count: one copy SM sends at most about 35 GB/s of 4-byte peer stores, reached at 512 threads/CTA; 1024 threads/CTA add nothing. 8 CTAs stop at 277 GB/s with any thread count. 16 CTAs is the fewest that reach the plateau.
- **The per-SM limit depends on the compiled loop, not only on the hardware.** The earlier build ran the same source loop as the same 22 instructions per store in a different order: 37.7 GB/s per SM at 1024 threads/CTA (8% more), but only 14.2 GB/s at 128 threads/CTA (28% less than now). What limits one SM (instruction issue, waiting between dependent instructions, or the store path) was not measured; an Nsight Compute pass at 1 CTA would show it. The plateau and the 16-CTA knee were the same in both builds.
- **Where the time goes changes, the total does not.** At 16–32 CTAs the issue loop lasts almost the whole transfer and the acknowledgement wait is short (15–70 µs). At 64 CTAs the issue loop ends about 750 µs earlier and the copy CTAs then wait about 750 µs for the acknowledgement. More CTAs let the stores leave the SMs earlier and queue further along the path; the time to deliver 2 GiB stays the same. The issue-loop time alone is therefore not a measure of copy progress.
- **Packing the copy into one GPC costs at most 0.8%.** 16 × 512 packed into G6 gives 368.2 GB/s against 367.8 spread (+0.1%); 16 × 256 loses 0.8%; 32 × 512 in two GPCs loses 0.1%. One GPC can carry the full copy.
- **Selected copy reference: 16 CTAs × 512 threads/CTA (367.8 GB/s spread, 368.2 GB/s packed into G6).** It uses the fewest SMs on the plateau (0.9% below the plateau value), which leaves 108 mapped SMs for the writer. 512 rather than 256 threads/CTA was chosen for headroom: at 256 threads/CTA, 16 SMs can send at most about 460 GB/s (16 × 28.8), 24% above what the link delivers; at 512 threads/CTA, about 560 GB/s (16 × 34.9), 51% above. With more headroom, a slowdown of the copy SMs under pressure is less likely to show as lower copy throughput, so a copy slowdown points to the shared path, not to the copy's own SMs. Threads/CTA costs the writer nothing: the writer is limited by SMs, not by threads. 32 × 512 (370.7 GB/s; G6 and G7 when packed) is a second reference with more supply if needed.

### Removed knob: stores per loop step

The first build had a knob `copy_batch` = N (1, 2, 4, 8), first called "stores in flight per thread". Each copy thread wrote N stores per loop step, at the same addresses and in the same order for every N. A store does not wait for a reply, so N could not change how many stores are in flight; it only changed the instruction sequence. The compiler kept one bounds-check branch per store, so N > 1 added about 7 instructions per store (22 → 29). The knob was removed on 2026-10-30 at the user's request.

Copy throughput in GB/s with the earlier build (spread, median of 3–5 runs):

| Copy CTAs × threads/CTA | 1 store per step | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 × 128 | 14.2 | 8.8 | 9.4 | 9.8 |
| 1 × 1024 | 37.7 | 30.1 | 31.0 | 32.0 |
| 16 × 256 | 369.5 | 249.3 | 271.4 | 286.2 |
| 16 × 512 | 368.8 | 335.1 | 364.1 | 364.4 |
| 16 × 1024 | 366.9 | 364.7 | 365.0 | 365.8 |
| 128 × 1024 | 369.8 | 370.6 | 370.5 | 370.3 |

Over all 136 earlier settings, N > 1 never raised throughput by more than 0.3% and lowered it by up to 41% (2 CTAs × 512 threads/CTA, N = 2). All rows: [copy_loop_step_summary.csv](results/2026-10-29/copy_loop_step_summary.csv).

## Writer alone

### Motivation

The pressure must be known before it is applied: how fast the writer can store to local HBM, how many SMs it needs, whether the rate limit gives the requested rate, and whether packing the writer into few GPCs lowers its ceiling (the plan's hypothesis of a limit at each GPC's connection into the crossbar).

### Setup

| Knob | Role | Values |
|---|---|---|
| **Writer CTAs × threads/CTA** | **Independent variable (step 1)** | **CTAs 4, 8, 16, 24, 32, 48, 64, 96, 128 at 1024 threads/CTA; threads/CTA 256, 512, 1024 at 32 and 64 CTAs**; no rate limit; spread |
| **Target write rate** | **Independent variable (step 2)** | **670, 1341, 2011, 2682 GB/s (20, 40, 60, 80% of the 3352 GB/s HBM peak) and no limit**, at 64 CTAs × 1024 threads/CTA |
| **Placement** | **Independent variable (step 3)** | **Spread** over all 8 GPCs vs **packed** into G6, G7, G1, G2 (18 + 18 + 16 + 12 SMs), same rates |
| Copy | Constant | Off. One copy-role CTA (on SM 0, G0) only times the 20 ms windows; all other SMs hold idle CTAs |

Configs: [configs/writer_alone/](configs/writer_alone/), named `writer_w<CTAs>_t<threads/CTA>_<rate>_<placement>_R<run>.cfg`.

### Method

Step 1 on GPUs 4 → 5 (5 runs). Steps 2 and 3 on GPUs 0 → 1 (3 runs; the no-limit spread point reuses step 1). These runs used the build that still had the copy knob; the writer code was not changed. One run each with the final build gave 794 GB/s (16 CTAs, −1.1%), 3158 GB/s (64 CTAs spread, −1.1%), 2811 GB/s (64 packed, no limit, −0.3%) and 2656 GB/s (64 packed, 80% target, −0.2%), so the writer results stand; E7.3 measures its own writer-alone controls. The selection rule was set before step 2: the fewest writer CTAs at 1024 threads/CTA within 95% of the no-limit ceiling (as in the plan-04 read calibration). The write rate counts logical store bytes over the writer's active interval; all writer CTAs covered the whole measured window in every run.

### Results

![Writer rate against writer CTAs and threads per CTA](results/2026-10-29/figures/writer_sweep.png)

*Writer alone, no limit, spread. Left: writer CTAs at 1024 threads/CTA. Right: threads/CTA at 32 and 64 CTAs. Median of 5 runs; bars min–max (smaller than the marker).*

| Writer CTAs × threads/CTA | Runs | Write rate GB/s (min–max) | % of HBM peak | Per CTA GB/s | Slowest ÷ fastest CTA |
|---|---|---|---|---|---|
| 4 × 1024 | 5 | 201 (199–201) | 6.0 | 50.2 | 1.00 |
| 8 × 1024 | 5 | 401 (398–401) | 12.0 | 50.2 | 1.00 |
| 16 × 1024 | 5 | 803 (797–803) | 24.0 | 50.2 | 1.00 |
| 24 × 1024 | 5 | 1204 (1195–1204) | 35.9 | 50.2 | 1.00 |
| 32 × 256 | 5 | 1733 (1719–1734) | 51.7 | 54.2 | 0.99 |
| 32 × 512 | 5 | 1692 (1679–1692) | 50.5 | 52.9 | 1.00 |
| 32 × 1024 | 5 | 1606 (1594–1606) | 47.9 | 50.2 | 1.00 |
| 48 × 1024 | 5 | 2409 (2391–2409) | 71.9 | 50.2 | 1.00 |
| 64 × 256 | 5 | 3278 (3259–3278) | 97.8 | 51.2 | 0.85 |
| 64 × 512 | 5 | 3276 (3271–3279) | 97.7 | 51.2 | 0.92 |
| **64 × 1024** | 5 | **3194 (3172–3194)** | **95.3** | 49.9 | 0.99 |
| 96 × 1024 | 5 | 3277 (3267–3279) | 97.8 | 34.1 | 0.52 |
| 128 × 1024 | 5 | 3274 (3272–3276) | 97.7 | 25.6 | 0.56 |

![Achieved against target write rate, spread and packed](results/2026-10-29/figures/writer_rate.png)

*64 writer CTAs × 1024 threads/CTA. Points: median of 3 runs; dotted lines: no limit; dashed: achieved = target.*

| Placement | GPCs | Target GB/s (% of peak) | Runs | Achieved GB/s (min–max) | % of peak | Achieved ÷ target | Slowest ÷ fastest CTA |
|---|---|---|---|---|---|---|---|
| spread | all 8 | 670 (20%) | 3 | 670 (670–670) | 20.0 | 100.00% | 1.00 |
| spread | all 8 | 1341 (40%) | 3 | 1341 (1341–1341) | 40.0 | 100.00% | 1.00 |
| spread | all 8 | 2011 (60%) | 3 | 2011 (2011–2011) | 60.0 | 100.00% | 1.00 |
| spread | all 8 | 2682 (80%) | 3 | 2682 (2682–2682) | 80.0 | 100.00% | 1.00 |
| spread | all 8 | no limit | 5 | 3194 (3172–3194) | 95.3 | — | 0.99 |
| packed | G1, G2, G6, G7 | 670 (20%) | 3 | 670 (670–670) | 20.0 | 100.00% | 1.00 |
| packed | G1, G2, G6, G7 | 1341 (40%) | 3 | 1341 (1341–1341) | 40.0 | 100.00% | 1.00 |
| packed | G1, G2, G6, G7 | 2011 (60%) | 3 | 2011 (2011–2011) | 60.0 | 100.00% | 1.00 |
| packed | G1, G2, G6, G7 | 2682 (80%) | 3 | 2660 (2658–2660) | 79.3 | 99.17% | 0.97 |
| packed | G1, G2, G6, G7 | no limit | 3 | 2819 (2818–2822) | 84.1 | — | 0.80 |

Write rate per GPC (sum of per-CTA rates, median over windows and runs), no limit:

| Writer setting | GPC: SMs used → GB/s (GB/s per SM) |
|---|---|
| 64 × 1024, packed | G1: 16 → 686 (42.9) · G2: 12 → 601 (50.1) · G6: 18 → 810 (45.0) · G7: 18 → 724 (40.2) |
| 64 × 1024, spread | G0: 7 → 351 (50.1) · G1: 9 → 449 (49.9) · G2–G7: 8 each → 399 (49.9) |
| 128 × 1024, spread | G0: 7 → 239 (34.1) · G1–G5: 16 each → 408–410 (25.5) · G6, G7: 18 each → 410, 408 (22.8) · unmapped: 5 → 171 (34.1) |

Full tables: [writer_sweep_summary.csv](results/2026-10-29/writer_sweep_summary.csv), [writer_rate_summary.csv](results/2026-10-29/writer_rate_summary.csv).

### Interpretation

- **One writer SM stores about 50 GB/s, and HBM writes saturate at about 3277 GB/s (98% of peak).** The rate grows in proportion to the CTA count up to 48 CTAs and flattens from 64. This is close to peak for writes; it is a logical store rate, and a DRAM counter check (ncu, counters are open on this machine) has not been done.
- **Selected writer setting: 64 CTAs × 1024 threads/CTA.** It is the fewest CTAs at 1024 threads/CTA within 95% of the ceiling (3194 GB/s; 48 CTAs give only 72%). Its CTAs run at equal rates (slowest ÷ fastest 0.99). 64 + 16 copy CTAs = 80 ≤ 124 mapped SMs.
- **The rate limit gives exactly the requested rate** at 20–80% of peak when spread (within 0.01%), with all CTAs equal.
- **Packed writers reach a lower ceiling: 2819 GB/s (84%) against 3194 GB/s (95%) spread, with the same 64 CTAs.** Inside a full GPC, each SM drops from 50 to 40–45 GB/s, while the 12 SMs used in G2 keep 50 GB/s. A full GPC delivers about 690–810 GB/s of stores. This supports the plan's hypothesis of a limit at each GPC's connection, although the GPCs do not share one exact cap, and the packed writer falls 0.8% short of the 80% target. The copy needs about 370 GB/s, about half of one GPC's limit, which agrees with the small cost of packing the copy.
- **At saturation, HBM write bandwidth is shared per GPC, not per SM.** With 128 writer CTAs, every full GPC gets about 409 GB/s whether it has 16 or 18 writer SMs. This is a hint about how the crossbar arbitrates and matters for E7.3, where the copy's GPC competes with the writer's GPCs.

## Settings for the next steps

| Role | Setting | Alone |
|---|---|---|
| Copy (reference) | 16 CTAs × 512 threads/CTA, 2048 MiB; packed into G6 for "different GPCs" | 368.2 GB/s packed, 367.8 GB/s spread |
| Copy (more supply, optional) | 32 CTAs × 512 threads/CTA; packed into G6, G7 | 370.4 GB/s packed |
| Writer | 64 CTAs × 1024 threads/CTA at 20/40/60/80% of peak and no limit, in the GPCs not used by the copy | 670–2682 GB/s exactly; no limit depends on placement (2819–3194 GB/s) |

Stored in [results/2026-10-29/selection.json](results/2026-10-29/selection.json). When the copy is packed into G6, the writer gets G0–G5 and G7 (106 mapped SMs); its no-limit ceiling there must be measured as part of E7.3, since it depends on which GPCs it uses.

## Deviations from the plan

1. **GPUs:** 4 → 5 and 0 → 1 instead of one pair from 0, 1, 6, 7 (user's allocation). All four GPUs have the same GPC map; the user asked to treat both pairs as the same.
2. **Unmapped SMs 124–131:** placement uses 124 mapped SMs; 128-CTA settings also use 4 unmapped SMs.
3. **Copy floor grid added** (1–16 CTAs, 128–1024 threads/CTA), because the plan grid was flat from its first point.
4. **3 runs per setting instead of 5** from 2026-10-30 (user's request). All copy results come from 3-run settings.
5. **"Stores in flight per thread" removed** from the copy (user's request, 2026-10-30): it could only set the stores per loop step. The copy grid was remeasured with the build without it; see [Removed knob](#removed-knob-stores-per-loop-step).
6. **Writer-alone runs use one copy-role CTA as the window timer;** the copy itself is off.
7. **No profiler passes** for these calibration steps.

## Session log

| Date (UTC) | GPUs | What ran | Records |
|---|---|---|---|
| 2026-10-29 | 4, 5 | Harness built. E7.0 probe (20 launches per cluster size). Copy plan grid, first attempt rejected (misaligned parts), rerun after the fix (64 settings × 5). Writer CTA/thread grid (13 settings × 5) | `results/2026-10-29/{gpc_map,copy_alone,writer_alone}/raw/`, `session_sweeps.log` |
| 2026-10-29/30 | 4, 5 | Copy floor grid rounds 1–2 and part of 3 (stopped when the user moved the work) | `session_floor.log` |
| 2026-10-30 | 0, 1 | E7.0 probe on GPUs 0, 1 (same map). Host-only rebuild to allow GPUs 0, 1; device code checked identical (SASS). Rest of the floor grid (stopped after round 3 at the user's request); writer rate, spread and packed; copy spread vs packed | `session_gpu0_1.log`, `session_copy_placement.log`, `build_provenance.json` |
| 2026-10-30 | 0, 1 | `copy_batch` removed from the harness. Check runs (copy 16 × 512, 8 × 1024, 1 × 1024; writer 16 and 64 CTAs, spread and packed). Copy plan grid, floor grid and placement remeasured with the final build (34 settings + 3 packed, 3 runs each) | `refactor_check/raw/`, `session_no_knob.log`, `build_provenance.json` |

Raw run records, device snapshots and logs are kept only in the local `results/2026-10-29/` directory (ignored by git); without them, rerun the phases. Reproduce: `make` in `common/`, then `python3 scripts/run_gpc_map.py --gpus S,D`, `build_gpc_map.py`, and `run_store_congest.py PHASE --gpus S,D` for `copy_sweep`, `copy_floor`, `writer_sweep`, `writer_rate`, `writer_placement`, `copy_placement`; then `analyze_store_congest.py` with the plotting Python.
