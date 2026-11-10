# 01 · Mixing `st_multicast` and `ld_reduce` on one GPU

This plan covers the problem-matrix cell **NVLS-based collective implementation × shared-resource contention**. It reproduces section 3.4 of [Shouxu's scale-up switch report](../../../../../notes/references/scaleup-switch%20(shouxu%20expr%20report).pdf). The report titles that section "Hypothesis 3", which is a typo for hypothesis 4.

Words, harness design and metric IDs are in the [common setup](../00_common_setup.md). Status and results are in the [report](README.md).

**Note (2026-11-01): scope cut, agreed with the user.** The plan was larger than the question needs. These parts are dropped: E1.1 rounds 2–3 (one round of 60 runs already chose the setting; E1.2 and E1.3 repeat the chosen setting 6 times per arm), E1.4b, the `ncu` pass in E1.5 (each run's whole-run link bytes are used instead), and E1.6. The sections below stay as written, as the record of what was first planned.

**Note (2026-11-01): harness differs from the plan's launch.** Each run launches 132 CTAs, one per SM. CTAs without a role exit at once, instead of 64 CTAs with idle ones polling. Each run has a fixed time window. Each CTA reserves over half of the SM's shared memory, which NCCL's kernels do not do.

**Common question.** When each GPU issues `st_multicast` and `ld_reduce` at the same time, does busBW fall by about 20% compared with one operation alone, while plain loads and stores mixed in the same way do not fall?

## What the report claims

The report gives two tables. The ld/st table's third column says "st_multicast" and "ld_reduce" for plain reads and writes; we treat this as a copy error and use the "ld only" and "st only" column.

| Report table | Work | CTAs | Report busBW, GB/s |
|---|---|---|---|
| Multimem | `st_multicast` only (AllGather) | 32 | 700 |
| Multimem | `ld_reduce` only (ReduceScatter) | 32 | 700 |
| Multimem | Both, on separate CTAs (AllGather + ReduceScatter) | 32 + 32 | 560 |
| Multimem | Both, on the same CTAs (AllReduce: `ld_reduce` then `st_multicast`) | 32 | 550 |
| ld/st | Plain load only | 32 | 662.3 |
| ld/st | Plain store only | 32 | 668.1 |
| ld/st | Plain load + plain store | 32 + 32 | 663 |

The report concludes three things:

1. Mixing the two multimem operations on one GPU lowers busBW from about 700 to about 550–560 GB/s.
2. It does not matter whether the same CTAs or separate CTAs issue them.
3. Plain loads and stores do not show this.

It then offers a possible cause, a limit on in-flight multimem requests shared by both operations. The report did not test this cause.

**The reproduction target is the three measured claims, not the cause.**

## Three problems to settle first

### The report's numbers cannot come from this H100 link

The report shows 767.8 GB/s for one plain-load transfer. Our H100 link is 18 × 26.562 ≈ 478 GB/s per direction, and our plain-store copy reaches about 371 GB/s. So the report was measured on faster hardware. The report does not say which. The numbers fit a link of twice our speed, such as NVLink 5, but this is only arithmetic and must be confirmed with the author. If the report used an sm_100 GPU, NCCL's kernels also differ: TMA variants run only when `minCompCap >= 100`.

So we test the **ratios**, not the absolute values. If the ratios carried over (a hypothesis, not a prediction from any model), we would see roughly 350–385 GB/s alone and 280–305 GB/s mixed.

**Note (2026-11-04).** The author told the user that the report was measured on GB300 with fifth-generation NVLink. GB300 is a Blackwell GPU, so NCCL's TMA kernel variants can run there and not on our H100. Our harness and NCCL runs use the non-TMA multimem kernels.

**Note (2026-11-01).** The E1.0 kernel check printed a busBW above this mixed range, at 4 CTAs chosen by NCCL, with logging on and no matched reference; see the [report](README.md). Do not use this rough range to judge results. The reproduction test below uses ratios and is unchanged.

### H100 multimem may not work in this VM

Multimem needs NVLS multicast, which NCCL reads from `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`. The attribute has not been read on this machine. The GPUs are in pass-through mode and no NVSwitch device is visible in the VM. E1.0 decides this before any other work.

**Note (2026-11-01).** Settled for GPUs 4–7: E1.0 found multimem supported and working; see the [report](README.md).

### The report does not define the "560" and "663" numbers

The report does not say how it combines two roles into one busBW. Each role has its own rate, and the two possible readings mean very different things for the links. Here is a simple traffic model at n = 4. It assumes a multicast store leaves the GPU once and arrives from 3 peers at each receiver, an `ld_reduce` reply arrives once and each GPU serves 3 readers, and request packets are ignored. **This model is not verified; E1.5 tests it.**

| Work and reported busBW | How to read the number | Bytes in per GPU, GB/s | Bytes out per GPU, GB/s |
|---|---|---|---|
| `st_multicast` only, 700 | | 700 | 233 |
| `ld_reduce` only, 700 | | 233 | 700 |
| AllReduce, 550 | Formula known: 550 / 1.5 = 367 GB/s per rank | 367 | 367 |
| Both on separate CTAs, 560 | **Each role** runs at 560 | 747 | 747 |
| Both on separate CTAs, 560 | **The two roles sum** to 560 (280 each) | 373 | 373 |

These two readings lead to opposite conclusions:

- **If each role runs at 560:** each link direction carries about 747 GB/s, close to the 768 GB/s the report measured for plain loads. The links may simply be full. No shared in-flight limit is needed to explain it. AllReduce would then not be the same effect, because its links carry only 367 GB/s.
- **If the roles sum to 560:** the links are only half full, like AllReduce, so something other than link capacity limits the rate. This fits the report's reading.

The AllReduce row is not ambiguous because its formula is known. So AllReduce is the primary test of this plan. For the row with separate CTAs, we report both readings and let the link counters decide.

**Note (2026-11-07): the AllReduce formula was not known.** The table above and the sentence before this note assumed that the report's numbers are nccl-tests busBW (550 / 1.5). The report never says so; it only writes "busBW". The author has since confirmed to the user that it did not use the nccl-tests busBW. The report now uses link-model busBW (metric N5 in the [common setup](../00_common_setup.md#link-model-busbw-n5)): algBW × 1 for AllGather and ReduceScatter and algBW × (1 + 1/n) for AllReduce. Under it, the AllReduce test above stays the primary test. The rows above are kept as first written.

**Note (2026-11-01): this traffic model is wrong on our machine.** Whole-run link bytes in E1.1 show `st_multicast` at x GB/s per GPU gives about 4x in and x out, not 3x in. `ld_reduce` gives about x in and 4x out. AllReduce gives about 5x each way, not 4x. So the issuing GPU seems to count as a receiver of its own data. busBW counts only 3x (or 6x for AllReduce), so busBW and link load are not in a fixed ratio across the arms. The table above is kept as first written. Treat its link columns as wrong by this factor.

## Questions for the author

Send these before implementing the harness. Each one has a default so the plan can proceed without an answer.

| Question | Why it matters | Our default if no answer |
|---|---|---|
| Which GPU, NVLink generation and driver? | Decides how absolute values compare | Compare ratios only. **Answered 2026-11-04:** GB300, NVLink 5; driver not stated |
| How was busBW computed, for every row? (First written for the two "+" rows only) | See the table above | Report both readings. **Partly answered 2026-11-07:** not the nccl-tests busBW; the exact formula is not known |
| What traffic pattern did the "+" rows use? Ring for plain ld/st? | Values 662.3 and 668.1 match the ring rows in section 3.2, so we assume a ring | Ring `G0 → G1 → G2 → G3 → G0` |
| Can you share the benchmark code? | Would remove most other unknowns | Our role loops from the NCCL kernels |
| Threads per CTA, data type, pack size, buffer size? | Not stated; they change the request supply | 256 and 512 threads/CTA calibrated; `float`, 16-byte packs; 8 GiB |
| Fixed bytes or fixed time? Did both roles fully overlap? | With fixed bytes, the faster role may have run alone at the end | Fixed time with measured overlap |
| Did the ranks use the same or different address ranges? | NCCL commit `995eb333` spreads multimem addresses by rank "to avoid contention" | Separate 2 GiB parts per rank, no extra offset; E1.6 tests offsets |

## Possible explanations this plan must keep apart

| Explanation | What it says | Evidence that would support it | Sub-experiment |
|---|---|---|---|
| Shared in-flight limit on the GPU (report's idea) | The two operations share a limit on requests in flight | Drop with links not full; drop starts at low supply on each side | E1.2, E1.5, E1.6 |
| Link capacity | Bytes per direction already reach the link limit | NVLink bytes near the limit found with plain loads | E1.5 |
| Too little request supply | Each thread in AllReduce waits for `ld_reduce` before it can store, so 32 CTAs issue less | More CTAs raise AllReduce | E1.1 |
| Address or memory contention | Accesses from several ranks collide on addresses or HBM | Address offsets change the drop; HBM bytes high | E1.6, E1.5 |
| NVSwitch interference | The switch, not the GPU, is the limit | Drop also appears when different GPUs issue each operation | E1.4 |
| Power or clock | Mixed runs run at lower clocks | Clock-event reasons differ between arms | All (B2) |

Even a clean result only narrows this list. Counters do not name a physical queue.

## Reproduction test

These limits are set before any run. Ratios use per-round pairs of run medians, at least five rounds. The reference for a ratio is the mean of the two single-operation busBW values in the same round.

**Note (2026-11-07).** The limits below are unchanged. After the runs, the busBW they apply to changed from nccl-tests busBW (N2) to link-model busBW (N5), because the author confirmed the report does not use nccl-tests busBW. This is a change made after seeing the data, and the [report](README.md#how-busbw-is-computed-in-this-report) gives both values.

| Report claim | Measure | Reproduced | Partly | Not reproduced |
|---|---|---|---|---|
| Multimem mix falls (report ratio about 0.79) | AllReduce busBW ÷ reference | Median ≤ 0.90 and at most one round above 0.90 | Median 0.90–0.97 | Median > 0.97 |
| Same and separate CTAs behave alike | Separate-CTA combined busBW ÷ AllReduce busBW, under whichever reading the author confirms, otherwise under the sum reading | 0.95–1.05 | 0.90–1.10 | Outside 0.90–1.10 |
| Plain ld/st does not fall (report ratio about 1.0) | Mixed busBW ÷ mean of plain-load-only and plain-store-only | Median ≥ 0.95 | 0.90–0.95 | Median < 0.90 |

A result is only valid if E1.1 shows the reference is at its plateau and N4 shows at least 95% overlap. All comparisons need a correct run (A11).

## E1.0 Can this machine run multimem at all?

### Motivation

Everything else depends on this. It also checks the tools before any harness work.

### Configuration

| Knob | Value |
|---|---|
| Ranks | 4 |
| Measured | `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`; `props.multimemSupport` from `ncclCommQueryProperties`; NCCL's `NVLS multicast support is ... available` log line |
| Forcing | Do **not** set `NCCL_NVLS_ENABLE=1`. It skips the check and tells us nothing |

### Procedure

1. Get the GPU allocation and check that the GPUs are idle.
2. Run the existing example [`docs/examples/07_kernel_fusion/02_rmsnorm_multimem`](../../../../../docs/examples/07_kernel_fusion/02_rmsnorm_multimem/README.md) on 4 GPUs with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT`. It checks `multimemSupport` and exercises `multimemLoadSum` and `multimemStore`.
3. Run `all_reduce_perf` from nccl-tests against the in-tree library with `-R 2` and `NCCL_DEBUG_SUBSYS=TUNING`, at one size, and confirm from the log that the kernel `AllReduce_RSxLDMC_AGxSTMC` was chosen.
4. Decide using the table below.

| Result | Next step |
|---|---|
| Supported and the example passes | Continue with E1.1 |
| Reported as not supported | Multimem rows cannot run in this VM. Stop, and report to the user. Options: ask the admin about the fabric setup, or use another machine. E1.3 (plain ld/st) can still run but cannot reproduce the main claim |
| Reported as supported but the example fails or hangs | Keep the logs and report to the user. Do not retry with forced settings |

### Privileged access

4 free GPUs. No extra permission. The admin may be needed for the fabric setup if not supported.

### Metrics

| Metric | Expected signature | Separates from |
|---|---|---|
| A11 | Correct results if multimem works | Not a performance measurement |
| B2 | Recorded for the session | |

## E1.1 Where does each operation stop gaining from more CTAs?

### Motivation

The report's 32 CTAs may not be enough on our GPU, or may be more than needed. A mix can only show a drop in a shared limit if each operation alone can reach its plateau. AllReduce needs its own sweep, because each thread waits for its `ld_reduce` before it can store. A lack of request supply there could look like interference.

### Configuration

| Knob | Values | Fixed within the comparison |
|---|---|---|
| **Active CTAs of the one role (independent variable)** | **4, 8, 16, 32, 48, 64** | Launch of 64 CTAs × threads/CTA; other CTAs are idle |
| Threads/CTA | 256, 512 (add 1024 if the curve is still rising and NCCL limits do not apply) | Same for role and idle CTAs |
| Role | `multicast`, `reduce`, `allreduce`, `plain_load` on ring, `plain_store` on ring | One role per sweep |
| Constants | 8 GiB regions, 16-byte packs, 300 ms window, 4 GPUs | |

Example label: "`allreduce`: 32 CTAs × 512 threads/CTA, 4 GPUs".

### Procedure

1. Run each role and setting with 3 runs, rotating the order of settings. About 150 runs, at roughly 20 s each, is about one hour.
2. Stopping rule, set in advance: the plateau is reached when two successive CTA steps improve the median by less than 1%, and the spread of the runs is under 2%. If 64 CTAs is still rising, extend it or state why it stays incomplete.
3. Select the reference for each role. The report's 32 CTAs stays as the main setting if it is on the plateau. If it is not, run E1.2 and E1.3 at **both** 32 CTAs and the plateau setting.

### Privileged access

4 free GPUs. Otherwise none.

### Metrics

| Metric | Expected signature | Separates from |
|---|---|---|
| N1, N2 | Rises then flattens. A rising `allreduce` beyond 32 CTAs would support "too little request supply" | Separates request supply from a shared limit |
| N3 | Even progress across CTAs | Shows starved CTAs |
| B3 | Bytes follow the role | Whole-run check only |

## E1.2 Multimem table: each operation alone, then both

### Motivation

This is the main result. It models an AllReduce, where each GPU issues both operations for its whole run. A wall-clock busBW alone hides which role lost speed, and how the links were used.

### Configuration

| Arm | Role table per GPU (active CTAs × threads/CTA) | Comparison role |
|---|---|---|
| Single multicast | `multicast` 32 × T; idle 32 | Reference |
| Single reduce | `reduce` 32 × T; idle 32 | Reference |
| **Mix on separate CTAs** | `multicast` 32 × T; `reduce` 32 × T | Perturbed |
| **Mix on same CTAs** | `allreduce` 32 × T; idle 32 | Perturbed |

| Knob | Value |
|---|---|
| **Independent variable** | **Which operations run, and whether they share CTAs (the four arms above)** |
| Fixed | T from E1.1; launch of 64 CTAs × T; 8 GiB regions; 16-byte `float` packs; 300 ms window; same 4 GPUs; roles spread over GPCs; same shared-memory reservation |
| Measured | N1 per role per GPU, N2, N3, N4 |

This is the report's single level of pressure. E1.6 adds the required ladder.

### Procedure

1. Check that the allocated GPUs are idle.
2. Run the four arms once per round, for **6 rounds**, rotating the arm order. Repeat at the plateau setting if E1.1 differs from 32 CTAs.
3. Compute N2 for each arm. For the mix on separate CTAs, report per-role rate and the sum.
4. Apply the reproduction test.

### Privileged access

4 free GPUs. Otherwise none.

### Metrics

| Metric | Expected signature (hypothesis from the report) | Separates from |
|---|---|---|
| N2 | Both mixes about 0.8 × single-operation busBW | Primary outcome. Cannot separate causes alone |
| N1 | Report does not say whether both roles lose equally. Equal loss suggests fair sharing; unequal loss suggests one role takes priority | Separates role priority from equal sharing |
| N3 | Even across CTAs unless a role is starved | Separates starving from a uniform slowdown |
| N4 | At least 95% | Rejects runs where the roles did not overlap |
| B2 | Same clocks in all arms | Separates power and clock effects |

## E1.3 Plain ld/st table: the same mix without multimem

### Motivation

This is the control behind the report's third claim. If plain ld/st also falls in our setup, the drop is not specific to multimem. Plain loads also produce traffic in both directions (requests one way, data the other), so this tests a mix with similar direction use.

### Configuration

| Arm | Role table per GPU (active CTAs × threads/CTA) | Comparison role |
|---|---|---|
| Plain load only | `plain_load` 32 × T; idle 32 | Reference |
| Plain store only | `plain_store` 32 × T; idle 32 | Reference |
| **Plain mix** | `plain_load` 32 × T; `plain_store` 32 × T | Perturbed |

| Knob | Value |
|---|---|
| **Independent variable** | **Which plain operations run (the three arms above)** |
| Traffic pattern | Ring `G0 → G1 → G2 → G3 → G0`. In `plain_load`, each GPU loads from the next. In `plain_store`, each GPU stores to the next |
| Fixed | Same as E1.2. Use T from E1.1, and the same launch |
| Measured | N1, N2, N3, N4 |

### Procedure

1. Run the three arms for 6 rounds, rotated, in the same session as E1.2 where possible.
2. Report N2 under both readings (per role and summed), and apply the reproduction test. Note that a summed plain mix and a summed multimem mix do not put the same load on the links; see the table above and E1.5.

### Privileged access

4 free GPUs. Otherwise none.

### Metrics

| Metric | Expected signature (hypothesis from the report) | Separates from |
|---|---|---|
| N2 | Plain mix about equal to plain alone | Separates a multimem-specific effect from any load + store mix |
| N1 | Each role's share of the link | Shows how the capacity is divided |
| N4 | At least 95% | Rejects runs without overlap |

## E1.4 Same mix, different GPUs issue each operation

### Motivation

The report's section title says "at each GPU". That claim rests on the previous section 3.3: with two GPUs running AllGather and two others running ReduceScatter, each group still reached 690 GB/s. If this control fell too, the location would not be the GPU. It must be repeated here, not borrowed from the report.

### Configuration

| Arm | G0, G1 | G2, G3 |
|---|---|---|
| Multicast on two GPUs | `multicast` 32 × T | idle |
| Reduce on two GPUs | idle | `reduce` 32 × T |
| **Split mix** | `multicast` 32 × T | `reduce` 32 × T |

| Knob | Value |
|---|---|
| **Independent variable** | **Whether the two operations run on different GPUs, at the same time** |
| Multicast group | **Group of all 4 GPUs**, as the default. A 2-GPU group is a variant (`E1.4b`), because the report does not state which it used, and NCCL's symmetric kernels only enable multimem above 2 ranks |
| Fixed | Same as E1.2 |

### Procedure

1. Run the three arms for 6 rounds, rotated, in the same session as E1.2.
2. Compare each role in the split mix with its alone arm. Apply the same limits as the plain ld/st row: median ≥ 0.95 reproduces the report's "no drop".

### Privileged access

4 free GPUs. Otherwise none.

### Metrics

| Metric | Expected signature (hypothesis from the report) | Separates from |
|---|---|---|
| N1, N2 | No drop: each role about equal to its alone arm (the report gives 690 GB/s per group) | Separates a limit inside the GPU from one in the switch, under this traffic |
| C9 | Link use from E1.5 | Fan-out differs here, so the links are not loaded as in E1.2 |

The switch sees different traffic here than in E1.2. No drop does not fully clear the switch. Keep that limit in the report.

## E1.5 How full are the links? (counter pass)

### Motivation

The table in "Three problems to settle first" shows that the "560" number could mean full links or half-full links. Only a byte count per direction can decide. It also tests the traffic model itself.

### Configuration

| Knob | Value |
|---|---|
| Arms | All arms of E1.2, E1.3 and E1.4 at the selected setting |
| **Independent variable** | **Same arms as the timing runs**; the new part is the observation |
| Instrument | `ncu --replay-mode application --clock-control none --cache-control none` with `nvltx__bytes_packet_*`, `nvlrx__bytes_packet_*`, then DRAM families (C1) and L2 families (C3), per GPU, one GPU per pass |
| Fixed | Harness and config unchanged from the timing runs |

### Procedure

1. First, read B3 from the timing runs, as a coarse check.
2. Validate on known traffic: plain store from one GPU, where link bytes must match stored bytes. Record what protocol bytes do.
3. Collect each arm, 3 passes per GPU. Check N1, N4 and A11 in every pass, and reject a pass that lost the overlap (C8).
4. Compare measured link bytes with the traffic-model table. Report how far the model is off. If the model is wrong, say so and correct the reasoning above with a dated note.

### Privileged access

Performance counters through `ncu`. They worked for plan 08. Fallback: B3 whole-run totals, which also include setup and checking traffic; say so in the report. Without either, the "560" question stays open, and the report must say so.

### Metrics

| Metric | Expected signature | Separates from |
|---|---|---|
| C9 | Each direction near the plain-load limit: link capacity. Well under it: something else | Separates link capacity from a shared limit in the GPU |
| C1 | HBM bytes near the role rates; HBM far below its 3352 GB/s peak | Separates HBM limits from others |
| C3 | L2 traffic per arm | Recorded for interpretation, cannot separate alone |
| C8 | Passes valid | Rejects passes that removed the interaction |

## E1.6 Extension: how the drop grows, and whether addresses matter

Run this only if E1.2 is reproduced or partly reproduced. **This goes beyond the report.** It adds the pressure ladder the conventions require, and tests explanations the report did not test.

### Motivation

The report has one pressure level. A shared in-flight limit should behave differently from a full link when the second operation is weak: at low pressure, a link that is not yet full should lose nothing. NCCL's own commit spreads multimem addresses "to avoid contention", but `allreduceMultimem` has no such spread, so addresses are a live alternative.

### Configuration

| Sub-experiment | **Independent variable** | Values | Fixed |
|---|---|---|---|
| E1.6a | **Active `reduce` CTAs next to 32 `multicast` CTAs** | **0 (reference), 4, 8, 16, 32** | Launch of 64 × T; achieved rate of each level is measured, not assumed |
| E1.6b | **Active `multicast` CTAs next to 32 `reduce` CTAs** | **0 (reference), 4, 8, 16, 32** | Same |
| E1.6c | **Per-rank address offset** | **None; NCCL's rank pattern `rank ^ ((rank >> 1) & 1)` × 32 MiB (`ncclSymkMcPerRankOffsetBytes`)** | Both mix arms of E1.2 |

### Procedure

1. Calibrate levels: run each with E1.1's method and record the achieved rate of the weaker role.
2. Run interleaved reference and perturbed runs, with at least 5 pairs per level, rotating the levels.
3. Plot the rate of the 32-CTA role against the achieved rate of the other role, with zero-based axes.

### Privileged access

4 free GPUs. Otherwise none.

### Metrics

| Metric | Expected signature | Separates from |
|---|---|---|
| N1 | Near-linear fall from the first level suggests a shared limit; no fall until the link total is reached suggests link capacity | Separates a shared limit from link capacity |
| N2 with offset | A large change with E1.6c suggests address contention | Separates address effects from limits that do not depend on addresses |
| C9 | Link use at each level, one pass per level at most | Supports the reading above |

## E1.7 Cross-check with real NCCL collectives

### Motivation

Our role loops are not the author's code. The report's section 3 also gives a number from NCCL itself (AllReduce about 550 GB/s with registered buffers). The in-tree kernel `AllReduce_RSxLDMC_AGxSTMC` runs the same `ld_reduce` then `st_multicast` loop, so this ties our harness to real collective behavior. Fixed-byte runs also show if the fixed-time design changed the result.

### Configuration

| Knob | Value |
|---|---|
| Tools | `all_gather_perf`, `reduce_scatter_perf`, `all_reduce_perf` from `/root/nccl-tests/build`, against the in-tree library |
| Registration | `-R 2` (`SYMMETRIC_REGISTER`, which calls `ncclCommWindowRegister` with `NCCL_WIN_COLL_SYMMETRIC`) |
| **Independent variable** | **Collective: AllGather (`AllGather_STMC`), ReduceScatter (`ReduceScatter_LDMC`), AllReduce (`AllReduce_RSxLDMC_AGxSTMC`)**, selected with `NCCL_SYM_KERNEL=<collective>:<kernel>`, for example `NCCL_SYM_KERNEL=allreduce:AllReduce_RSxLDMC_AGxSTMC`. E1.0 verified this for AllReduce only |
| Fixed | 4 GPUs, 8 GB total, `float` sum. Record the CTA count from the log |

**Note (2026-11-01).** This row first said `NCCL_ALGO` with `SYMK_` names. E1.0 showed that `NCCL_ALGO` rejects them. E1.0 also showed NCCL choosing 4 CTAs for AllReduce at 1 GiB, so CTA count must be recorded for each collective and not assumed equal to the report's 32 blocks. Compare with E1.2 at the same CTA count where possible.

The report's section 2 numbers come from NCCL too: AllGather about 700 GB/s, ReduceScatter about 670 GB/s, with registered buffers. Unregistered runs (its 620 GB/s number) are outside this plan.

### Procedure

1. Run each collective for 5 rounds, rotated. Confirm the chosen kernel in the NCCL log for every run, and treat any other kernel as invalid.
2. Apply the same ratio: AllReduce busBW ÷ mean of AllGather and ReduceScatter busBW. Compare it with the harness result from E1.2.

### Privileged access

4 free GPUs. Otherwise none.

### Metrics

| Metric | Expected signature | Separates from |
|---|---|---|
| N2 | Ratio about 0.79 and close to E1.2 | A large gap with E1.2 points to the harness or to NCCL overhead |
| B3 | Whole-run link totals | Coarse only |

## Order of work

**Note (2026-11-01).** Steps 1, 7 and 8 did not run: the questions were not sent, the `ncu` pass was replaced by whole-run link data, and E1.6 was dropped. E1.7 ran before the link analysis. E1.2's busBW result was not reproduced, and rates fell only partly as expected; see the [report](README.md).

| Step | Work | GPU needed | Rough size | Stop or continue |
|---|---|---|---|---|
| 1 | Send questions to the author | No | | Do not wait for answers to run E1.0 |
| 2 | E1.0 multimem support | Yes | Under one hour | Done 2026-11-01: supported on GPUs 4–7, continue |
| 3 | Build `h100x4/nvls/common/`, reusing patterns from `store_congest.cu` | Short tests | Largest coding part | |
| 4 | E1.1 calibration | Yes | About 1 hour | Must reach a plateau or state why not |
| 5 | E1.2, E1.3, E1.4 in one session | Yes | About 30 minutes | Apply the reproduction test |
| 6 | E1.7 NCCL cross-check | Yes | About 30 minutes | |
| 7 | E1.5 counters | Yes | 1–2 hours | |
| 8 | E1.6 extension | Yes | 1–2 hours | Only if step 5 reproduces at least partly |
| 9 | Report in `README.md` | No | | Report every claim as reproduced, partly, or not, with the cause kept separate |

## What this plan will not establish

- It will not prove a physical in-flight limit. Timing and counters can make it more or less likely, but cannot name the internal resource.
- H100 results do not transfer to the report's platform, and the report's results do not transfer here. The two link speeds differ. A limit that binds on one may not on the other.
- Our role loops are close to NCCL's kernels, but are not the author's benchmark unless the author shares it.
