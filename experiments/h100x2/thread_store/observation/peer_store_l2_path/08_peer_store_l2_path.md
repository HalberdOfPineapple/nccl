# 08 · Path of incoming peer stores through the destination's L2

This plan covers the problem-matrix cell **thread-issued peer stores × memory access and locality**. It tests an assumption that [plan 03](../memory_access/03_memory_access.md#repeated-writes-and-destination-caching) and [plan 07](../data_store_congest/07_data_store_congest.md) only state: that peer stores land in the destination GPU's L2.

**Scope note (2026-10-31).** By request, only E8.0 sanity checks and E8.1 are reported. E8.2 to E8.4 runs stopped before analysis, and E8.5 did not run. The collection method that worked is `ncu --replay-mode application`, which this plan did not list. Differences from this plan are in the [report](README.md).

**Common question.** When a thread on the source GPU stores to the destination GPU's memory through an NCCL LSA pointer over NVLink, does the data pass through the destination GPU's L2 cache on its way to HBM?

"Passes through" can mean several things, so the plan separates three steps. Each sub-experiment answers one.

| Step | Question | Answered by |
|---|---|---|
| Enter | Does each incoming store cause an access to the destination's L2? | E8.1, E8.4 |
| Stay | Does the L2 keep the data for a while, so HBM is written later or not at all? | E8.2, E8.3 |
| Take space | Does the data take a place in the L2 and push out other data? | E8.5 |

This plan injects no problem, so it has no healthy/perturbed pairs. The reference is **the same store pattern issued by a thread on the destination GPU to the same memory (local stores)**, which we assume uses the L2 normally. E8.0 checks that assumption before anything depends on it. Only E8.5 has a pressure-style comparison.

Words used in this plan:

| Word | Meaning |
|---|---|
| Source `I`, destination `D` | `I` is the GPU whose threads issue the stores. `D` is the GPU that owns the memory. No third GPU is used |
| Peer store | A store by a thread on `I` through the pointer from `ncclGetLsaPointer`. The memory is `ncclMemAlloc` device memory on `D`, mapped on `I` by NCCL |
| Local store | A store by a thread on `D` to the same memory, with the same pattern |
| Footprint | The number of different bytes a run stores to. Stores repeat over this range until the run's total is reached |
| Sector | The unit in which L2 counters count. Expected to be 32 bytes; E8.0 confirms this with local stores |
| PM sampling | Hardware counters read at a fixed time interval on one GPU. It does not re-run any kernel, unlike the profiler's replay modes |

## The possible paths

```text
I: SM -> crossbar -> [I's L2?] -> NVLink send -> NVSwitch -> NVLink receive -> D: hub? -> [D's L2?] -> HBM
```

The part in question is the end of this line. The plan must separate these models. They are hypotheses, and the result may fit none of them.

| Model | What happens at `D` |
|---|---|
| **Write-back** | The store enters the L2 and stays there. HBM is written only when the line is pushed out |
| **Write-through** | The store accesses the L2, which updates or drops any matching line, and also goes to HBM at once |
| **Bypass** | The store goes to HBM without any L2 access |

What each observation should show under each model:

| Observation | Write-back | Write-through | Bypass |
|---|---|---|---|
| L2 write sectors on `D` that follow the pushed bytes (E8.1) | Yes | Yes | No |
| HBM write bytes on `D` while repeatedly storing to a small footprint (E8.2) | Far below the pushed bytes | About equal to the pushed bytes | About equal to the pushed bytes |
| HBM write bytes on `D` when clean data later pushes the footprint out of L2 (E8.2) | About the footprint | About zero | About zero |
| First read on `D` right after the push (E8.3) | Fast, like after local stores | Fast if the line is updated, slow if it is dropped | Slow, like after eviction |
| Line already in `D`'s L2 before the push, then read after it (E8.4) | New value, write lookup hits | New value, write lookup hits | **Old value is possible**, unless something else keeps the L2 correct |
| `D`'s other data pushed out of L2 by a large push footprint (E8.5) | Yes | Only if a store miss takes a place in L2 | No |

## Why throughput cannot answer this

Copy throughput stays near 370 GB/s in plans 04 and 07 under most conditions, so something earlier in the path limits it. That rate is about 11% of HBM peak, so neither L2 nor HBM needs to slow the copy. The same throughput therefore fits all three models. A completion acknowledgement does not help either: it shows that data can be read correctly, not where that data is. Only counters on `D` and read timing on `D` can separate the models.

## Common experiment configuration

Values below apply to every sub-experiment unless its table says otherwise. Everything else is inherited from [00_common_setup.md](../00_common_setup.md).

| Knob | Setting | Note |
|---|---|---|
| GPUs | One pair from the user's allocation, chosen before each session | Plan 07's SM-to-GPC maps exist for GPUs 0, 1, 4, 5. Do not mix pairs in one comparison |
| Clocks and power | Unchanged; observed only | Profilers use no clock control |
| Peer stores | Communication: 16 CTAs × 512 threads/CTA on `I`, 4-byte stores, CTA-local stride, values computed in registers, no source loads | Plan 07's store-only copy, because `I`'s memory then carries no copy data and `I`'s counters stay clean. Parts start on 128-byte boundaries |
| Local stores | Local writer: 16 CTAs × 512 threads/CTA on `D`, same store width, order, values and memory | Rate is measured, not forced equal; report it |
| Stored values | `hash(element index, pass number)`, new on every pass | Values change on every pass, so nothing can skip a store because the value is unchanged, and compression cannot shrink HBM traffic |
| Receiver | 1 CTA on `D`, one polling lane | Its control traffic is measured in the no-traffic control |
| Memory | `ncclMemAlloc` + `ncclCommWindowRegister`, 2048 MiB payload area | Same mapping as earlier plans |
| Eviction buffer | Separate 256 MiB `cudaMalloc` buffer on `D`, about 5 × the 50 MiB L2, written before the measured phases and only read during them | Reading it pushes L2 contents out without adding any store to HBM. E8.0 checks that it is enough |

### One launch, separated phases

A run is one launch per GPU. It must not use a new kernel launch between phases. The runtime may flush caches at launch boundaries (not documented; E8.0 tests it), which would hide exactly what this plan measures.

| Phase | What runs | Purpose |
|---|---|---|
| Clean | `D` reads the eviction buffer | Removes earlier dirty lines, so the L2 starts in a known state |
| Quiet 1 | Nothing, 5 ms | HBM writes must fall to the background level |
| Prepare | Only in E8.4 and E8.5: `D` reads or stores its data | Sets the L2 state under test |
| Push | Stores from `I` (or local stores from `D`), ending with the usual fence, "done" flag and acknowledgement | Measured traffic |
| Quiet 2 | Nothing, 5 ms | Shows whether the L2 writes data back without being forced |
| Evict | `D` reads the eviction buffer | HBM writes here measure the dirty data the L2 was holding |
| Check | `D` reads the whole footprint and compares every value with the last pass | Correctness. Runs after all measured phases |

The quiet phases are long compared with the sampling interval, so phase boundaries can also be found in the counter data itself. Phase start and end times are recorded with `%globaltimer` on the GPU where they are used.

### Observations

| Observation | Id | Instrument and endpoint | Window | Limit |
|---|---|---|---|---|
| Copy duration and throughput | A1 | Harness timer, `I` | Push | With no repeated footprint this is the usual copy. With a small footprint, it is bytes stored over time |
| Phase timers, roles and SM IDs, correctness | A2–A6, A10–A12 | Harness, both | Whole run | As in the common setup |
| NVLink bytes sent and received | C9 | `nvltx__*` on `I`, `nvlrx__*` on `D` | Each phase | Proves how much data crossed the link. Includes protocol traffic and any background |
| HBM read and write bytes | C1, C2 | `dram__bytes_read`, `dram__bytes_write` on both | Each phase | GPU-wide |
| L2 sectors by operation, source and hit or miss | C3 | L2 counter list below, on both | Each phase | GPU-wide. The meaning of "hub" is not documented; E8.1 tests it by dose and response |
| Read-back hop time | A13 | Harness timer, `D` | Read after push | An average over many loads, not the latency of one load |
| Values that are old or wrong | A14 | Harness, `D` | Read after push | Zero wrong values does not show which path was used |
| Device state, whole-run NVLink counters | B2, B3 | Run wrapper, both | Whole run | Context only |

Counters to try. All appear in the installed tool's list for `gh100`, but **none has been collected yet**; listing does not show that a counter works.

| Purpose | Counter names |
|---|---|
| L2 sectors overall | `lts__t_sectors`, `lts__t_sector_hit_rate` |
| Share of L2 sectors from the hub, writes to device memory | `lts__average_t_sector_srcnode_hub_aperture_device_op_write` with `_lookup_hit` and `_lookup_miss` |
| Share of L2 sectors from the hub, reads | `lts__average_t_sector_srcnode_hub_aperture_device_op_read` with `_lookup_hit` and `_lookup_miss` |
| Share from other sources, as a check | `lts__average_t_sector_srcnode_gpc`, `lts__average_t_sector_srcnode_fbp` |
| Source L2 used for peer-bound stores (read on `I`) | `lts__average_t_sector_aperture_peer_op_write` with `_lookup_hit` and `_lookup_miss` |
| NVLink payload | `nvlrx__bytes_packet_request_data_user`, `nvltx__bytes_packet_request_data_user`, and their `_protocol` forms |
| HBM | `dram__bytes_write`, `dram__bytes_read` |

The `lts__average_*` names are shares of all sectors, so a count is share × `lts__t_sectors`. They appear in the PM-sampling list, while the plain counters (for example `lts__t_sectors_op_write`) appear in the profiling list. Which counters can be read together decides the groups in E8.0.

## E8.0 · Make the counters work, and check they tell the truth

### Motivation

No hardware counter has ever been collected on this harness. Kernel replay hung, and range attempts gave no valid range (see the [harness README](../../common/README.md)). The likely cause of the hang, not yet confirmed, is that a replayed kernel on one GPU waits for a kernel on the other GPU that is not replaying. Counters are also the main evidence here, so they need checking against traffic of known size. A counter that reads zero could mean no traffic, or that it does not see traffic started by another GPU.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Collection method** | **Independent variable**, tried in this order; stop at the first that passes every check | **(1) `nsys` GPU metrics with `gh100-ct`, sampling `D` (no code change; contents of this set must be checked); (2) CUPTI PM sampling (`cupti_pmsampling.h`) on `D` and `I`, from a helper or inside the harness; (3) CUPTI range profiler with the harness repeating the transfer; (4) `ncu --replay-mode app-range`** |
| **Known traffic** | **Independent variable** | **None (receiver only); local stores 1024 MiB; local reads 1024 MiB; peer stores 1024 MiB** |
| Sampling interval | Calibrated | Start at 100 µs. Must show the phases and leave no gaps between samples |
| Repeats | Constant | 3 runs per case in this validation |

Kernel replay is not retried: it already hung on this kind of two-GPU launch.

### Procedure

1. Build the new harness (see [Harness work](#harness-work)). Check that it reproduces plan 07's copy throughput at 16 CTAs × 512 threads/CTA, with counters off.
2. List the counters each method can collect together, and write the resulting groups and pass counts into the README.
3. Run each known-traffic case with counters on. Compare copy throughput with the counters-off run; a change above about 1% is reported as probe cost.
4. Checks that must pass before any later step:
   1. Receiver only: every counter stays near background.
   2. Local stores: L2 write sectors × sector size is at least the stored bytes (it can be higher, because one sector can be counted more than once when stores are not merged); HBM write bytes over all phases are about 1024 MiB.
   3. Local reads: HBM read bytes are about 1024 MiB when the footprint is larger than L2.
   4. Peer stores: `nvlrx` user bytes on `D` and `nvltx` user bytes on `I` both match the stored bytes within an explained factor.
   5. Counters on `D` do change during peer stores, even with only one small receiver CTA running. If they stay flat, check whether counters only count while a local kernel is scheduled, and whether PM sampling is limited to one context.
5. Launch-boundary test: use the local-store case with a small footprint, and compare HBM writes during Evict when Evict runs in the same launch and when it runs in the next launch. A big difference confirms that phases must stay in one launch.
6. Eviction check: repeat Evict with 1, 2 and 4 passes over the buffer. Choose the smallest count after which more passes add no HBM writes.

If every counter method fails, E8.1 and E8.2 cannot run. E8.3 and E8.4 still give timing and correctness evidence, and the report must say the counter steps are missing.

### Privileged access

Performance counters must be open. `RmProfilingAdminOnly` read 0 on 2026-10-30, and the current user is root in the container. Profilers must not use clock control. Free GPUs are required, because the counters are GPU-wide and another tenant's traffic on `D` would be counted. Fallback: E8.3 and E8.4 only.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| C9 | Bytes on `I` send and `D` receive match the stored bytes, plus a fixed factor (hypothesis) | Setup evidence |
| C2 | Local-store bytes reach HBM by the end of Evict (hypothesis) | Setup evidence: shows the local reference and eviction work |
| C3 | Local stores count as GPC-sourced writes; peer stores count under another source node, if at all (hypothesis) | Setup evidence |
| C8 | Records the method, groups, passes and probe cost | Valid against invalid counter data |

## E8.1 · Do incoming stores access the destination's L2?

### Motivation

This is the direct test of the "enter" step. A large footprint is used so that the L2 cannot hide traffic. HBM writes then happen under all three models, and they cannot separate the models here; only L2 counts can. Using several sizes shows whether L2 counts follow the data ("dose and response"), which is stronger than one non-zero number, and it shows which source node label belongs to NVLink traffic.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Bytes stored once each** | **Independent variable** | **256, 512, 1024, 2048 MiB (footprint = bytes, one pass)** |
| **Who stores** | **Independent variable** | **Peer stores (16 CTAs × 512 threads/CTA on `I`); local stores (16 CTAs × 512 threads/CTA on `D`); no stores (receiver only, same duration)** |
| Store width | Constant | 4 bytes |
| Repeats | Constant | 5 runs per case, rotated order, cases interleaved |

### Procedure

1. Run the opening no-stores control.
2. Run 5 rounds, rotating the order of size and stores.
3. Run the closing no-stores control.
4. Fit L2 write sectors on `D` against `nvlrx` user bytes, for each source node. Report slope, intercept and spread.
5. Read `I`'s counters: does `aperture_peer` write traffic appear on `I`'s L2? This shows whether stores also pass the source's L2.

### Privileged access

Counters from E8.0, on both GPUs. Fallback: none for this sub-experiment.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1 | About the plan-07 reference (hypothesis); checks the counters did not change the workload | Reference only |
| C9 | `nvlrx` bytes on `D` grow in proportion to stored bytes (hypothesis) | Confirms traffic crossed NVLink, not another path |
| C3 on `D` | Write-back and write-through: hub-sourced L2 write sectors grow with `nvlrx` bytes. Bypass: stay at the no-stores level | Enter or bypass. Cannot separate write-back from write-through |
| C3 on `I` | Peer-aperture write sectors grow with stored bytes if stores pass `I`'s L2 (hypothesis) | Source-side path |
| C2 | HBM writes on `D` are about the stored bytes under all models | Cannot separate; checks the accounting |

## E8.2 · Does the L2 keep incoming data? (main test)

### Motivation

This is the main test of the "stay" step. Stores repeat over one footprint until a fixed total is reached. If the L2 keeps dirty lines and the footprint fits, later passes overwrite the L2 copy and HBM sees almost nothing. If it does not keep them, HBM sees every pass. Plotting HBM write bytes against footprint gives a clear shape, and local stores on the same axes show what L2 retention looks like on this GPU. Throughput would look the same in all cases, so this effect would otherwise be invisible.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Footprint** | **Independent variable** | **1, 4, 16, 32, 64, 128, 512, 2048 MiB (L2 is 50 MiB)** |
| Passes over the footprint | Coupled to footprint | 8192, 2048, 512, 256, 128, 64, 16, 4, so the total is always 8192 MiB |
| **Who stores** | **Independent variable** | **Peer stores (16 CTAs × 512 threads/CTA on `I`); local stores (16 CTAs × 512 threads/CTA on `D`)** |
| Fence between passes | Constant in the base sweep | None; only the final fence |
| Repeats | Constant | 5 runs per case; peer and local interleaved at each footprint, order rotated |

Variants, each at footprints 4, 32 and 512 MiB, peer and local:

| Variant | **Independent variable** | Question |
|---|---|---|
| E8.2b | **Store shape: 4-byte stores that cover every byte (base); 16-byte stores that cover every byte; 4-byte stores to one word in each 128-byte line** | Are wider stores or partly written lines treated differently? For the partly written case the total is set in stored bytes, and the expected ratios are recalculated |
| E8.2c | **Fence: none (base); a system fence by every writing thread after each pass** | Does a fence on `I` force `D`'s L2 to write data to HBM? This matters for how much the completion protocol costs |

### Procedure

1. Run the sweep with the phase sequence above.
2. Compute for each run:
   1. **Link ratio** = `nvlrx` user bytes on `D` during Push / stored bytes. **This must be checked first.** If it is well below 1 at small footprints, stores were merged before they reached `D`, for example on `I`, and low HBM writes then say nothing about `D`'s L2.
   2. **Write-through ratio** = HBM write bytes on `D` during Push / stored bytes.
   3. **Held ratio** = HBM write bytes on `D` during Evict / footprint.
   4. **Self-writeback** = HBM write bytes on `D` during Quiet 2 / footprint.
3. Plot the four ratios against footprint, peer and local on the same axes.
4. Run E8.2b and E8.2c.

### Privileged access

Counters from E8.0. Fallback if only NVLink and DRAM counters work (for example from `nsys`): this test still runs, because it needs no L2 counter.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1 | No large change with footprint if neither L2 nor HBM limits the copy (hypothesis) | A change would show the path matters to performance |
| C9 | Link ratio about 1 at every footprint (hypothesis) | Merging at `D` from merging earlier in the path |
| C2 on `D` | Write-back: write-through ratio is low below 50 MiB and rises to about 1 above it, held ratio about 1 below it. Write-through or bypass: write-through ratio about 1 everywhere, held ratio about 0 | Write-back from the other two. Cannot separate write-through from bypass |
| C2, peer against local | Same curve means the same L2 treatment. A different knee means a different policy | Local stores as a reference |
| C3 on `D` | Write lookup hits rise at small footprints if the L2 keeps data (hypothesis) | Supports C2 |

## E8.3 · Does a read on the destination find the data in L2?

### Motivation

This test uses timing and needs no counters, so it gives independent evidence. It is also the case that matters for a real workload: a kernel on `D` reads data right after receiving it. If the L2 kept the data, the read is faster than one from HBM. `I` writes a linked list (each line holds the index of the next line, in a random order), and one thread on `D` follows it, so each load depends on the last one and cannot be prefetched or run in parallel.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **How the data reached `D`** | **Independent variable** | **Peer stores, read at once; peer stores, evicted, then read; local stores, read at once (fast reference); local stores, evicted, then read (slow reference)** |
| **Footprint** | **Independent variable** | **8, 32 MiB (fit in L2); 256 MiB (does not fit; all four cases should match)** |
| **Wait between push and read** | **Independent variable** | **0 and 5 ms** |
| Writers | Constant | 16 CTAs × 512 threads/CTA, peer or local |
| Reader | Constant | 1 CTA × 1 active thread on `D`, on a different SM from all local writers; one pass over the whole list |
| Line | Constant | 128 bytes, every byte written: the list index plus hashed filler. A partly written line could force an HBM read and look like a miss |
| List order | New on every transfer | Random, from the transfer number. Reading a stale list would produce a wrong hop count, which A14 reports |
| Repeats | Constant | 5 runs per case, 6 transfers per run, cases interleaved |

### Procedure

1. Check the two references first: the fast and slow references must differ clearly at 8 MiB and must match at 256 MiB. If they do not, the test cannot separate anything; fix the reader (L1 reuse, TLB, timer resolution) before going on.
2. The reader waits for the acknowledgement with acquire semantics, then starts the timer, reads the whole list, and stops the timer.
3. Run all cases, rotating order. Report the mean time per hop for each run.
4. Run a separate counter pass, if counters work, and compare L2 read hits and HBM reads with the timing result.

### Privileged access

None for timing. Counters are optional.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A13 | Peer-then-read matches the fast reference if the data stays in L2; matches the slow reference if it does not (hypothesis). Peer-then-evict matches the slow reference | Stay or not, without counters. A slow read cannot separate write-through that drops the line from bypass |
| A13, wait 5 ms | A rise after waiting shows the L2 lets data go by itself (hypothesis) | Time-limited hold from a long hold |
| A14 | Zero | Validity |
| C3, C2 (optional pass) | Read hits and low HBM reads go together with fast hops (hypothesis) | Confirms the timing |

## E8.4 · What happens to a line already in the destination's L2?

### Motivation

A pure bypass has a visible weakness. If `D`'s L2 holds an old copy and a store goes straight to HBM, a later read on `D` could return the old value. So this test puts a line in `D`'s L2 first, then stores to it from `I`. Getting old values would be a major finding. Getting new values rules out a simple bypass, since the L2 must have been told. Whether the next read hits then shows whether the line was updated or dropped.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **State of the footprint in `D`'s L2 before the push** | **Independent variable** | **Not present (evicted); present and clean (`D` read it twice); present and dirty (`D` stored to it)** |
| **Who stores afterwards** | **Independent variable** | **Peer stores (16 CTAs × 512 threads/CTA on `I`); local stores (16 CTAs × 512 threads/CTA on `D`)** |
| Footprint | Constant | 8 MiB, one pass |
| Reader | Constant | 16 CTAs × 512 threads/CTA on `D`, on SMs other than local writers; reads every word after the acknowledgement with acquire semantics |
| Repeats | Constant | 5 runs per case, 6 transfers per run, cases interleaved |

### Procedure

1. Confirm the prepared state with counters, where available: read hits on the second preparing read for "clean".
2. Run all cases and compare every word read with the expected last value. Keep the first bad addresses and their values.
3. On any old value, repeat with a stronger fence before the reader to rule out a mistake in the test or SM L1 reuse. The report keeps both results.
4. Compare the hit and miss counts of the stores and of the following read.

### Privileged access

None for the correctness result. Counters from E8.0 for hits and misses.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A14 | Zero old values under write-back and write-through; old values possible under bypass (hypothesis) | Bypass from the other two, without counters |
| C3 on `D`, stores | Write lookup hits on present lines, misses on evicted lines, if stores access the L2 (hypothesis) | Enter or bypass; a second check on E8.1 |
| C3 and C2 on `D`, read after store | Read hits: the line was updated. Read misses and HBM reads: the line was dropped (hypothesis) | Update from drop, which E8.2 cannot separate |

## E8.5 · Do incoming stores push out other data from the L2?

### Motivation

This is the "take space" step, and the result that matters most for interference. A kernel on `D` keeps a small buffer in L2. If incoming stores take L2 places, a large push evicts that buffer, its reads start going to HBM, and the kernel slows down with no change visible at the source. This extends the [plan 07 destination test](../data_store_congest/07_data_store_congest.md) (E7.7, not yet run) from "are flows competing?" to "is L2 space the thing they compete for?". The small-footprint push is the key control: it sends the same bytes per second, so bandwidth competition stays while capacity pressure is removed.

### Configuration

| Knob | Role | Values |
|---|---|---|
| **Push footprint, in addresses separate from the reader's buffer** | **Independent variable** | **No push; 1 MiB (same rate, no capacity pressure); 32 MiB; 256 MiB; 2048 MiB** |
| Communication | Constant | 16 CTAs × 512 threads/CTA on `I`, 8192 MiB total. Footprint and passes are coupled as in E8.2 |
| Reader on `D` | Constant, calibrated | Target: 16 CTAs × 512 threads/CTA repeatedly reading its own 16 MiB buffer. Calibrate CTAs and buffer size so that, alone, HBM reads are close to zero and the rate is stable |
| Run pattern | Constant | Each run alternates no-push and push transfers, 6 + 6 measured, as in plan 07 |
| Repeats | Constant | 5 runs per level; repeat the no-push reference at the end |

### Procedure

1. Calibrate the reader alone. Record its rate, read hits and HBM reads at each candidate setting.
2. Compare copy throughput with and without the reader, at each footprint. This comes first in the report.
3. Run the footprint sweep with alternating transfers; rotate level order between rounds.
4. Read the reader's rate and HBM reads, compared with no-push.
5. Run the counter pass, if available.

### Privileged access

None for rates. Counters from E8.0 are needed to confirm HBM reads. Without them, a fall in reader rate cannot be cleanly tied to cache misses, and the report must say so.

### Metrics

| Metric | Expected signature | Separates |
|---|---|---|
| A1 | Copy changes little, as in earlier plans (hypothesis) | Primary outcome, against copy without the reader |
| A9 | If stores take L2 space: reader rate falls with footprint, more than at 1 MiB. If only bandwidth is shared: 1 MiB and large footprints give the same fall (hypothesis) | Capacity from bandwidth |
| C2, C3 on `D` | HBM reads rise and read hit rate falls with footprint, if stores take L2 space (hypothesis) | Confirms the cause of the reader's slowdown |
| A8 | Reader active during the whole push | Validity |

## How the results answer the question

Conclusions come from several tests together. The report must state which were available.

| Results | Conclusion |
|---|---|
| E8.1 hub writes follow NVLink bytes; E8.2 low HBM writes at small footprint with link ratio about 1, and held ratio about 1; E8.3 fast read | **Write-back:** data passes through and stays in `D`'s L2 |
| E8.1 hub writes follow NVLink bytes; E8.2 HBM writes about equal stored bytes, held ratio about 0; E8.4 new values | **Write-through:** data accesses L2 but goes on to HBM. E8.3 and E8.4 say whether the line is updated or dropped |
| E8.1 no hub or other writes; E8.2 HBM writes about equal stored bytes; E8.3 slow read | **Bypass:** data does not access L2. E8.4 then says how correctness is kept |
| E8.2 link ratio well below 1 at small footprint | Stores merged before reaching `D`. E8.2 says nothing about `D`'s L2 at those points; use E8.1, E8.3 and E8.4 |
| Peer and local curves differ, but neither model fits | Report as it is. For example, a different hold size, or behaviour that depends on store size or fence |

## Interpretation limits

- **Counter names are not documentation.** NVIDIA does not describe the path, and "hub" may not mean what we assume. E8.1 ties it to NVLink traffic by dose and response. That shows a link between counts and traffic, not a physical route.
- **Counters are GPU-wide.** They cover every role on the GPU, and background. The no-stores control and a small receiver reduce this, but subtraction is not exact attribution.
- **One setup only.** Results apply to this memory (`ncclMemAlloc`, pinned device memory, mapped by NCCL), this driver and this store pattern. Another mapping or another GPU type may differ.
- **Local stores are a reference we assume uses the L2.** E8.0 and the local curve in E8.2 test this; if local stores do not behave as a write-back L2 would, the comparison must be rethought.
- **Eviction may be incomplete.** L2 replacement and address hashing are not documented, so the knee in E8.2 may sit below 50 MiB, and it need not be sharp.
- **Timing is not latency.** Time per hop is an average, and includes TLB and reader overhead. Only differences against the references in the same run set carry meaning.
- **No result names a queue.** Between NVLink receive and L2 there are parts this plan cannot see.

## Harness work

This new harness is planned and not yet written.

| Item | Plan |
|---|---|
| Program | New `common/src/peer_store_path.cu`, built as `common/bin/peer_store_path`. `store_congest` and `push_harness` stay unchanged |
| Reused | Copy loop, completion protocol, role table and SM-to-GPC map from `store_congest`; run wrapper guards from `run_one.py` with `--gpus` |
| New in the kernel | Phase sequence inside one launch; footprint and passes; eviction reader; linked-list writer and reader; prepare step; old-value check |
| New in the host | Counter collection for the method chosen in E8.0; `phases` array in the JSON output |
| New scripts | `run_peer_store_path.py`, `analyze_peer_store_path.py` (with figures, chart rules as usual) under `common/scripts/` |

## Order of work

| Step | Work | GPU time (rough) |
|---|---|---|
| 1 | Harness; plan-07 throughput check | Under 15 min |
| 2 | E8.0 counter validation. **Stop and report before going on** | 1–2 h, mostly debugging |
| 3 | E8.2 base sweep (main test) | About 30 min |
| 4 | E8.1 | About 20 min |
| 5 | E8.3, E8.4 | About 30 min each |
| 6 | E8.2b, E8.2c | About 30 min |
| 7 | E8.5, if the earlier steps make it worthwhile | About 1 h |
| 8 | Report in `peer_store_l2_path/README.md` | None |
