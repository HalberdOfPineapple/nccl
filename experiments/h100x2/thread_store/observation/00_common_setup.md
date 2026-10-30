# Common setup for the thread-store observation series

This file defines the implemented measurement subset for the current [shared-resource experiments](shared_resources/04_shared_resource_contention.md). Other observation plans remain exploratory; legacy metric IDs in those plans must be reconciled before implementing them. The previous archive is not the metric registry for this harness.

## Machine, roles, and workload

| Property | Current setup |
|---|---|
| Execution authorization | User allocated physical GPUs 4–7; current two-GPU series uses 4 → 5 only |
| Hardware | NVIDIA H100 80GB HBM3; 132 SMs and 50 MiB L2 per selected GPU (queried through CUDA attributes); topology reports NV18 between 4 and 5 |
| Source identity | GPU 4, UUID `GPU-e4e41214-ef43-a1c0-80ab-391fd7b067a5` |
| Destination identity | GPU 5, UUID `GPU-4355f0c0-dd9b-8592-3d88-d250d9730a63` |
| Other tenants | GPUs 0–3 are busy; do not use them. Guards check memory/utilization on selected devices before every invocation; process enumeration alone is insufficient in this container. |
| Software | In-tree NCCL `build/lib/libnccl.so.2.32.3`; CUDA compiler from `/usr/local/cuda`; Nsight Systems 2026.4.1; Nsight Compute 2025.3.1; driver 595.71.05. Save exact compiler/build metadata with results. |
| Payload | 256 MiB scalar `uint32_t` bit patterns (equivalent 4-byte transfer width to planned `int32_t`); local source loads and NCCL LSA peer stores |
| Default issuing geometry | 32 CTAs × 256 threads, contiguous CTA regions; baseline sweeps 4/8/16/32 CTAs, 64/128/256/512 threads, and global stride |
| Co-runner | Source arithmetic or source/destination streaming reads from a separate 1 GiB allocation; 256 threads/CTA; scalar and vector read calibration variants; measured M ladder rather than assumed saturation |
| Spatial separation | Cooperative fused launch; 116,736 bytes dynamic shared memory/CTA plus static shared state; occupancy API must report one resident CTA/SM; recorded SM IDs must be unique per launch |
| Clock policy | Unchanged; no clock locking or power-limit changes. Profilers must use `--clock-control none`. |

## Completion protocol and measurement boundaries

Each rank registers one symmetric NCCL window containing the destination payload area and cache-line-separated control signals. Source rank 0 accesses destination rank 1 with `ncclGetLsaPointer`. An independent local source allocation supplies scalar loads. Destination receiver/control and destination memory co-runner roles share one fused kernel on distinct SMs; source communication and source co-runner roles likewise share one fused kernel.

Every local co-runner records its start before incrementing a readiness counter. Source communication CTAs wait for all local co-runners and a destination-ready signal; the destination publishes readiness after its local co-runners are ready. This protocol requires all blocks to fit simultaneously; the cooperative launch and occupancy check enforce that capacity. No role assumes CTA index is a physical SM ID.

After the store loop, every writing thread executes a system fence; a CTA barrier precedes its leader's release publication to the destination-local completion flag. The destination's single polling lane acquires every CTA completion flag and releases an acknowledgement into source memory. Every source CTA acquires that acknowledgement before recording its end. The last source communication CTA stops source co-runners; the destination receiver stops destination co-runners after acknowledgement publication. Kernel launches happen on separate host threads so mandatory cross-GPU progress does not depend on the profiling tool returning from a serialized host launch.

Completed-transfer duration is `max(source CTA end_ns) - min(source CTA start_ns)` on the source GPU's timer. It includes CTA start skew, stores, ordering, publication, receiver polling, and acknowledgement observation. It excludes host setup, initialization, correctness checking, and co-runner cleanup. Each source CTA's issue interval includes source loads, stores, checkpoint branches, and the following CTA synchronization; it is not pure instruction-issue latency. Receiver-local waiting can include host launch skew before source execution and is not one-way fabric latency. Never subtract raw timers across GPUs.

Every iteration changes the source pattern and poisons destination memory before measurement. A destination verification kernel reads the entire payload after both communication kernels finish and checks every element. Any mismatch, invalid timing, or duplicate SM mapping aborts the run. Validation traffic is outside the device measurement interval but remains visible in coarse whole-process telemetry.

## Metric registry

Tier A records are produced by the harness. Tier B/C are separate acquisition passes; they are marked unavailable unless their collection is validated. JSON names and the schema version are stable interfaces. Units ending in `ns` come from `%globaltimer`; no cycle-to-time conversion is used. Raw detailed-only fields are zero-initialized when `detailed=0` or no transfer is performed; these are missing-value sentinels, not zero-duration observations. Derived tables represent the uncollected phases as empty/null values.

| ID | Semantic observation | Definition / acquisition | Scope and limitation |
|---|---|---|---|
| A1 | Acknowledged completed-transfer duration and useful throughput | Source max end minus min start; payload bytes divided by duration, decimal GB/s | End-to-end device protocol, not raw NVLink capacity |
| A2 | CTA start time | `start_ns` per source communication CTA | Includes CTA start skew; same-GPU comparisons only |
| A3 | CTA issue-loop duration | `issue_end_ns - start_ns` | Includes local loads, stores, sparse probes, and CTA sync; detailed mode only |
| A4 | CTA ordering interval | `fence_end_ns - issue_end_ns` | All writers fence, then synchronize; not a direct physical drain latency; detailed mode only |
| A5 | CTA publication interval | `publish_end_ns - fence_end_ns` | Leader's release publication and bookkeeping; detailed mode only |
| A6 | Acknowledgement waiting and receiver-local wait | Source `end_ns - publish_end_ns`; receiver `end_ns - start_ns` | Source waiting includes remaining CTAs; receiver wait includes readiness and launch skew |
| A7 | Sparse software issue progress | Up to eight `checkpoints_ns` values per source CTA, equal work increments | Leader-lane loop progress, not remote arrival; no additional chunk fences |
| A8 | Co-runner active interval | Per co-runner CTA `start_ns`, `end_ns`; minimum coverage of communication/control interval on same GPU | Covers the role lifetime, including startup/tail; not instruction-by-instruction overlap |
| A9 | Achieved co-runner work and checksum | `work_units`: arithmetic = 256 rounds × 8 FMA × threads × 2 FLOPs; reads = 256 KiB logical reads per CTA burst | Rates use sum work over union active interval; not exact work during communication alone and not physical HBM bytes |
| A10 | CTA role and physical SM placement | `role`, `sm_id`, CTA index; per-device grid, SM count, shared reservation | Validates observed spatial separation, not GPU-wide resource independence |
| A11 | Full-payload correctness | Destination mismatch count must be zero for each iteration; run `verified` flag | Solo controls do not transfer data; `verified` means applicable checks passed, not a transfer happened |
| A12 | SM-to-GPC map | Per GPU, from cluster launches ([plan 07](data_store_congest/07_data_store_congest.md), `bin/gpc_probe`, `build_gpc_map.py`): SMs that ran CTAs of one cluster share a GPC; GPC numbers follow each group's smallest SM ID; an SM never placed in a cluster of 4 or more CTAs is "unmapped" (-1). Saved as `data_store_congest/configs/gpc_map/gpu<N>.json` with the GPU UUID; every plan-07 record repeats the map (`gpc_of_sm`) and the role of each SM (`role_of_sm`) | Measured, not documented by NVIDIA; does not place unmapped SMs; the GPC numbers are labels, not hardware indices |
| B1 | CUDA kernel/API timeline | Separate Nsight Systems trace | Fused CTA roles need A8/A10; kernel intervals alone do not prove role overlap |
| B2 | Device operating state | Snapshot XML plus sampled clocks, power, temperature, utilization, and clock-event reasons | Whole-invocation samples include setup and verification; sub-transfer causality requires finer alignment |
| B3 | Directional NVLink observations | Read-only `nvidia-smi nvlink -gt d` before/after; retain tool errors if unsupported | Device/link aggregate counters include background traffic; no counter resets |
| C1 | Source endpoint memory demand | Supported `dram__bytes_read.sum`, write bytes, throughput; resolved via installed profiler | Fused aggregate, not per-role or per-operation traffic |
| C2 | Destination endpoint memory demand | Same supported DRAM families collected in separate destination pass | Same aggregation boundary; incoming stores need not immediately reach HBM |
| C3 | Endpoint L2 demand/cache behavior | Supported L2 sector/byte and hit-rate families | Device/window evidence; no exact communication attribution by subtraction |
| C4 | Scheduler/issue activity | Supported scheduler statistics, eligible/active warp families | Fused aggregate unless a more specific attribution is available |
| C5 | Communication load-path waiting | Source-PC/source-correlated samples at local loads/dependencies | Sampling support and sample counts required; stall names do not identify physical queues |
| C6 | Peer-store/ordering-path waiting | Source-correlated samples at peer stores and fence instructions | May reflect either endpoint; no individual remote-store latency |
| C7 | Arithmetic activity and generated instructions | Supported instruction/compute metrics plus SASS and compiler register/spill report | Generated SASS proves retained instructions; aggregate issue metrics mix roles |
| C8 | Profiling validity and acquisition metadata | Versions, selected metrics, replay/cache/clock policy, overlap/correctness checks, timeout/error log | Invalid or incomplete profiler runs are not usable counter evidence |

## Run and analysis protocol

Build with `make -C experiments/h100x2/thread_store/common`. The harness takes one config and one new JSON output path; `scripts/run_one.py` adds device guards, snapshots, coarse telemetry and a 45-second invocation timeout. `scripts/run_pair.sh` runs five interleaved healthy/perturbed pairs. Defaults are three warm-ups and twelve retained transfers per invocation; standalone co-runner controls use two warm-ups and five retained 20 ms windows. Preserve the difference between independent invocations and transfers nested within them.

The primary comparison unit is one invocation's median, paired by run order. Report the distribution of those medians, paired relative changes, and `(median_P - median_H) / IQR_H`, explicitly stating the metric direction. If healthy IQR is zero, leave the standardized effect undefined and report the absolute difference. Plot phase distributions and achieved pressure rather than only throughput. Baseline geometry sweeps repeat in rounds to reduce confounding by run order. Instrumentation-overhead pairs compare minimal boundary/placement probes to detailed checkpoint/phase probes; neither mode is completely uninstrumented.

Save raw JSON, snapshots, logs, resolved counter lists and derived CSVs under the dated results directory; profiler binaries are ignored. `collect.py` derives tables and `make_figures.py` generates report figures. No hand-edited raw output. Every session is recorded in the experiment README, including deviations, unavailable observations, and the shared-machine caveat.

## Access and measurement cost

| Capability | Requirement | Fallback |
|---|---|---|
| Selected GPUs | Existing user allocation 4–7, idle checks before every invocation | Stop on occupied selected devices; never touch GPUs 0–3 |
| Tier A | Ordinary CUDA/NCCL execution with LSA support | No substitute for correctness/placement; abort on failure |
| Telemetry/link observations | Read-only NVML/`nvidia-smi` access | Retain errors and do not treat unavailable as zero |
| Hardware profiling | Supported installed Nsight version and enabled performance counters | Device phase/progress evidence; disclose unresolved mechanism |
| Clock intervention | Additional user authorization | Observe existing clocks; disable profiler clock control |

## Binary revisions within the current session

The initial scalar-read campaign uses the source snapshot `shared_resources/results/2026-10-27/harness_source_scalar.cu` and hashes in `provenance.json`. The later vector-read campaign adds the `vector_reads` config/JSON field and changes fused-kernel resource allocation through the compiled variant; it therefore remeasures matched healthy references rather than pooling raw times with the initial binary. Its source/binary hashes are retained in `vector_provenance.json`. Both variants retain the same scalar communication operations, completion protocol, and metric definitions. Exact rates and outcomes belong in the experiment report.

The matched-duration destination-workload follow-up uses `solo_ns=3300000` and five interleaved reads-alone/reads-with-copy pairs at 100 and 131 read CTAs, with three warm-ups and twelve retained windows per invocation. Communication throughput remains the primary outcome of the experiment. Analyze it first using the original paired copy-alone / copy-with-reads runs; the copy-disabled follow-up arm is not a healthy-copy reference. The matched-duration follow-up adds destination logical read rate as a secondary observation, and its copy-enabled throughput is reported separately without inventing cross-cohort pairs. The host-only duration/parser and NVTX additions preserve the vector campaign’s exact generated device SASS; `final_provenance.json` records that check. Acquisition outcomes and logs are documented in the report.

## Destination-pressure follow-up allocation

The subsequent three-step follow-up uses the user's new allocation, physical GPU 6 → GPU 7. Earlier results above remain associated with GPU 4 → GPU 5; comparisons must not mix these sessions. The CUDA source and binary are unchanged, while the Python run wrapper now selects guards and telemetry from the config's physical GPU IDs.

| Setting | Follow-up value |
|---|---|
| Source | GPU 6, UUID `GPU-3ebe5de5-496c-2d06-ac87-55f171a54e27` |
| Destination | GPU 7, UUID `GPU-a8a608db-0a7a-970b-210c-28f9c04360c8` |
| Topology | NV18; record fresh snapshots for this pair |
| Communication settings | 16, 32, 64 CTAs at 256 and 512 threads/CTA, plus the wider communication-only calibration in the plan |
| Destination reads | 8, 32, 100, 131 CTAs at each group thread count; vector reads, no pacing |
| Raw measurements | Local ignored `shared_resources/results/2026-10-27-gpu6-7/raw/` |
| Reproducibility | Configs, scripts, build hashes, derived summaries and figures retained in Git; raw files required for reanalysis |

The measured rate continues to use the complete co-runner active interval, not just steady communication. No new metric IDs or measurement definitions are introduced.

## Corrected communication reference for destination-pressure tests

The [subsequent calibration](shared_resources/results/2026-10-27-saturation-gpu6-7/README.md) extends the former software limits before selecting a reference. Use `shared_resources/configs/saturated_destination_reference.cfg`: physical GPU 6 → GPU 7, **128 communication CTAs × 512 threads/CTA**, **2048 MiB**, **8 independent scalar elements per thread batch**, CTA-local stride, no destination reads, and one destination control CTA × 512 threads. This selection applies to the destination-pressure experiment; source-side co-runner experiments need their own resource-budget calibration.

The current parser supports `payload_mib` (default 256, up to 4096) and `copy_batch` (1, 2, 4, 8; default 1). Raw JSON adds `copy_batch`; older records without it used batch 1. Existing metric meanings are unchanged. Software checkpoints count stores issued, even when source loads have been grouped before those stores. Compilation changed across calibration phases, so historical baselines must not be pooled with this reference. Fresh nearby controls remain required in any later pressure run.

## Rate-controlled memory pressure (2026-10-28)

New pressure kinds 4 (source) and 5 (destination) use the rate-controlled bulk-copy reader. Their read rate (A9) counts finished 64 KiB chunks over the pressure CTAs' active time; it is requested bytes, not counted HBM traffic. Records also contain the target rate, an idle mode, and, for alternating runs, whether each measured transfer was idle or reading. No new metric ids. GPU allocation from this date: physical GPUs 0, 1, 6, 7; each experiment stays on one pair, and the pair is recorded in every raw record. See the [report](shared_resources/README.md#rate-controlled-memory-reads-shared-method).

## Store-congestion harness (2026-10-29)

Plan 07 uses a separate binary, `common/bin/store_congest` (source `common/src/store_congest.cu`); `push_harness` is unchanged. One CTA runs on every source SM and takes the role that the role table gives its SM: copy, writer, or idle. The copy computes its values in registers and does no source loads; its completion protocol and timers (A1–A7) are the same as in `push_harness`. Copy parts start on 128-byte boundaries. The writer's work (A9) counts 64 KiB blocks of 16-byte stores into a local 1 GiB buffer; its rate is requested bytes over the writers' active time, not counted HBM traffic. Records give each CTA's role as text (`copy`, `writer`, `idle`; `receiver` on the destination), its SM ID and its role index. GPU allocation for plan 07 from this date: physical GPUs 4 and 5. From 2026-10-30 also GPUs 0 and 1. A second build, `common/bin/store_congest_tma` (same source, `-DSTORE_CONGEST_TMA`), adds TMA co-runners in the writer role: bulk reads (plan 04's read co-runner; A9 counts 64 KiB chunks read, checked by a checksum) and bulk stores (A9 counts 64 KiB chunks written, every word checked). See the [report](data_store_congest/README.md).
