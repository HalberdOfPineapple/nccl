# Thread-store experiment harness

Build from this directory with `make`, using the existing in-tree NCCL build. The executable is ignored under `bin/`. CUDA C++17 and the H100 `sm_90` target are required. The current harness deliberately accepts only the user-authorized physical device indices 4–7; the guarded wrapper reads the physical device pair from each config (default 4 → 5). The destination follow-up uses 6 → 7 under the updated user allocation. Do not set `CUDA_VISIBLE_DEVICES` when invoking them, because their configs and snapshots use physical indices.

The precise workload, completion protocol, and stable output field meanings are documented in [the common setup](../observation/00_common_setup.md). This implementation is separate from archived experiments; it does not reuse their metric schema or results. One process creates two host threads, each with its own GPU, NCCL rank, nonblocking stream, registered window, and cooperative fused launch. Shared-memory reservation and occupancy checks enforce one block per SM; runtime records validate distinct SM IDs. Full destination correctness is checked after every iteration.

Each `.cfg` contains `key=integer` lines, with blank lines and full-line comments allowed. Supported keys are `source_gpu`, `destination_gpu`, `ctas`, `threads`, `co_ctas`, `kind`, `pacing`, `iterations`, `warmup`, `detailed`, `global_stride`, `communication`, `vector_reads`, and `solo_ns`. `kind` is 0 for no pressure, 1 for source FMA, 2 for source reads, and 3 for destination reads. `vector_reads=1` selects 16-byte co-runner reads with independent checksum chains and unrolling; it does not change the communication load/store width. With `communication=0`, control blocks run a fixed interval (20 ms by default; `solo_ns` selects a duration in nanoseconds) instead of copying, providing a standalone co-runner control with the same geometry/reservation. `detailed=0` retains start/end, role/SM, co-runner, and correctness observations but omits issue/checkpoint/fence/publication records.

From the repository root:

```bash
make -C experiments/h100x2/thread_store/common
python3 experiments/h100x2/thread_store/common/scripts/run_one.py \
  experiments/h100x2/thread_store/observation/shared_resources/configs/healthy.cfg \
  /tmp/thread_store_healthy.json
bash experiments/h100x2/thread_store/common/scripts/run_pair.sh \
  /tmp/thread_store_pairs 5 \
  experiments/h100x2/thread_store/observation/shared_resources/configs/healthy.cfg \
  experiments/h100x2/thread_store/observation/shared_resources/configs/source_memory_100.cfg \
  source_memory_100
python3 experiments/h100x2/thread_store/common/scripts/collect.py /tmp/thread_store_pairs
python3 experiments/h100x2/thread_store/common/scripts/summarize.py /tmp/thread_store_pairs
```

The wrapper checks that both selected GPUs are idle, allows a brief cooldown for context teardown, records before/after device XML and read-only NVLink counters, and samples operating-state telemetry during the process. It refuses to overwrite a raw output. Pair/campaign resume logic skips completed JSON files; use a new result directory for a new session. The current campaign scripts use the dated 2026-10-27 directory explicitly; update that destination and record new provenance for later sessions.

The independent statistical unit is a harness invocation; nested transfers quantify within-run variation. `collect.py` produces `transfers.csv` and `runs.csv`; `summarize.py` produces paired effect tables and JSON; `make_figures.py` requires Matplotlib and renders standalone PNGs. The current session's plotting environment is `/tmp/thread_store_analysis_env`, separate from the repository. Exact Python dependencies are recorded with the dated result metadata.

Run `profile_campaign.sh` only after primary timing has finished and after checking the selected GPUs are still allocated and idle. The validated workflow collects CUDA/NVTX timelines without clock changes; `--profile-range` adds a source-thread host NVTX range covering both ranks and host barriers outside the device measurement. The optional `--profile-range-destination` places that host range on the destination thread. Kernel replay hung, while CUDA start/stop and NVTX application-range attempts produced no valid Nsight Compute range in this session; logs are retained under results, and hardware-counter/source-PC collection remains unvalidated. Completion and role overlap must still be checked after collection. A counter report whose execution lost the intended interaction is not evidence about that interaction. Binary profiler artifacts are ignored under results.

## Destination-pressure follow-up on GPUs 6 → 7

`run_destination_followup.py` implements the three steps in the shared-resource plan: communication alone, destination reads alone, and interleaved combined/reference pairs. It retains complete CTA/thread geometries, rotates condition order between rounds, and adds matched-duration reads-only comparisons and minimal/detailed probe pairs. It refuses to resume after a source/binary hash change. Configurations are saved under `configs/destination_followup_gpu6_7/`; local raw records are under `results/2026-10-27-gpu6-7/raw/`.

After timing finishes, run `analyze_destination_followup.py <result-directory>` with the plotting Python environment, then `profile_destination_followup.py <result-directory>` for separate validated endpoint timelines. The new scripts do not change the CUDA workload. Hardware counters remain unvalidated; the timeline script does not retry known failing counter-replay modes.

Raw records, snapshots, telemetry, profiler outputs, and logs are ignored by Git and kept locally. Configs, scripts, compact summaries, figures, and reports remain versioned. Reproducing a summary requires its local raw directory or rerunning the recorded configs; a clean checkout does not contain the raw measurements.

## Saturation calibration correction

The current harness supports up to 132 communication CTAs and 1024 threads/CTA, subject to the actual device and occupancy checks. `payload_mib` selects 1–4096 MiB (default 256). `copy_batch` selects 1, 2, 4, or 8 independent scalar source loads followed by their matching scalar peer stores per thread (default 1). Completion semantics and full-buffer validation are unchanged; batch size is an explicit request-generation setting, not a measured metric. The output records `copy_batch`; historical files without that key used one element at a time.

`run_saturation_calibration.py`, `run_payload_calibration.py`, and `run_batch_calibration.py` record separate calibration revisions under `results/2026-10-27-saturation-gpu6-7/`, with raw files ignored locally. Each runner pins its source/binary hashes and refuses to resume under a later revision. Prior binaries are retained under ignored `bin/push_harness_before_*` and source snapshots under the local raw directory. Reproducing an older phase requires its recorded revision; do not bypass the hash check. The run wrapper accepts `--binary` for explicit historical comparisons.

## Independent role thread counts for source-memory pressure

The source-memory plan needs communication threads per CTA (Tc) and read threads per CTA (Tm) to vary independently. Three config keys implement this; older configs without them behave as before.

| Key | Meaning | Default |
|---|---|---|
| `threads` | Active communication/control threads per CTA (Tc). Output key `threads` keeps this meaning. | 256 |
| `memory_threads` | Active co-runner threads per CTA (Tm) | same as `threads` |
| `block_threads` | Physical block size of the fused launch, shared by every role | larger of `threads` and `memory_threads` |
| `read_unroll` | Requested independent 16-byte loads per read lane before accumulation: 1, 2, 4, 8 or 16; each value is a separately compiled kernel | 4 |

Lanes beyond a role's active count do no work but still take part in every CTA barrier, so they keep their registers reserved. The source-memory campaign holds `block_threads=1024` in every run, so changing Tm never changes the communication CTAs' physical shape. A copy-only reference must use the same `read_unroll` instantiation as its pressure partner. The JSON output appends `memory_threads`, `block_threads` and `read_unroll`; `collect.py` treats records without them as the earlier equal-thread layout.

The vector reader in this revision trims each read CTA's slice of the 1 GiB buffer to whole 256 KiB bursts, so no burst wraps inside the slice, and it issues full load batches without a bounds predicate. The earlier wrap branch and predicated loads made the compiler issue loads in pairs. At destination reads 131 × 512 the logical rate rose from about 2072 to 2678 GB/s in the validation runs, while the copy-only reference stayed within about 0.1%. Results from the previous reader are therefore not comparable with this revision. Compiler grouping of loads is limited: in SASS, requested depths 8 and 16 group at most about four to five loads before accumulation, so a larger `read_unroll` does not guarantee proportionally more outstanding requests.

`run_source_memory.py <phase>` runs the source-memory campaign one phase at a time (validate, reads, unroll, communication, controls, contention, mutual, probes) under `results/2026-10-27-source-memory-gpu6-7/`. Phases after `unroll` read the calibrated choices from that directory's `selection.json`. `analyze_source_memory.py` produces the tables and figures. The previous binary is retained locally as `bin/push_harness_before_source_memory`.

## Rate-controlled memory pressure (kinds 4 and 5)

The fused kernel can host the [rate-controlled memory co-runner](../../../h100x1/memory_co_runner/common/README.md) on the source (`kind=4`) or the destination (`kind=5`). The header is included from `experiments/h100x1/memory_co_runner/common/include/`.

| Key | Meaning | Default |
|---|---|---|
| `co_ctas` | Pressure CTAs | 0 |
| `chunk_kib`, `chunks_in_flight` | Bytes in flight per pressure CTA | 64, 3 |
| `target_gbps` | Total target read rate of all pressure CTAs, GB/s; 0 = no limit | 0 |
| `pressure_idle` | 1 = pressure CTAs start and wait for the stop signal without reading (the reference) | 0 |
| `pressure_alternate` | 1 = launches alternate idle, reading, idle, …; 2 = reading first. Records list idle or reading per measured transfer in `pressure_idle_by_iteration`. | 0 |
| `shared_bytes` | Dynamic shared memory for every CTA of the launch on both GPUs; must keep one CTA per SM | just over half an SM |

`--gpus SOURCE,DESTINATION` (harness and `run_one.py`) replaces the config's devices, so one config runs on any allocated pair. The allocation from 2026-10-28 is GPUs 0, 1, 6, 7; GPUs 4 and 5 are no longer accepted. `run_rate_pressure.py {source|destination} PHASE --gpus S,D` runs the campaign and `analyze_rate_pressure.py` makes its tables and figures. The build before this change is kept locally as `bin/push_harness_before_rate_reads`.

## Store-congestion harness (plan 07)

[Plan 07](../observation/data_store_congest/07_data_store_congest.md) uses two separate programs; `push_harness` is unchanged.

`bin/gpc_probe GPU REPEATS OUT.json` launches thread-block clusters of 2, 4, 8 and 16 CTAs (one CTA per SM) and records each CTA's cluster and SM ID. `run_gpc_map.py --gpus 4,5` runs it under the GPU locks; `build_gpc_map.py` groups SMs that shared a cluster into GPCs and writes `observation/data_store_congest/configs/gpc_map/gpu<N>.json`.

`bin/store_congest CONFIG OUT.json --gpus S,D --gpc-map MAP.json` launches one CTA on every source SM (cooperative launch, 1024 threads, one CTA per SM). Each CTA reads its SM ID and takes the role the role table gives that SM. The host builds the table from the map and the config, and refuses a map whose GPU UUID is not the source GPU's.

| Key | Meaning | Default |
|---|---|---|
| `copy_ctas`, `copy_threads` | Copy CTAs × active threads/CTA | 64, 512 |
| `copy_gpcs`, `writer_gpcs` | Bit mask of GPCs (map numbers) the role may use | 255 (all) |
| `copy_placement`, `writer_placement` | 0 = spread: one SM per allowed GPC in turn; 1 = packed: all SMs of the largest allowed GPC first | 0 |
| `writer_ctas`, `writer_threads` | Writer CTAs × active threads/CTA (threads must divide 4096) | 0, 1024 |
| `target_gbps` | Total target write rate of all writer CTAs, GB/s; 0 = no limit. Each CTA writes one 64 KiB block, then waits for the next block's start time | 0 |
| `writer_idle`, `writer_alternate` | Idle writers (the reference); alternate idle and writing launches inside one run (1 idle first, 2 writing first) | 0, 0 |
| `use_unmapped` | Allow roles on unmapped SMs after the mapped SMs are used | 0 |
| `communication`, `solo_ns` | 0 = no copy; copy CTA 0 times a window of `solo_ns` after all writers start | 1, 20 ms |
| `payload_mib`, `warmup`, `iterations`, `detailed`, `block_threads`, `shared_bytes` | As in `push_harness` | 2048, 3, 12, 1, 1024, just over half an SM |

Copies are assigned first, then writers, from the remaining SMs. The store-only copy issues one 4-byte peer store per loop step. Its `copy_batch` key (stores per loop step) was removed on 2026-10-30: stores do not wait for a reply, so it could not change the stores in flight, and values above 1 only added instructions; configs that still contain it are rejected. Records made before then carry `copy_batch` (schema 1) and come from the earlier build. `run_store_congest.py PHASE --gpus S,D` runs the plan-07 phases (`copy_sweep`, `copy_floor`, `copy_placement`, `writer_sweep`, `writer_rate`, `writer_placement`, `contention`, `read_contrast`, `tma_writer_sweep`, `tma_writer_rate`, `tma_store_contrast`, `destination`; 3 runs per setting from 2026-10-30). `analyze_store_congest.py` (calibration) and `analyze_store_contention.py` (copy with a co-runner) make the tables and figures with the plotting environment. `bin/store_congest_tma` is the same source built with `-DSTORE_CONGEST_TMA`: it adds `writer_kind` (0 thread stores, 1 TMA bulk reads from plan 04's read co-runner, 2 TMA bulk stores) and `chunks_in_flight` (64 KiB chunks per co-runner CTA, 1–8); TMA kinds use threads 0 and 32 of a co-runner CTA, TMA reads need `shared_bytes` of at least chunks × (64 KiB + 8). `run_writer_counters.py --gpus S,D` runs the writer alone under `ncu` (application replay; kernel replay hung earlier; clock control off), and `analyze_writer_counters.py` compares requested bytes with DRAM and L2 counters. `bin/store_congest_dest` is the same source built with `-DSTORE_CONGEST_DEST`: both ranks launch one CTA per SM with their own role table, with `writer_rank` (0 source, 1 destination), `receiver_gpcs` and the option `--destination-gpc-map` (phase `destination`). Mixed runs set `writer_alternate` (1 idle first, 2 writing first) so each run holds 6 idle and 6 active transfers. The allocation for plan 07 is GPUs 4, 5 (from 2026-10-29) and 0, 1 (from 2026-10-30); `run_one.py` accepts them with `--gpc-map`.

## Peer-store path harness (plan 08)

[Plan 08](../observation/peer_store_l2_path/08_peer_store_l2_path.md) has its own program; `push_harness` and `store_congest` are unchanged.

`bin/peer_store_path CONFIG OUT.json --gpus S,D [--profiled-rank R]` runs each phase of a transfer as its own named kernel, so `ncu`'s per-launch rows separate the phases. Only GPUs 6 and 7 are allowed. `scripts/run_peer_store_path.py` writes the configs and runs them, with `ncu --replay-mode application` (kernel replay hung on this two-GPU launch). `scripts/peer_store_counters.py RAW_DIRECTORY` prints each phase's counters, and `scripts/analyze_peer_store_path.py RESULTS_DATE_DIRECTORY` writes the summary and figures. Results are in the [report](../observation/peer_store_l2_path/README.md).
