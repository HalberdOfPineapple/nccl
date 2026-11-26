# Scalability series: AllReduce from 2 to 8 GPUs

**Status (2026-11-16): the user approved all four plans and gave all 8 GPUs for 8 hours, with 8-GPU runs first.** Each plan has its own directory with a plan file and a report.

## Goal

Measure how fast one AllReduce call is when the number of GPUs grows from 2 to 4, 6 and 8, for three ways to implement it in this NCCL checkout: Ring, NVLS, and plain loads and stores to peer memory. The target workload is LLM training and inference, so the payload is large (tens of MiB to several GiB per GPU).

The common question of the series: **when the GPU count grows, how does the AllReduce rate of each implementation change, and what observation explains the change?**

This series has no injected problem. The comparison is between implementations and between GPU counts. It gives the no-interference reference that later observation series need at each GPU count.

## Where this series sits

| Item | Choice | Note |
|---|---|---|
| Tree | `experiments/h100-scale/` | The [conventions](../../../CLAUDE.md) put experiments whose controlled variable is the GPU count in `<gpu_type>-scale/` (rule added 2026-11-16) |
| Data path level | `allreduce` | The series compares several implementations of one collective, so the level names the collective. It touches three rows of the [problem matrix](../../../../notes/scale_up_communication_problem_matrix.md): thread-issued peer loads and stores, NVLS-based collective, and Ring (not yet a row) |
| Series | `scalability` | What changes is the GPU count |
| Harness | [`../common/`](../common/README.md) | A runner script around nccl-tests, modeled on [`run_nccl_check.py`](../../../h100x4/nvls/common/scripts/run_nccl_check.py). No new kernel |

## The four implementations compared

"Arm" means one implementation under comparison. Full selection settings are in [00_common_setup.md](00_common_setup.md).

| Arm | Code | What one GPU does | GPU counts |
|---|---|---|---|
| Ring | [`runRing`](../../../../src/device/all_reduce.h#L14) | Data moves around a ring of GPUs in chunks: first a reduce pass, then a copy pass | 2, 4, 6, 8 |
| NVLS | [`NCCL_ALGO_NVLS`](../../../../src/device/all_reduce.h#L390) | Uses multimem instructions: the switch sums one load over all GPUs and copies one store to all GPUs. Data goes through NCCL's own buffers | 2 (to be checked), 4, 6, 8 |
| Symmetric multicast | `AllReduce_RSxLDMC_AGxSTMC` in [all_reduce.cuh](../../../../src/device/symmetric/all_reduce.cuh) | Same multimem instructions, applied directly to the user's registered buffers | 4, 6, 8. Not 2: [sym_kernels.cc:195](../../../../src/sym_kernels.cc#L195) turns multimem off for 2 GPUs |
| Symmetric plain load/store | `AllReduce_RSxLD_AGxST` in the same file | Loads its part from every peer and sums it, then stores the result to every peer, one peer at a time | 2, 4, 6, 8 |

Why two NVLS arms: the plain load/store version exists only as a symmetric kernel. "Symmetric multicast" is its twin: same kernel family, same buffers, only the instruction type differs. "NVLS" is the long-standing path that `NCCL_ALGO=NVLS` selects. If only one NVLS arm is wanted, keep the one that matches the question: "NVLS" for "what NCCL users get", "Symmetric multicast" for "multimem against plain instructions".

The TMA kernel (`AllReduce_RSxTmaLD_AGxTmaST`) is left out: NCCL enables it only on newer GPUs ([sym_kernels.cc:141](../../../../src/sym_kernels.cc#L141)), not on H100.

## Results (2026-11-16)

All four plans ran on 2026-11-16 between 06:29 and 09:25 UTC: 868 runs, 0 rejected. Main finding: **the order of the arms flips with the GPU count.** At 2 GPUs Ring is fastest; at 8 GPUs NVLS and symmetric multicast are 30–33% faster than Ring in algBW. Ring and NVLS keep a nearly constant busBW, and the link counters confirm their byte factors exactly. No arm uses more than 76% of the NVLink rating.

| Plan | Report | Main result |
|---|---|---|
| 01 | [Selection check and CTA calibration](selection_calibration/README.md) | All 15 cells run the intended code (NVLS also at 2 GPUs). Symmetric multicast needs only 2–4 CTAs. NCCL's CTA choice holds plain load/store back at 2 GPUs (0.61 of its best) and 4 GPUs (0.84). Ring at 2 GPUs is still rising at NCCL's 64-CTA limit |
| 02 | [GPU-count scaling](gpu_count_scaling/README.md) | algBW at 2 → 8 GPUs: Ring 339.5 → 207.7, NVLS 206.5 → 270.3, symmetric multicast 251.9 (4 GPUs) → 276.2 GB/s. The symmetric kernels drop at 6 GPUs, and at 6 GPUs depend on the GPU set (up to 14%); the uneven split of 2 GiB is ruled out |
| 03 | [Payload size](payload_size/README.md) | GPU count decides first: against Ring, NVLS goes from 0.60 (2 GPUs) to 1.31 (8 GPUs) at 8 GiB, symmetric multicast from 1.03 (4 GPUs) to 1.31. Payload matters second, for classic NVLS only: at 32 MiB it is no faster than Ring at any GPU count, while symmetric multicast is 1.22–1.52 times Ring. Ring is the most payload-sensitive arm at 2 GPUs (corrected 2026-11-18) |
| 04 | [Link bytes per payload byte](link_bytes/README.md) | Measured link bytes equal 2(N − 1)/N (Ring, plain) and 1 + 1/N (multimem) within 0.1%. Share of the 478 GB/s rating: Ring 0.71–0.76, NVLS 0.63–0.65, symmetric multicast 0.58–0.66 |
| 06 | [NCCL's example multicast kernels](custom_mc_kernels/README.md) (run 2026-11-17) | The two-shot example kernel matches NCCL's library kernel at 2 GiB (0.99–1.02) with 4 times its CTA count (same data in flight). One-shot sends N bytes per payload byte, so it falls from 168.7 (2 GPUs) to 45.4 GB/s (8 GPUs), at the same per-GPU sending rate as Ring. Multicast works at 2 GPUs through the device API |

## Plans (as proposed)

| Plan | Short name | Question | **Controlled variable** | Fixed | Runs (about) | Needs |
|---|---|---|---|---|---|---|
| [01](selection_calibration/01_selection_calibration.md) | Selection check and CTA calibration | Does each arm really run at each GPU count, and how many CTAs does each arm need before more CTAs stop helping? | **CTA count**, for each arm × GPU count | Payload 2 GiB per GPU | 15 arm × GPU-count cells × about 5 CTA counts × 3 rounds = 225 | Must run first |
| [02](gpu_count_scaling/02_gpu_count_scaling.md) | GPU-count scaling at a large payload | How does the AllReduce rate of each arm change from 2 to 8 GPUs? | **GPU count: 2, 4, 6, 8** | Payload 2 GiB per GPU; CTA count from plan 01, in two settings | 15 cells × 2 CTA settings × 5 rounds = 150, plus the GPU-choice control | Plan 01 |
| [03](payload_size/03_payload_size.md) | Payload size at each GPU count | From which payload size does each arm reach its large-payload rate, and does the order of the arms change with size? | **Payload per GPU: 32 MiB, 128 MiB, 512 MiB, 2 GiB, 8 GiB** | GPU count, one of 2, 4, 6, 8 per comparison; CTA count chosen by NCCL | 15 cells × 5 rounds = 75 (one run steps through all sizes) | Plan 01, first step only |
| [04](link_bytes/04_link_bytes.md) | Link bytes per payload byte | How many bytes does each GPU send and receive on NVLink for one payload byte, and is any arm at the link limit at any GPU count? | **GPU count**, for each arm | Payload 2 GiB per GPU, many calls per run | 15 cells × 3 rounds = 45 | Plan 02 settings |
| [05](registration_effect/05_registration_effect.md) | Classic NVLS with and without registration, against the symmetric kernel | Of the gap between classic NVLS and symmetric multicast, how much comes from the copies and few issuing warps, and how much from per-step flags? | **Arm: NVLS unregistered, NVLS registered (new), symmetric multicast** | 4 and 8 GPUs; payload 32 MiB to 2 GiB per GPU | 30 payload sweeps + 72 CTA-sweep runs + 12 link-byte runs | Added 2026-11-17, not run; needs a GPU allocation |
| [06](custom_mc_kernels/06_custom_mc_kernels.md) | NCCL's example multicast AllReduce kernels | How fast are the one-shot and two-shot example kernels of `contrib/custom_algos/allreduce` against NCCL's library kernel at large payloads, and what limits each? | **Kernel: one-shot, two-shot end barrier, two-shot poison, NCCL library** | 2–8 GPUs; payload 32 MiB to 2 GiB per GPU; own harness (MPI, one process per GPU) | about 150 | Added 2026-11-17; ran 2026-11-17, 196 runs |

Recommended order: 01 → 02 → 04 → 03. Plan 05 was added on 2026-11-17, after plans 01–04 ran; its registration check (E5.0) must pass before its comparison runs. Plan 06 was added the same day and is independent of plan 05. Plans 01 and 02 answer the main question. Plan 04 gives the observation that explains the curves in plan 02. Plan 03 shows how far the answer holds for smaller payloads.

### Plan 01: selection check and CTA calibration

- **Why first.** NCCL chooses the CTA count itself, and the choice differs by arm and may differ by GPU count. In the 4-GPU work NCCL chose 4 CTAs × 512 threads/CTA where the source report used 32 ([report](../../../h100x4/nvls/observation/multimem_load_store_mix/README.md)). Without this plan, a difference between arms could be a difference in CTA count.
- **Step A, selection check.** For each of the 15 cells, run once and read from the NCCL log: algorithm or kernel name, protocol, CTAs × threads/CTA. Confirm 0 wrong values. Open point: whether `NCCL_ALGO=NVLS` runs with 2 GPUs.
- **Step B, calibration.** For each cell, raise the CTA count (planned: 2, 4, 8, 16, 32, plus NCCL's own choice) and record the rate. Stop rule, fixed before running: less than 1% gain over two successive increases. A curve that still rises at the last point is an incomplete calibration, not a result.
- **Output.** For each cell, two settings for the later plans: "CTA count chosen by NCCL" and "smallest CTA count on the flat part".

### Plan 02: GPU-count scaling at a large payload

- **Main sub-experiment.** For each arm, measure at 2, 4, 6 and 8 GPUs, with both CTA settings from plan 01. Arms are rotated inside each round, at least 5 rounds.
- **Control: which GPUs.** Runs with 6 or 8 GPUs must use both CPU sockets; runs with 2 or 4 need not. So a change from 4 to 6 GPUs could be a socket effect. The control repeats 2 and 4 GPUs on other GPU sets (for example 0–3, 4–7, and 0, 1, 4, 5) and compares.
- **Expected shapes (hypotheses, from counting bytes per GPU only).** With S the payload per GPU and N the GPU count:

| Arm | Bytes each GPU sends, per call | algBW limit from the link alone, relative to N = 2 (N = 4 for multimem) |
|---|---|---|
| Ring | 2(N − 1)/N × S | 1.00, 0.67, 0.60, 0.57: falls with N |
| Symmetric plain load/store | 2(N − 1)/N × S, if the kernel does what its name says | Same as Ring, but with no multi-step ring |
| NVLS and Symmetric multicast | (1 + 1/N) × S, the model from the 4-GPU report | 1.00, 1.07, 1.11 for N = 4, 6, 8: rises a little with N |

- **Known limit of these hypotheses.** At 4 GPUs the measured multimem rate was 250.7 GB/s (payload bytes per GPU ÷ time), which by the model is 313 GB/s on the link, below the 478 GB/s link rating. So the link is not the only limit at 4 GPUs, and the real curves may not follow this table.

### Plan 03: payload size at each GPU count

- LLM work produces several payload sizes. Arithmetic examples, not measurements: PyTorch DDP sends gradients in buckets of 25 MiB by default; a tensor-parallel layer output of 8 sequences × 2048 tokens × 4096 values × 2 bytes is 128 MiB; a 1-billion-parameter gradient in 32-bit floats is about 4 GB.
- One nccl-tests run steps through all five sizes. Each comparison holds the GPU count fixed.
- Result to report: rate against size for each arm, one panel per GPU count, same y-axis; and the size where each arm reaches 95% of its 8 GiB rate.

### Plan 04: link bytes per payload byte

- Read the NVLink byte counters of every GPU before and after a run with many calls (`nvidia-smi nvlink -gt d`, read-only). Divide by payload bytes moved. Compare with the byte counts in the plan 02 table.
- This checks the kernel descriptions instead of trusting them, and it turns each measured rate into "share of the link rating in use". It is the first observation beyond throughput for this series.
- Limits: the counters cover the whole run, including warm-up calls, and do not show the path inside the switch.

## Decisions (2026-11-16)

| # | Decision | Choice |
|---|---|---|
| 1 | Plans | All four kept, as proposed |
| 2 | NVLS arms | Both kept |
| 3 | Main payload for plans 01, 02, 04 | 2 GiB per GPU |
| 4 | GPU allocation | All 8 GPUs for 8 hours from 06:26 UTC; 8-GPU runs first, then the GPUs may go to others |
| 5 | Convention text | Rule for `<gpu_type>-scale/` added to `AGENTS.md` and `experiments/CLAUDE.md` |
| 6 | busBW | Arm-specific factor, given by the user: (1 + 1/N) for the multimem arms (metric S6) |
| 7 | Plan 06 GPU allocation (2026-11-17) | All 8 GPUs for 3 hours from 23:12 UTC; GPU work ended 23:57 |

**Deviation from the run order in the common setup.** The common setup asks for GPU counts in blocks, repeated in reverse order. Because 8-GPU runs come first, all 8-GPU runs of every plan ran before any other GPU count (order 8, 6, 4, 2). The reverse repeat was not run: the same cell measured in different sessions of the day differs by at most 0.5%.

## Rules for this series

The [experiment conventions](../../../CLAUDE.md) apply. These rules are added because the series compares arms instead of a healthy and a perturbed run.

1. **A scaling curve joins only matched points.** Same arm, same payload, same CTA setting, same binary. Different CTA settings are different curves.
2. **Every label carries the full setting:** arm, GPU count and GPU list, payload per GPU, CTAs × threads/CTA.
3. **A run counts only if its log names the intended algorithm or kernel** and it reports 0 wrong values.
4. **algBW means payload bytes per GPU ÷ time of one call** (metric S1). busBW uses the factor of the arm (metric S6): 2(N − 1)/N for Ring and Symmetric plain load/store, (1 + 1/N) for NVLS and Symmetric multicast. The `busbw` that nccl-tests prints uses the Ring factor for every arm, so it is not used.
5. **Byte-count models are hypotheses** until plan 04 checks them at that GPU count.
