# Repository guidance for research work

## Research context and scope

This NCCL checkout also hosts early research on the observability of memory-semantic scale-up communication, using the NCCL device API for controlled experiments. Preserve upstream NCCL behavior and unrelated local changes when working on research files.

Read these sources before designing or interpreting experiments:

- [Research framework](notes/large_scale_ml_diagnosis_framework.html): observation plane, degradation detection, root-cause analysis, and optional mitigation are separate functions. Preserve the distinction between direct evidence, derived quantities, and causal inference.
- [Problem matrix](notes/scale_up_communication_problem_matrix.md): implementation × problem family; entries are candidate problems, not validated mechanisms or exhaustive coverage.
- [Experiment conventions](#experiment-conventions), included below: directory layout, planning, harness readability, run protocol, reporting, and analysis rules apply to work under `experiments/`. The matching experiment-local copy is [experiments/CLAUDE.md](experiments/CLAUDE.md).
- [Thread-store observation plans](experiments/h100x2/thread_store/observation/PLANS.md): current series scope and shared observation-design rules. Read the relevant individual plan before editing it.

The current phase discovers which observations distinguish perturbed communication from its matched no-interference reference and, where possible, distinguish competing explanations. Do not turn exploratory expectations into established results, promise a complete detector/RCA system, or assume an injection always produces degradation.

## Plain language

Write for a reader whose first language is not English. The reader should spend all attention on what the experiment does and means, not on decoding words. This applies to conversation with the user, plans, reports, figure titles, and code comments.

- Use short sentences and common words. Say "how many bytes per second the kernel reads" rather than "achieved read throughput", unless the short form is defined once and then reused.
- Use a technical or rare word only when no simpler word says the same thing. Define it the first time it appears.
- Use one name for one thing. Do not switch between synonyms for variety.
- Prefer a table or a numbered list to a long paragraph when there are several settings, steps, or results.
- State numbers with their units and say what they are compared with.

## Designing and recording experiments

- Present experiment configuration knobs in tables, covering both constants and varied settings. Boldly identify the controlled independent variable, its values, and what stays fixed within each comparison; distinguish measured responses from controlled knobs.
- Configuration labels in setup/result tables and figures must include all coupled launch dimensions, with workload roles explicit (for example, communication: 4 CTAs × 256 threads/CTA). Use separate columns or complete tuples; never label a configuration by only the swept value. Keep the sweep identity separate when the same full configuration is repeated in different sweeps.
- Use zero-based y-axes for throughput, duration, and achieved-rate figures, with rounded headroom and shared scales across comparable panels. Do not autoscale to tiny intervals that exaggerate changes; report precise small differences in tables.
- Keep communication semantics, endpoint roles, completion boundaries, perturbation knobs, and matched controls explicit. Spatial separation of CTA roles does not imply independence of GPU-wide resources.
- Each experiment needs an observation subsection beyond throughput: what is measured, by which instrument, on which endpoint and time window, at what granularity/cost, and which alternative explanation it could distinguish. State counter unavailability and interpretation limits.
- Lead communication-experiment results and interpretation with communication throughput/duration against a valid reference. Co-runner performance is a secondary observation; a copy-disabled control cannot supply a healthy-copy throughput baseline.
- Treat injection configuration as evaluation context, not as an observable available to a future diagnosis system. Record actual achieved pressure and overlap independently.
- Distinguish issue-loop duration, ordering/completion-protocol time, and destination observation. Do not describe a timer around posted stores as individual remote-store latency.
- When contention experiments require a saturated communication reference, establish where additional issuing threads stop improving throughput before running the contention grid. A planned sweep ending at an increasing curve is incomplete calibration. Extend avoidable harness limits or investigate the remaining constraint; do not treat finishing the grid as satisfying this prerequisite.
- Compare interleaved matched controls and perturbations; preserve negative results and unresolved ambiguities. Profile separately from primary timing runs and verify that profiling preserves the interference being investigated.
- Use semantic metric names. The implemented measurement subset and stable metric registry are in `experiments/h100x2/thread_store/observation/00_common_setup.md`. Reconcile older exploratory plans with that registry before implementation; do not claim candidate profilers or counters have been validated merely because they are listed.
- Plans contain intended configurations and hypotheses; measured outcomes belong in experiment READMEs and dated results. Follow the reporting, figure, and analysis conventions below when results exist.

## Experiment utilities

### Memory co-runner with a target read rate

Use this when an experiment needs HBM read traffic at a chosen level next to other work in a fused kernel.

| Item | Detail |
|---|---|
| Code | Device function in [hbm_read_pressure.cuh](experiments/h100x1/memory_co_runner/common/include/hbm_read_pressure.cuh); how to call it from a fused kernel is in the [harness README](experiments/h100x1/memory_co_runner/common/README.md) |
| How it reads | Thread 0 of each pressure CTA starts Hopper bulk copies (TMA) of fixed-size chunks from a 1 GiB buffer into shared memory; thread 32 watches the stop flag; other threads are idle. Not the copy engine. |
| The knob | **Target read rate**: total bytes per second for all pressure CTAs, set as a percent of HBM peak (3352 GB/s on these H100s). CTA count and bytes in flight stay fixed while the target changes. |
| Calibrated setting | **24 pressure CTAs × 3 chunks of 64 KiB in flight per CTA** (192 KiB shared memory per CTA). Alone on GPU 6, achieved rate = target within ±0.02 percentage points from 5% to 90% of peak; the ceiling with no limit is 91.3%. Report: [hbm_read_rate](experiments/h100x1/memory_co_runner/calibration/hbm_read_rate/README.md). |
| Alternative | 64 CTAs × 1 chunk of 96 KiB fits a 114 KiB shared-memory reservation; ceiling 93.6% |
| Near-peak setting | **48 pressure CTAs × 3 × 64 KiB with a 95% target** reached about 94% of peak alone in the fused thread-store kernel; 24 CTAs stop at about 91% |

Rules when using it:

- Record both the target rate and the achieved rate in every run. The target is injection context; the achieved rate is the measured pressure. Achieved below target means the memory system did not keep up.
- The rate counts bytes the program asked for, not hardware-counted HBM bytes, unless a counter pass confirms them.
- Every CTA of a fused launch gets the same dynamic shared memory. Using 192 KiB per pressure CTA means every CTA in that launch, including communication CTAs and the no-pressure reference, uses the same reservation. This leaves less L1 cache, so measure communication references again under that reservation; do not reuse references from a smaller one.
- Pressure CTAs take SMs: communication CTAs + pressure CTAs + other roles must fit in the SM count, one CTA per SM.
- Calibrate again before relying on numbers from another GPU model, another clock state, or a changed chunk setting.

## Execution and change discipline

- Documentation work does not authorize GPU experiments. Before GPU work, obtain the user's GPU allocation unless already provided, inspect those devices for other tenants, and do not interfere with others' processes. Do not change clocks or device settings without authorization.
- Follow the shared harness layout below when implementing it; do not assume planned harnesses, scripts, profilers, or permissions already exist. Keep raw experiment records, snapshots, telemetry, logs, large profiler artifacts, and binaries out of version control using `.gitignore`; retain them locally. Keep configurations, analysis scripts, compact derived summaries, figures, and reports reviewable in Git.
- Preserve existing user edits. Do not edit archived copies outside the repository unless requested. Never commit; leave committing to the user. Do not stage unrelated files.
- For documentation-only edits, check references, consistency, and the diff; GPU execution is unnecessary. For code changes, use the applicable build and meaningful verification, including the experiment conventions for behavior-preserving harness refactors.

## Experiment conventions

The following guidance applies to work under `experiments/`. It includes the full conventions from [experiments/CLAUDE.md](experiments/CLAUDE.md), including the reporting and calibration lessons from this session. Keep the two copies consistent; repository-wide context and change safeguards above also apply.

### 1. Directory hierarchy

```
experiments/<gpu_type>x<num_gpus>/<data_path>/<expr_series>/<expr_name>/
```

| Level | Meaning | Examples |
|---|---|---|
| `<gpu_type>x<num_gpus>` | GPU model and the **number of GPUs the experiments in that tree use** (not the size of the machine). A series whose sub-experiments need different GPU counts is split across trees (`h100x2` for the 2-GPU sub-experiments, `h100x3` for 3-GPU ones), each with its own plans copy or a pointer to the canonical plan. The machine itself is recorded in `00_common_setup.md`. | `h100x2`, `h100x4`, `b200x8` |
| `<data_path>` | The row of the problem matrix in [notes/scale_up_communication_problem_matrix.md](notes/scale_up_communication_problem_matrix.md): the concrete communication implementation under test. | `thread_store`, `thread_load`, `tma`, `copy_engine`, `nvls` |
| `<expr_series>` | What the series is for. `observation` = which metrics expose which injected problems; later series may be `detection`, `rca`. | `observation` |
| `<expr_name>` | One experiment (one matrix column, or one scenario). Named after the problem family, not after the metric. | `shared_resources`, `work_distribution`, `request_generation`, `memory_access`, `synchronization` |

Fixed files at each level:

- `<data_path>/common/`: the harness shared by every series on that data path (kernels, co-runners, scripts, Makefile). Experiments never copy the harness; they call it with a config. When the same data path exists under several `<gpu_type>x<num_gpus>` trees, one tree holds the harness and the others link to it (`h100x3/thread_store/common -> ../../h100x2/thread_store/common`), so there is one build.
- When a series is split across GPU-count trees, the full plan set and the main report live in the smallest tree (here `h100x2`); the other trees hold only their own configs, results and a short report that links back, and the main report carries a cross-tree summary table.
- `<expr_series>/PLANS.md`: index of the series and the design rules shared by all its experiments. `<expr_series>/00_common_setup.md`: baseline configuration, metric tiers, run protocol, privileged-access table. `<expr_series>/<expr_name>/NN_<plan_name>.md`: the plan for one experiment, stored beside its report, configs, and results. Keep the existing numbered plan filename when moving it; the directory names describe the experiment (for example, `shared_resources/04_shared_resource_contention.md`). Shared `PLANS.md` and `00_common_setup.md` remain at series level. Create an experiment directory for each individual plan, including cross-cutting analysis plans; update incoming links, links inside moved plans, and report-generator templates whenever paths change.
- `<expr_name>/README.md`: status and log of this experiment (what was run, when, on which GPUs, deviations from the plan, results summary). `<expr_name>/configs/`: one file per run configuration, named `<sub_experiment>_<level>_<variant>.cfg`. `<expr_name>/results/<date>/`: raw outputs (JSONL, snapshots, profiler files) and derived tables. Keep raw run records, snapshots, telemetry, logs, and large profiler files out of git (add them to `.gitignore` in `results/`); retain the local files. Version configurations, analysis scripts, compact derived summaries, figures, and reports. State that raw measurements require the local results directory or a rerun.

### 2. Writing an experiment plan

A plan is read by someone who will run it months later without the author. Structure beats prose. Use plain language that explains what work runs, what changes, what stays fixed, and what is measured. Prefer “increase the number of issuing threads” to unexplained terms such as “sweep communication concurrency”; define necessary technical terms where they appear.

1. **Open with the matrix cell and one common question.** State which row × column of the problem matrix the experiment covers and the single question every sub-experiment answers.
2. **One sub-experiment per injected problem.** Number them `E<plan>.<n>`. Use these base headings in this order, with series-required observation subsections added where specified: **Motivation**, **Configuration**, **Procedure**, **Privileged access**, **Metrics**.
3. **Motivation** says what the injection models in a real workload and what a naive metric (usually wall time) would miss.
4. **Configuration** presents all constant and varied knobs in a table and explicitly emphasizes the controlled independent variable (in bold), its values, and what stays fixed within the comparison. Distinguish controlled knobs from measured responses. Configuration labels in setup/result tables and figures must include all coupled launch dimensions, with workload roles explicit (for example, communication: 4 CTAs × 256 threads/CTA). Use separate columns or complete tuples; never label a configuration by only the swept value. Keep the sweep identity separate when the same full configuration is repeated in different sweeps. It names the knob, its healthy value and at least three intended pressure levels, whose achieved strength must be calibrated rather than assumed from the knob values in a table. Variants get a letter suffix (`E4.4b`). Everything else is inherited from `00_common_setup.md`; do not restate it.
5. **Procedure** is a numbered list: calibration, interleaved healthy/perturbed runs, profiler passes, analysis step. Default is the interleaved `H, P, H, P, ...` pattern with at least 5 pairs per level.
6. **Privileged access** lists what the sub-experiment needs beyond an ordinary user (counter permission, root for clock locking, free GPUs, extra ranks) and the fallback if it is not available. Say "None" explicitly.
7. **Metrics** is a table with three columns: metric id from the common setup, expected signature, and which other sub-experiment it discriminates from (or "cannot separate" for contention experiments). Expected signatures are hypotheses; never write them as results.
8. **Metric ids are stable.** `A1`…`A11` (Tier A), `B1`…`B3` (Tier B), `C1`…`C8` (Tier C) are defined once in `00_common_setup.md`. Add new ones there, never inline in a plan.
9. **Every run records the implemented common metric set**, not only the metric the experiment is about. Explicitly record unavailable metrics and probe-mode omissions; listing a candidate counter does not establish that it was collected. Cross-experiment discriminability depends on it.
10. **Record what cannot be controlled**: clocks, co-tenant load, NVLink counters before and after. The machine is shared.
11. **Keep direction and roles explicit.** For a push, name the issuing rank `I`, the destination `D`, third parties `T`. Say which endpoint a metric is read on.
12. **Figures in reports.** Every result section of an experiment report has at least one figure generated by a script kept beside the harness (`make_figures.py`), stored under `results/<date>/figures/` and embedded in the README. Use these chart rules: one axis per chart, fixed categorical colours (healthy = blue, perturbed = orange), thin marks, a legend for two or more series, and look at the rendered image before publishing it. Throughput, duration, and achieved-rate y-axes start at zero with rounded headroom; comparable panels share scales. Never autoscale to a tiny interval that exaggerates small changes; retain precise differences in result tables.
13. **Names, not labels, in reports.** Write every metric's meaning in words ("fence drain time", "achieved throughput") and every sub-experiment by what it injects ("a compute-bound co-runner on the issuing GPU"). An id such as `(A5b)` or `(plan E4.1)` may follow the full name once, in parentheses, never stand alone, and never appears in a heading, a finding title or a figure title. Do not invent finding labels (`H1`, `E4.1-b`); refer to a finding by its section or by its content. A report opens with a short table mapping the words it uses to the ids in `00_common_setup.md`.
14. **A report is self-contained.** It opens with the question and a one-paragraph answer. Every result section is a complete chain under fixed sub-headings: **Motivation** (what the injection models and what the plan expected), **Setup** (ranks, GPUs, knob and its levels, config names), **Method** (what was actually run, calibration, deviations from the plan), **Results** (figure and table), **Interpretation** (findings in bold-leaded bullets). Present the primary communication-throughput/duration analysis before secondary co-runner performance in both Results and Interpretation. Do not use a copy-disabled arm as a healthy-copy throughput reference or invent pairs across separate cohorts. A reader must not need the plan file to understand the report, and the plan is linked for the expectations it states.
15. **Markdown style:** one paragraph or list item per line, no hard wrapping; tables for anything with more than two attributes; planned configuration values belong in plan tables; measured results and result screenshots belong in `<expr_name>/README.md` and `results/`, not plan files.
16. **Plain language.** Write for a reader whose first language is not English: short sentences, common words, one name per thing, and a technical word only when no simpler word exists (define it at first use). See [Plain language](#plain-language) above.

### 3. Writing harness and analysis code

Code under `experiments/` is read by people who did not write it and who are trying to understand what a number in a report means. Readability is part of the deliverable.

1. **Every name says what it holds.** No single-letter or abbreviated variables (`a`, `bp`, `tb1`, `sCo`, `nth`); write `assignment`, `backPressureCycles`, `barrierEndNs`, `coRunnerStream`, `threadsPerCta`. Struct fields follow the same rule (`sourceRank`, `destRank`, `isPull`, not `src`, `dst`, `pull`). A time variable carries its unit (`...Ns`, `...Ms`, `...Cycles`).
2. **One statement per line.** Never join declarations or statements with `;` on one line, and never pack an `if` body or a loop body after the condition.
3. **One stage per function.** A kernel or a thread body longer than a screen is split into named stages (`buildAssignments`, `preloadKernels`, `calibrateGlobalTimer`, `launchCoRunner`, `stopCoRunner`, `writeRunJson`). Argument parsing, validation and output writing are separate functions.
4. **Comment the why, not the what.** Each stage opens with a comment saying what it does and, where there was a bug or a pitfall, why it is done this way (non-blocking streams, kernel preloading, the single polling thread in co-runners). Record-struct fields that implement a metric carry the metric id in a trailing comment.
5. **Output keys are an interface.** JSON key names and the `schema` arrays are read by the analysis scripts and by every recorded result; renaming a variable never renames a key. Say so in a comment next to the writer.
6. **Verify a refactor against recorded data.** After any change that is meant to keep behaviour, rebuild and run one guarded healthy/perturbed pair on the authorised GPUs, feed the output through the unchanged analysis scripts, and compare with the baseline numbers in the report before staging.
7. **Binaries never enter the index.** `common/bin/` is ignored; if a build product was staged before the ignore rule existed, `git rm --cached` it.

### 4. Running an experiment

- Before any GPU work: use the user’s existing allocation if one is provided; otherwise obtain it. Check `nvidia-smi` for other tenants on the allocated devices and never interfere with their processes. Do not change clocks or power settings without authorization. Documentation-only work does not require GPU execution.
- Build the harness from `<data_path>/common/` with `make`; do not build inside an experiment directory.
- Each run is one invocation of the harness with one config file plus the snapshot script around it. Never hand-edit results.
- Healthy and perturbed runs are interleaved in one sequential runner session (`run_pair.sh` or its experiment-specific equivalent) so that their operating conditions are as similar as practical. Hold per-device locks before checking occupancy. Wait for actual process completion before starting another GPU job; a returned session ID is not completion. Preserve overlapping or otherwise invalid attempts locally with exclusion reasons, and reacquire affected pairs.
- Write a short entry in `<expr_name>/README.md` after every session: date, GPUs, configs run, anything that deviated from the plan, where the results are.
- Tier B/C profiler passes (nsys, ncu) are separate runs after the Tier A pass; they change timing.
- Never commit; stage and leave the commit to the user.

### 5. Analysis

- **Calibration belongs to its parent experiment.** Keep calibration configuration tables, selection criteria, measured results, figures, and the resulting reference configuration in the main `<expr_name>/README.md`, alongside the interference comparisons that use them. Dated reports and result directories preserve run evidence; links to them do not replace a self-contained main report. Update the main answer and conclusions when follow-up measurements change the reference or findings.
- **Communication performance comes first.** For experiments on communication degradation, open the Results and Interpretation with completed communication throughput (and duration where useful), its matched reference, and the size of the change. Destination memory-operation or other co-runner performance follows as supporting evidence. A slowdown of the destination workload alone does not establish communication degradation. For example, in “Destination workload progress with and without incoming stores,” compare copy-only against copy-with-destination-workload first; then compare destination-workload-only against destination-workload-with-copy. These comparisons answer different questions and need their own controls.
- **Make every comparison identifiable.** Setup tables list both variable and constant knobs and emphasize the **controlled independent variable**. Result tables and figure labels retain the complete relevant configuration: communication CTAs × threads/CTA, traversal, and co-runner CTAs × threads/CTA with endpoint/role. “4 communication CTAs” is insufficient; use “communication: 4 CTAs × 256 threads/CTA.” Shared constants may appear in an explicit table heading or figure caption, but readers must not need another section to recover them.
- **Keep visual and numerical effect sizes honest.** Throughput, duration, and achieved-rate plots start at zero, use rounded headroom, and share scales across comparable panels. Do not magnify negligible differences through a narrow y-axis range. Give absolute values, percentage changes, and reference variability in tables; when a tiny reference IQR produces a large standardized effect, explain that denominator and assess the practical magnitude separately. Preserve a finding of no substantial communication regression even when a secondary metric changes strongly.
- When contention experiments require a saturated communication reference, establish where additional issuing threads stop improving throughput before running the contention grid. A planned sweep ending at an increasing curve is incomplete calibration. Extend avoidable harness limits or investigate the remaining constraint; do not treat finishing the grid as satisfying this prerequisite.
- Use the shared collector to turn raw records into tables. Keep reusable analysis and figure generators under `<data_path>/common/scripts/`; one-off analysis notebooks may live beside the dated data. Generate figures from recorded measurements, never hand-enter or alter observations to match a narrative.
- Report effect sizes against the interleaved healthy runs (`(median_P − median_H) / IQR_H`), not raw percentages alone. State the aggregation level and pairing; normally summarize invocation medians and retain transfer-level distributions for tails. Do not treat nested transfers as independent repetitions. If reference IQR is zero, report it and use absolute/percentage differences instead of dividing by zero. Distinguish a median of paired changes from a ratio of pooled medians.
- A finding that contradicts a plan's expected signature is a result, not an error. Record it in the experiment README and, if it changes the design, update the plan with a dated note rather than silently rewriting the expectation.

### 6. Extending reports and interpreting calibration

- **Extend scaling figures; do not make the reader choose between old and new data.** When a CTA-count or threads-per-CTA sweep is extended, put the earlier and new measured configurations together in an overview figure in the main report. Show CTA scaling at fixed threads/CTA and thread scaling at fixed CTA count, using separate panels where helpful. Include the high-concurrency confirmation points explicitly; adding a separate table or moving an older figure does not satisfy a request to extend the scaling plot. Keep earlier detailed figures and data unless their removal is requested, or explain an explicit replacement that preserves their information.
- **Never join incompatible measurements into a single scaling curve.** Distinguish GPU pair, binary, payload, batch/request depth, traversal, and fixed launch dimension in legends/captions. Connect only points from a matched sweep; show different configurations as separate series even when they share an axis. Do not interpolate unmeasured points, hide downturns, or imply that all candidate combinations were run. Mark exploratory single-invocation points separately from repeated confirmations and define error bars.
- **Define what a plateau establishes.** Show the full measured scaling progression, the complete selected configuration, adjacent changes, repeat counts, and variation. State the operational stopping tolerance before interpreting it; for example, less than 1% improvement over two successive feasible request-supply increases with a small spread across the confirmed points. A locally flat program-throughput curve does not prove HBM, L2, or NVLink saturation. A real hardware limit, an avoidable harness limit, and an unexplained downturn are different findings. If the curve still rises, continue calibration or state why it remains incomplete before drawing contention conclusions.
- **Calibrate co-runner bandwidth progressively before choosing pressure settings.** Sweep CTA count and threads per CTA, include intermediate counts such as 64 where feasible, and measure actual work per elapsed time. A sparse 8/32/100 list is not an established mild/moderate/strong ladder. If the strongest point is still improving, investigate additional independent requests, loop overhead, compiler output, and occupancy before calling it maximal. Preserve flat and nonmonotonic settings. Sweep the calibrated configurations in the contention experiment and report both standalone and concurrent achieved rates; equal configuration does not guarantee equal pressure under interference.
- **Account for shared resource budgets and coupled knobs.** With one CTA per SM, source communication C and source co-runner M must satisfy C + M ≤ the available SM count. A communication reference using nearly every source SM cannot also host a large spatially separated source co-runner. Calibrate feasible communication budgets, hold communication geometry fixed within each pressure sweep, and use a matched no-pressure control for every group. If one harness knob changes both communication and co-runner threads, say so or implement independent controls before claiming a one-variable comparison. Record active role threads and physical block dimensions when they differ.
- **Do not silently transfer a reference between experiments.** Preserve the full tuple: endpoints, communication CTAs × threads/CTA, payload, batch, traversal, binary, completion protocol, and reservations. Recalibrate or remeasure appropriate controls after changes. A destination-pressure result with a new reference does not update an older source-pressure experiment. Clearly label what has been executed and what remains only a revised plan.
- **Keep progress evidence separate from mechanism claims.** Completed throughput is the primary communication result; source phase timing, destination observation, co-runner work and placement support interpretation. Phase medians are not additive, a posted-store timer is not remote-store latency, active-interval coverage is not continuous request activity, and logical read GB/s is not physical HBM traffic. A large co-runner slowdown with little communication change is an important result, not proof of either no interaction or a particular arbitration policy. Name competing explanations and the measurements needed to distinguish them.
- **Validate the integrated report after every extension.** Update its question/answer, scope, configuration tables, figures, conclusions, and session log together. Remove stale claims that calibration ended at an old software limit or that newly used devices remain unused. Check figure paths and links, compare displayed numbers with saved summaries, inspect rendered figures, and verify that requested earlier and new configurations remain visible. Preserve raw evidence, rejected-run records, and negative results; do not rerun GPUs just to reorganize documentation.
