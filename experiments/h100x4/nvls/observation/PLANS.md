# Observation series: observing NVLS collectives

## Goal

Find which observations explain performance changes in NVLS collectives. The series starts by repeating measurements that other people reported, on our own machine and with our own measurement rules. A reproduction answers two things: does the same effect appear here, and does it mean what the report says it means.

This series covers the **NVLS-based collective implementation** row of the [problem matrix](../../../../notes/scale_up_communication_problem_matrix.md). Plan 01 has run, with a reduced scope; see its [report](multimem_load_store_mix/README.md). The harness is in `../common/`.

## Plans

| Plan | Problem family | Source being reproduced | What it does |
|---|---|---|---|
| [01_multimem_load_store_mix.md](multimem_load_store_mix/01_multimem_load_store_mix.md) | Shared-resource contention | [Shouxu's scale-up switch report](../../../../notes/references/scaleup-switch%20(shouxu%20expr%20report).pdf), section 3.4 | Does a GPU that issues `st_multicast` and `ld_reduce` at the same time lose throughput, while plain loads and stores do not? |

## Rules for this series

The [experiment conventions](../../../CLAUDE.md) apply in full. These rules are added because this series repeats someone else's numbers.

1. **Write the reproduction test before running.** State which number or ratio counts as "reproduced", "partly reproduced" and "not reproduced". Do not change these limits after seeing the data. If a limit turns out to be wrong, keep it, report the result under it, and add a dated note.
2. **Compare ratios first, absolute numbers second.** Our hardware may differ from the source's. A different absolute number is not a failure.
3. **Reproducing a number does not confirm its explanation.** The source's explanation is a hypothesis. Keep the two apart in every report.
4. **List what the source does not state.** Each missing setting becomes a recorded choice in our plan and a question for the author. Do not present our choice as the author's setting.
