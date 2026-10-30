# Memory co-runner calibration series

This series measures memory-heavy GPU roles on their own, so that communication experiments can use them as known, adjustable pressure. Each role is built to fit inside a fused kernel: it runs on its own SMs, one CTA per SM, next to communication CTAs.

This tree is not a row of the [problem matrix](../../../../notes/scale_up_communication_problem_matrix.md). `memory_co_runner` takes the place of `<data_path>` because the code here is a co-runner shared by several communication paths, not a communication path. Experiments use one GPU, so the tree is `h100x1`.

| Plan | Question |
|---|---|
| [01_hbm_read_rate.md](hbm_read_rate/01_hbm_read_rate.md) | Can one role read HBM at a chosen rate, near zero to near peak, with few SMs? |

Shared rules: the [experiment conventions](../../../CLAUDE.md) apply. Rates are reported as a percent of the device's HBM peak, computed from its memory clock and bus width. A rate counted from requested bytes is not counted HBM traffic. Raw records stay local; summaries, figures and reports are versioned.
