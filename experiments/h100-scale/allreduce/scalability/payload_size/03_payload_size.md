# Plan 03: payload size at each GPU count

**Question:** from which payload does each arm reach its large-payload algBW, and does the order of the arms change with payload?

## E3.1 Payload sweep

### Motivation

LLM work produces many sizes: gradient buckets of about 25 MiB (PyTorch DDP default), tensor-parallel activations of about 128 MiB, whole-gradient AllReduce of several GiB. A ranking at 2 GiB may not hold at 32 MiB.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Payload per GPU: 32 MiB, 64 MiB, 128 MiB, 256 MiB, 512 MiB, 1 GiB, 2 GiB, 4 GiB, 8 GiB** (one run steps through all, factor 2) |
| Fixed per comparison | GPU count and set; arm; CTA setting "auto" (NCCL may pick a different count at each size, recorded per size) |
| Rounds | 5 |

### Procedure

1. 8 GPUs first, then 6, 4, 2.
2. For each arm and GPU count, the payload where algBW first reaches 95% of its 8 GiB value.

### Privileged access

Free GPUs.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| S1 | Rises with payload to a plateau; the multimem arms reach it later because their per-step size is fixed | |
| S3 | CTA count may change with size | A size effect from a CTA-count change |
