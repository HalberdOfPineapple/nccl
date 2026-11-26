# Plan 01: selection check and CTA calibration

**Matrix cell:** none; this plan sets up the reference for the whole series. **Question:** does each arm run what we asked at each GPU count, and from how many CTAs does each arm stop getting faster?

Arms, command, metrics and run rules are in [00_common_setup.md](../00_common_setup.md).

## E1.1 Selection check

### Motivation

NCCL can fall back to another algorithm or kernel without an error. A comparison of arms is only valid if each run used the intended code.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **Arm × GPU count: the 15 cells** |
| Payload | 2 GiB per GPU |
| CTAs × threads/CTA | Chosen by NCCL (recorded) |
| Calls | 5 warm-up, 20 timed, 1 check |

### Procedure

1. One run per cell, 8 GPUs first.
2. Read algorithm or kernel, protocol, CTAs and threads/CTA from the NCCL log (S3). Check 0 wrong values (A11).
3. For NVLS at 2 GPUs, record whether NCCL runs it, refuses it, or falls back.

### Privileged access

Free GPUs for the GPU set of each cell. Nothing else.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| S3 | Ring/Simple, NVLS/Simple, or the forced kernel with no other kernel line | A wrong run from a slow run |
| A11 | 0 wrong values | |

## E1.2 CTA calibration

### Motivation

NCCL picks the CTA count from its own model. If two arms get different CTA counts, a gap in algBW could be a gap in CTA count, not in the arm. Later plans need, for each cell, the CTA count where more CTAs stop helping.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **CTA count: NCCL's choice, 2, 4, 8, 16, 24, 32, 48, 64** |
| Fixed | Arm, GPU count and GPU set within each curve; payload 2 GiB per GPU; threads/CTA chosen by NCCL |
| CTA knob | Ring: `NCCL_MIN_CTAS` = `NCCL_MAX_CTAS`; NVLS: `NCCL_NVLS_NCHANNELS`; symmetric arms: `NCCL_SYM_CTAS` |
| Rounds | 3, cells rotated between rounds |

### Procedure

1. Run every CTA count of every arm at one GPU count in one session; 8 GPUs first.
2. For each curve, record the CTA count NCCL actually used (S3), which can differ from the request.
3. Stop rule, fixed before the data: the curve is flat from the first CTA count after which two successive increases each gain less than 1% in median algBW. If the last point still gains 1% or more, the calibration is incomplete for that cell.
4. Output two settings per cell: "auto" (NCCL's choice) and "flat" (smallest CTA count on the flat part).

### Privileged access

None beyond free GPUs.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| S1 | Rises with CTA count, then flat | A CTA-count limit from a limit of the arm |
| S6 | Same shape as S1 | |
| S3 | Used CTA count equals the request, up to each knob's limit | |
| B2 | No clock-event reasons other than idle | Thermal or power limits |
