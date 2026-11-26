# Plan 02: GPU-count scaling at a large payload

**Question:** when the GPU count grows from 2 to 8, how does algBW and busBW of each arm change?

## E2.1 Scaling with the GPU count

### Motivation

LLM training spreads the same gradient AllReduce over more GPUs. Wall time alone mixes two effects: a fixed per-GPU link and a factor that changes with N. busBW with the arm's own factor separates them: if busBW stays the same over N, the arm uses the link equally well at every N.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **GPU count: 2, 4, 6, 8** (main GPU sets in the common setup) |
| Arms | Ring, NVLS, Symmetric multicast (4, 6, 8 only), Symmetric plain load/store |
| CTA setting | Two curves per arm: "auto" and "flat" from plan 01 |

**Note (2026-11-16, after plan 01).** Ring, NVLS and symmetric multicast run only at NCCL's choice, which is within 4% of the best calibration value at every GPU count. Plain load/store runs at NCCL's choice and at 16 and 48 CTAs fixed at every GPU count, so each curve keeps the same CTA count; its "flat" count differs by GPU count (24, 16, 8, 16).
| Payload | 2 GiB per GPU |
| Rounds | 5 per cell, arms rotated |

Expected shapes, only from counting bytes per GPU (hypotheses): if every arm were limited by the link, busBW would be the same at all N for each arm, and algBW would fall with N for Ring and plain load/store (× 1.00, 0.67, 0.60, 0.57) and rise a little for the multimem arms (× 1.00, 1.07, 1.11 from N = 4).

### Procedure

1. 8 GPUs first, then 6, 4, 2.
2. Five rounds per GPU count, all arms and both CTA settings in each round.
3. Analysis: S1, S6 and S2 per arm and CTA setting.

### Privileged access

Free GPUs. 8 GPUs means the whole machine.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| S1 | See shapes above | |
| S6 | Flat over N if link-limited | Link-limited arm from an arm limited elsewhere |
| S2 | As above | |
| B2 | Same state across N | Clock or power changes from a GPU-count effect |

## E2.2 Control: which GPUs

### Motivation

6- and 8-GPU runs use both CPU sockets; 2- and 4-GPU main sets use one. If the GPU set itself changes algBW, a change from 4 to 6 GPUs could be a set effect.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **GPU set at a fixed GPU count**: 2 GPUs: {4,5}, {0,1}, {3,4}; 4 GPUs: {4–7}, {0–3}, {2–5}; 6 GPUs: {1–6}, {0–5} |
| Fixed | Each arm with its "auto" CTA setting; 2 GiB per GPU |
| Rounds | 3 |

### Procedure

1. Run after E2.1 at the same GPU count.
2. Compare algBW between sets of the same GPU count.

### Privileged access

Free GPUs of each set.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| S1 | Within 2% across sets (hypothesis: all GPUs share the same NVSwitch path) | GPU-count effect from a GPU-set effect |

## E2.3 Follow-up: GPU set or uneven split at 6 GPUs

**Added 2026-11-16 after E2.2 showed a set effect for the symmetric kernels at 6 GPUs.**

### Motivation

At 6 GPUs, 2 GiB does not divide evenly into 6 parts, while it does at 2, 4 and 8 GPUs. The 6-GPU drop of the symmetric kernels could come from that, or from which GPUs form the set.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variables** | **GPU set: {0–5}, {1–6}, {2–7}, {0,1,2,5,6,7}**; **payload: 2 GiB or 2 GiB − 128 B (6 equal parts of whole 16-byte units)** |
| Arms | Ring, NVLS, symmetric multicast at NCCL's choice; plain load/store at 16 CTAs |
| Rounds | 3 |

### Procedure

1. All cells rotated in one session.
2. Compare the two payloads per set, then the sets per payload.

### Privileged access

Free GPUs of each set.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| S1 | If the split is the cause: the 2 GiB − 128 B payload removes the drop in every set. If the set is the cause: the payload changes nothing and sets differ | Uneven split from GPU-set effect |
