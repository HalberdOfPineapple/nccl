# Plan 04: link bytes per payload byte

**Question:** how many bytes does each GPU send and receive on NVLink for one payload byte, at each GPU count, and how close is each arm to the link rating?

## E4.1 Link byte counters around many calls

### Motivation

busBW is computed from a byte model, not measured. If an arm moves more bytes than the model, busBW understates the link load. This is the first observation beyond throughput.

### Configuration

| Knob | Value |
|---|---|
| **Controlled independent variable** | **GPU count: 2, 4, 6, 8**, for each arm |
| Run pair | A long run (200 timed calls) and a short run (20 timed calls) with the same settings, back to back |
| Fixed | 2 GiB per GPU; CTA setting "auto"; 5 warm-up, 1 check |
| Rounds | 3 |

### Procedure

1. Read `nvidia-smi nvlink -gt d` per GPU before and after each run (B3), summed over the 18 links, transmit and receive separately.
2. nccl-tests runs the timed calls twice (out-of-place, then in-place) and adds warm-up and check calls. These fixed calls are the same in the long and short run, so S4 = (bytes of long run − bytes of short run) ÷ (2 × (200 − 20) × payload). This removes setup, warm-up and check traffic from the count.
3. Compare S4 with the byte model: 2(N − 1)/N for Ring and plain load/store, (1 + 1/N) for the multimem arms.
4. S5 = algBW × S4 ÷ 478 GB/s.

### Privileged access

None. The counters are read-only.

### Metrics

| Metric | Expected | Separates |
|---|---|---|
| S4 | Matches the byte model within a few % | Model error from an arm limit |
| S5 | Below 1 for every arm | Link-limited arm from an arm limited elsewhere |
