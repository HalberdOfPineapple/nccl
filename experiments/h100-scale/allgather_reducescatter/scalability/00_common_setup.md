# Common setup for the AllGather and ReduceScatter scalability series

**Status (2026-11-18): used by plan 01, which ran on 2026-11-18. The link-byte model below was confirmed by the link counters (plan 01 report).** Machine, software, GPU sets, base command, run protocol and the metric registry are those of the [AllReduce series common setup](../../allreduce/scalability/00_common_setup.md); only the differences are listed here.

## Payload and rate for these collectives

nccl-tests defines one size per call for both collectives, the whole buffer of one GPU:

| Collective | Size printed by nccl-tests (S) | What one GPU contributes | What one GPU gets |
|---|---|---|---|
| AllGather | Output buffer of one GPU | Its S/N piece | All N pieces, S |
| ReduceScatter | Input buffer of one GPU | Its whole input, S | The sum of its S/N piece |

Rate (algBW, metric S1) = S ÷ time of one call, as nccl-tests prints it. Sizes in this series are values of S.

## How each arm is selected

| Arm | AllGather | ReduceScatter | `-R` | CTA knob | Proof in the NCCL log |
|---|---|---|---|---|---|
| Ring | `NCCL_ALGO=Ring NCCL_PROTO=Simple` | same | 0 | `NCCL_MIN_CTAS` = `NCCL_MAX_CTAS` | `AllGather: ... -> Algo RING proto SIMPLE` (ReduceScatter alike) |
| Classic NVLS | `NCCL_ALGO=NVLS NCCL_PROTO=Simple` | same | 0 | `NCCL_NVLS_NCHANNELS` | `-> Algo NVLS proto SIMPLE` |
| Symmetric multicast | `NCCL_SYM_KERNEL=allgather:AllGather_STMC` | `NCCL_SYM_KERNEL=reducescatter:ReduceScatter_LDMC` | 2 | `NCCL_SYM_CTAS` | `-> Kernel AllGather_STMC` / `ReduceScatter_LDMC`, no other kernel |
| Symmetric plain load/store | `NCCL_SYM_KERNEL=allgather:AllGather_ST` | `NCCL_SYM_KERNEL=reducescatter:ReduceScatter_LD` | 2 | `NCCL_SYM_CTAS` | `-> Kernel AllGather_ST` / `ReduceScatter_LD` |

Symmetric multicast is not available at 2 GPUs ([sym_kernels.cc:195](../../../../src/sym_kernels.cc#L195)), as for AllReduce. The TMA and rail kernels are left out (TMA kernels are for newer GPUs; rail kernels need a network).

## What one call does, and the link bytes it should need

From the code; these are hypotheses until the link-byte step of plan 01 measures them. Bytes per GPU per call, payload S:

| Arm | AllGather: sent / received | ReduceScatter: sent / received |
|---|---|---|
| Ring | (N − 1)/N × S / (N − 1)/N × S | (N − 1)/N × S / (N − 1)/N × S |
| Symmetric plain | Stores its piece to every peer: (N − 1)/N × S / (N − 1)/N × S | Loads its piece from every peer: (N − 1)/N × S / (N − 1)/N × S |
| Symmetric multicast | One `multimem.st` of its piece: S/N sent; every GPU's piece arrives, its own included: S received | `multimem.ld_reduce` of its piece: it answers every GPU's loads for its whole input, S sent; S/N received |
| Classic NVLS | As symmetric multicast on the link (the copies through NCCL's buffer stay on the GPU) | As symmetric multicast on the link |

The busiest direction carries (N − 1)/N × S for Ring and plain, and S for the multicast arms. So unlike AllReduce, where multicast needs fewer link bytes than Ring (1 + 1/N against 2(N − 1)/N), **for these two collectives multicast needs N/(N − 1) times more bytes in its busiest direction**: the GPU's own piece goes through the switch. Whether the own piece really crosses the link is checked by the link counters (for AllReduce, plan 04 found that it does).

### busBW (metric S6) for this series

busBW = algBW × bytes in the busiest direction per payload byte:

| Arm | busBW | N = 2 | N = 4 | N = 6 | N = 8 |
|---|---|---|---|---|---|
| Ring, symmetric plain | algBW × (N − 1)/N (the nccl-tests `busbw`) | 0.500 | 0.750 | 0.833 | 0.875 |
| Symmetric multicast, classic NVLS | algBW × 1 | 1 | 1 | 1 | 1 |
