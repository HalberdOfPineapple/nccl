# 02 · Request generation and concurrency

Matrix cell: thread-issued peer stores × too few issuing warps; insufficient independent accesses; uneven issuing activity.

Question: the fabric is healthy and the work is balanced, but the issuer does not keep enough stores in flight to saturate it. Which observation shows that the limit is on the issuing side?

## Too few issuing CTAs

Fewer CTAs means fewer SMs issuing stores. Throughput should rise with the CTA count until the fabric or the destination saturates; the knee shows where the bottleneck moves from issuing to the path.

## Too few issuing warps

The same SMs issue, but with fewer warps per SM. This separates "not enough SMs" from "not enough warps per SM", and shows how close one warp per SM already comes to the per-SM store capacity.

## Frequent fences

The push-side form of insufficient independent accesses. A store does not stall its warp, but a fence waits until all of the warp's outstanding stores are acknowledged, so a kernel that fences often serialises its stores on the round trip. Variant: a release store per element instead of a separate fence, as a naive per-element signalling scheme does.

## Gaps in issuing

Warps go quiet between bursts of stores, as in a fused kernel that alternates compute and communication. Average throughput falls while every instantaneous measure of issuing looks healthy. Variant: the gap is filled with compute instead of a sleep.

## Uneven issuing across warps

Bytes are balanced per CTA, but inside each CTA a few warps do most of the issuing. Per-CTA observations cannot see this; the question is what granularity can.
