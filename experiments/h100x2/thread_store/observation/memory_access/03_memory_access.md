# 03 · Memory access and locality

Matrix cell: thread-issued peer stores × poor coalescing; scattered peer accesses; changes in buffer reuse.

Question: the same useful bytes are moved, but the store pattern multiplies the number of transactions or produces partial writes at the destination. Which observation exposes "more transactions per useful byte" as opposed to "fewer bytes per second"?

## Narrow stores

Stores of 4 bytes instead of 16 need four times as many instructions and requests per byte. This is an issuing cost rather than a fabric cost, so it should show in instruction and request counts, not in link bytes.

## Strided stores

Consecutive threads write addresses far apart, so one warp's request is spread over many sectors and lines and the destination receives partial writes. This is the canonical case of extra transactions per useful byte. A pull version is a cheap comparison, since load and store coalescing over NVLink may differ.

## Misaligned stores

A base offset that is not a multiple of the store width forces narrower stores or extra sectors per request: a mild, easy-to-miss problem that should still leave a signature.

## Scattered destinations

Random small stores into the destination, as index-driven pushes do (token dispatch into per-expert slots, embedding updates), with locality ranging from fully random to random within small regions.

## Repeated writes and destination caching

Stores to peer memory land in the destination's L2, which may combine repeated writes before they reach HBM. A workload change that spreads writes over a larger footprint can look like a fabric slowdown from the issuer. The question is whether write combining happens on this path and what separates "combining lost" from "link slower".
