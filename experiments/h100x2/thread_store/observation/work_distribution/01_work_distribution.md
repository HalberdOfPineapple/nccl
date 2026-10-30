# 01 · Work distribution and timing

Matrix cell: thread-issued peer stores × unequal bytes per peer or CTA; staggered data readiness.

Question: when demand is uneven across CTAs, destinations, ranks or time, which observation shows where the imbalance is, rather than only a slower kernel?

## Unequal bytes per CTA

One CTA gets several times its fair share of the bytes while the total stays the same. It finishes last and sets the completion time while the rest of the GPU is idle. Kernel time cannot tell this from a uniformly slower fabric; per-CTA timing should.

## Unequal bytes per destination

In a push to many peers, one destination GPU receives more than its share from every issuer, as a hot expert does in expert parallelism. The imbalance is now located at a destination, not a CTA, and the hot destination's ingress may saturate and slow every issuer. A variant keeps bytes balanced but serves the hot destination with fewer CTAs, so the limit is issue concurrency toward it rather than its ingress.

## Late participant

One rank launches late and every other rank waits for it at the start barrier. From any single rank this looks like a slow fabric; only cross-rank timing or attributing time to waiting separates "I waited" from "I was slow".

## Destination buffers released late

In a push the data is ready at the issuer by construction; what can be late is the destination buffer, still in use by the consumer. The issuer must wait for a "free" signal before overwriting, with the release either uniformly late or staggered across chunks. The observation should attribute the delay to the consumer's slow release rather than to the fabric or the issuer.
