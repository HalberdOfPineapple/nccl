# 05 · Synchronization and progress

Matrix cell: thread-issued peer stores × waiting for readiness, completion notification, or buffer reuse; a slow CTA delays dependent work.

Question: nothing is slow in itself, but a dependency makes work wait. Time grows while the waiting side looks idle rather than busy. Which observation attributes the wait to the right dependency and the right party?

In a push there are two places to wait: the issuer waits for destination buffers to be free and for barriers; the consumer waits for completion flags.

## A slow CTA at a barrier

One CTA on one rank arrives late at a barrier, so only the matching CTAs on the other ranks wait. The question is whether per-CTA waiting time points at both the waiters and the slow CTA. Variant: the delay comes after the barrier, before the stores.

## A slow rank at a barrier

All CTAs of one rank arrive late, as a rank with extra local work before its stores would. Unlike a late launch, the launches are aligned and the culprit is inside the kernel.

## Completion-flag polling

The consumer polls local flags that the issuer writes after its data. This isolates the notification path: how long after the last data store the flag becomes visible, and what different ways of signalling and polling cost. Variant: the flag is written late on purpose, so the data is on time but the notification is not.

## Buffer-reuse waits

A double-buffered pipeline in which the issuer may not overwrite a buffer until the consumer has released it. If the consumer is slow, the issuer waits; if the issuer is slow, the consumer waits. Both show up as a slower kernel on different ranks, and the observation must say which side caused it.

## Completion-signal granularity

Fine-grained signalling lets the consumer start early but multiplies fences and flag stores on the issuer and polling on the consumer; coarse signalling leaves the consumer idle until the end. The question is at what granularity signalling itself becomes a cost the observation plane must count.
