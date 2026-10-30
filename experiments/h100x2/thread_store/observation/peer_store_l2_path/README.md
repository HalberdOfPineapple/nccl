# Path of incoming peer stores through the destination's L2

**Status (2026-10-30): plan only. Nothing has been built or run, so this file has no results.**

[Plan](08_peer_store_l2_path.md) · [Common setup and metric ids](../00_common_setup.md) · [Harness](../../common/README.md)

**Question.** When a thread on the source GPU stores to the destination GPU's memory through an NCCL LSA pointer over NVLink, does the data pass through the destination GPU's L2 cache on its way to HBM?

**Answer.** Not known yet.

## Session log

| Date | GPUs | What ran | Results |
|---|---|---|---|
| 2026-10-30 | None | Plan written. Host-only queries of the installed profiler's counter lists; no GPU work | None |
