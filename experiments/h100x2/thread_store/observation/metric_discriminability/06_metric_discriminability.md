# 06 · Cross-cutting analysis: which observations separate which problems

This is the analysis that turns the experiments of plans 01 to 05 into the result of the series: what the observation plane needs for a push data path.

## Ideas

- **Detection.** For every injected problem and severity, which observations move clearly away from the healthy runs, and what is the mildest severity each one detects.
- **Confusion.** For every pair of problems, which observations differ between them. Two problems that no cheap, always-on observation separates are recorded together with the more expensive observation (profiler, hardware counters) that does, if any.
- **Localisation.** For problems with a known culprit (a CTA, a rank, a destination, a side of a pipeline), whether an observation points at the culprit or only at the parties that waited for it, and whether the issuer's or the consumer's side must be observed.
- **Cost and access.** For each observation, what it costs and what rights it needs, so the result says which tier of observation is enough for which problem.

## Deliverable

A short report under `notes/` with the detection, confusion and localisation tables and the cost of each observation, feeding the design of the observation plane for the next data path in the matrix.
