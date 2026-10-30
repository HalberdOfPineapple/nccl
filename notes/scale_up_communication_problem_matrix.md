# Categorizing Performance Problems in Scale-Up Communication

## Purpose and scope

This working categorization organizes preliminary experiments on performance degradation in scale-up communication. Its purpose is to help select controlled perturbations and discover useful observations. It is not an exhaustive catalogue of failures, a validated causal model, or a complete diagnosis-system design.

The organizing structure is a **data-path / implementation × problem-category matrix**:

- **Row:** the concrete communication implementation being investigated.
- **Column:** a recurring family of performance problems.
- **Cell:** one or more specific issues or hypotheses for that implementation.

Read across a row to investigate different problems along one path. Read down a column to compare how the same problem family manifests across implementations.

## 1. Problem categories

| Category | Defining question | Representative issues |
|---|---|---|
| **Work distribution and timing** | Is demand unevenly distributed across participants or time? | Unequal bytes per rank, peer, CTA, or warp; skewed data readiness; bursty arrivals. |
| **Request generation and concurrency** | Can the implementation expose enough ready operations to keep its resources busy? | Too few issuing warps; insufficient independent operations; submission gaps; uneven issuing activity. |
| **Memory access and locality** | How efficiently does the access pattern translate useful payload into memory transactions and reuse? | Poor coalescing; scattered accesses; alignment effects; reduced buffer reuse; extra transactions per useful byte. |
| **Shared-resource contention** | Do simultaneously active operations compete for the same service capacity? | Competition at issuing resources, endpoint memory resources, transfer engines, or shared fabric resources. |
| **Synchronization and progress** | Which dependencies prevent subsequent work from advancing? | Readiness waits; completion notification; buffer-reuse waits; producer–consumer coordination; slow participants delaying dependent work. |

These categories need not be mutually exclusive. For example, unequal work can create a late participant, which causes synchronization waiting elsewhere. The initiating imbalance and the resulting wait belong to different parts of the explanation.

## 2. Working problem matrix

**All entries below are candidate issues to investigate, not claims that every issue occurs on every platform.** Resource paths, available mechanisms, and completion semantics must be established for the concrete implementation.

| Data path / implementation | Work distribution and timing | Request generation and concurrency | Memory access and locality | Shared-resource contention | Synchronization and progress |
|---|---|---|---|---|---|
| **Thread-issued peer loads/stores** | Unequal bytes per peer/CTA; staggered data readiness. | Too few issuing warps; insufficient independent accesses; uneven issuing activity. | Poor coalescing; scattered peer accesses; changes in buffer reuse. | Competition for issuing SM resources, endpoint memory resources, or shared fabric capacity. | Waiting for readiness, completion notification, or buffer reuse; a slow CTA delays dependent work. |
| **TMA-based transfer over a specified supported route** | Unequal transfer sizes; uneven production of source tiles. | Too few outstanding transfers; descriptor submission gaps. | Transfer layout/alignment; staging-buffer access patterns. | Competition along the actual source–destination route. | Transfer-completion waits; staging-buffer availability; producer–consumer coordination. |
| **Copy-engine-based peer transfer** | Unequal copy sizes; skewed submission times. | Submission gaps; insufficient queued work. | Buffer placement and reuse; source/destination access behavior. | Competition for copy-engine service, endpoint memory resources, or fabric capacity. | Stream/event dependencies; completion waits; buffer reuse. |
| **NVLS-based collective implementation** | Unequal readiness or work across participants. | Insufficient parallelism in invoking the implementation's operations. | Operand/output layout and locality. | Competition for resources used by the particular implementation. | Collective coordination; slow participants delay dependent stages. |

### How to interpret the rows

The row labels are practical implementation families, not a claim that all four are mutually exclusive mechanisms at the same architectural level. Thread ld/st, TMA, and copy engines describe issuing or movement mechanisms; NVLS describes a collective mechanism with additional implementation choices.

For an actual experiment, refine the row to specify:

- Source and destination memory locations and GPUs.
- Which GPU and execution mechanism issue the operations.
- Push versus pull, where relevant.
- Access pattern and participating threads/CTAs.
- Completion and synchronization semantics.

In particular, “TMA” alone does not specify a complete transfer path. Do not assume that every row is exposed by the same NCCL device API or supported by the available hardware.

## 3. Physical components locate issues within cells

Physical components answer **where a problem may occur**. The matrix columns answer **what kind of problem it is**. Keep these separate rather than adding HBM, NVLink, or SMs as sibling problem categories.

For example, the **thread ld/st × shared-resource contention** cell can contain:

| Candidate location | Specific hypothesis |
|---|---|
| Issuing SM resources | Concurrent computation reduces the communication kernel's ability to issue accesses. |
| Source memory resources | Concurrent source-side traffic slows servicing of the payload. |
| Destination memory resources | Concurrent destination-side traffic interferes with delivery or consumption. |
| Shared fabric resources | Another transfer reduces the service available to the victim. |

An experiment may establish interference without identifying its exact internal hardware location. An injected memory-intensive kernel, for example, also uses execution resources; its presence does not prove that the slowdown originates in HBM.

## 4. High-level workloads map to multiple cells

Workload scenarios describe combinations of demand, dependencies, and resource sharing. They do not need to become additional problem categories.

| Scenario | Potentially relevant categories | Concrete interpretation |
|---|---|---|
| **Expert-parallel workload imbalance** | Work distribution and timing; shared-resource contention; synchronization and progress. | More tokens/bytes at particular ranks; more incoming demand at selected endpoints; late participants delay dependent stages. |
| **Compute–communication fused kernel** | Request generation and concurrency; memory access and locality; shared-resource contention; synchronization and progress. | Computation changes when communication becomes ready, which execution resources it can use, its memory behavior, and producer–consumer dependencies. |
| **Communication plus a memory-intensive workload** | Shared-resource contention; potentially memory access and locality, and request generation and concurrency. | The workloads overlap on memory resources and may also share execution resources or alter cache state. |
| **Multiple concurrent communication flows** | Shared-resource contention; potentially work distribution and timing. | Transfers overlap at shared endpoints or fabric resources, with effects depending on direction and timing. |

These are hypotheses about possible interactions, not predetermined explanations of observed slowdown.

