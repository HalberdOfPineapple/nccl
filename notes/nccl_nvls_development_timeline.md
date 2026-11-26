# NCCL NVLS Development Timeline: Classic NVLS and Symmetric-Memory Collectives

## Summary

**Classic NVLS predates NCCL's public symmetric-memory API.** NCCL initially managed NVSwitch multicast memory internally, allowing ordinary `ncclAllReduce` calls to use NVLink SHARP without requiring applications to register symmetric buffers. Later releases added registration-based optimizations, dedicated symmetric-memory collective kernels, and a device-side programming interface.

The historical progression is:

1. **NCCL 2.17 (2023):** Classic NVLS collective support.
2. **NCCL 2.19 (2023–2024):** NVLS user-buffer registration to avoid staging copies in eligible cases.
3. **NCCL 2.27 (July 2025):** Public symmetric-memory windows and specialized symmetric collective kernels (SymK).
4. **NCCL 2.28 (2025):** Device API exposing Multimem and reimplementation of symmetric kernels using that API.

**Terminology caveat:** NVLink SHARP / NVLS can refer to hardware acceleration, while `NCCL_ALGO=NVLS` specifically selects NCCL's traditional software algorithm. A symmetric kernel can use NVLink SHARP hardware without running the classic NCCL NVLS algorithm.

## 1. Release Timeline

| Release | Main development | Significance |
|---|---|---|
| **2.17** | NVLink SHARP / NVLS support for AllReduce | NCCL manages multicast memory internally and supports acceleration through ordinary collective APIs. |
| **2.19** | NVLS user-buffer registration | Eligible user buffers can be registered to reduce or eliminate staging, while retaining the classic NVLS execution framework. |
| **2.27** | Symmetric-memory window registration and specialized kernels | Applications can register symmetric buffers; new SymK kernels can operate directly on those windows. |
| **2.28** | Device API and SymK reimplementation | Multimem becomes a communication mechanism accessible to user-authored device kernels through NCCL's Device API. |

### NCCL 2.17: Classic NVLS

NCCL integrated Hopper-generation NVSwitch SHARP functionality into its established collective framework. The classic NVLS implementation uses multicast bindings and pipeline mechanisms managed by NCCL, so the application does not need to create a symmetric-memory window.

In the usual unregistered case, NCCL copies input data into its internal multicast-capable buffers, issues Multimem reduction and multicast-store instructions, then copies the outputs back. The algorithm uses channels, step buffers, and producer-consumer synchronization.

Official source: [NCCL 2.17.1 Release Notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2203/release-notes/rel_2-17-1.html).

### NCCL 2.19: NVLS User-Buffer Registration

NVLS registration support appeared **before** the symmetric-window API. This is an important intermediate step: classic NVLS is not necessarily synonymous with staging copies. With eligible buffers and registration, the classic NVLS algorithm can use direct user-buffer access and avoid some data movement.

Thus, a useful distinction is **registered classic NVLS versus symmetric-kernel NVLS**, not just *copying versus zero-copy*.

Official source: [NCCL User Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/usage/bufferreg.html). An earlier edition is also available at [NCCL 2.20 User Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2203/user-guide/docs/usage/bufferreg.html).

### NCCL 2.27: Symmetric Memory and Specialized Collective Kernels

NCCL added public symmetric-window support, including:

- `ncclCommWindowRegister`
- `NCCL_WIN_COLL_SYMMETRIC`
- Specialized symmetric collective kernels, with implementations that can use NVLS connectivity.

A kernel such as `AllReduce_RSxLDMC_AGxSTMC` directly performs reduction and multicast-store operations over appropriately registered user buffers. Unlike the usual classic NVLS pipeline, it does not have to copy data through reusable NCCL staging slots or synchronize each pipeline step with the same producer-consumer flag scheme.

Official source: [Enabling Fast Inference and Resilient Training with NCCL 2.27](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/).

**Qualification:** The NVIDIA blog presents the symmetric-memory architecture and its benefits; it does not provide a detailed comparison of the internally named `RSxLDMC_AGxSTMC` kernel against classic NVLS.

### NCCL 2.28: Device API

NCCL introduced the experimental device-side communication API, making communication mechanisms available to CUDA kernels, including:

- **LSA:** Load/store access across peer GPUs.
- **Multimem:** Multicast and reduction functionality on supported NVLink/NVSwitch systems.
- **GIN:** GPU-initiated networking beyond the local NVLink domain.

The 2.28.3 release notes also document the reimplementation of symmetric kernels using the Device API.

Official sources:

- [Fusing Communication and Compute with New Device API and Copy Engine Collectives in NVIDIA NCCL 2.28](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/)
- [NCCL 2.28.3 Release Notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2283/release-notes/rel_2-28-3.html)
- [NCCL Device-Initiated Communication Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)

## 2. Comparing the Two AllReduce Paths in NCCL 2.32.3

The supplied *NVLS AllReduce Call Paths* report traces two paths from `ncclAllReduce` to the GPU instructions on eight H100 GPUs.

| Dimension | Classic NVLS | Symmetric-memory NVLS |
|---|---|---|
| Historical lineage | NCCL 2.17 | NCCL 2.27 |
| Selection / scheduling | Traditional collective algorithm (`ncclGetAlgoInfo`) | Symmetric kernel scheduler (`ncclMakeSymmetricTaskList`) |
| Example kernel | `RunWorkColl<..., NVLS, SIMPLE>::run` | `ncclSymkRun_AllReduce_RSxLDMC_AGxSTMC` |
| Memory accessed by Multimem | NCCL-managed buffers in the unregistered case; eligible registered buffers in optimized cases | User buffers registered as symmetric windows |
| Data staging in the reported experiment | Yes | No |
| Synchronization | Per-step flags and pipeline stages | Barriers at start/end, no per-chunk pipeline flags in the data loop |
| Host-visible API | `ncclAllReduce` | `ncclAllReduce` |
| Key instructions | `multimem.ld_reduce`, `multimem.st` | `multimem.ld_reduce`, `multimem.st` |
| Tested CTA configuration, 8 H100s | 16 CTAs × 640 threads | 2 CTAs × 512 threads |

### Classic path: NCCL-managed multicast buffer

```text
User input
   ↓ scatter/copy (unregistered case)
NCCL-managed multicast buffer
   ↓ multimem.ld_reduce + multimem.st
NCCL-managed output staging
   ↓ gather/copy (unregistered case)
User output
```

This implementation fits NCCL's established channel-based collective infrastructure and can accommodate more general execution environments.

### Symmetric-kernel path: direct operation on registered windows

```text
User symmetric-memory input window
   ↓ multimem.ld_reduce
Reduced values in issuing GPU's registers
   ↓ multimem.st
User symmetric-memory output window on all ranks
```

This architecture shifts setup complexity toward collective window registration and simplifies the actual kernel's repeated data-processing loop.

**Hardware-versus-API distinction:** Both implementations require GPU memory bound into a multicast object at matching offsets. Classic NVLS constructs the relevant layout internally. The newer symmetric-memory API exposes registration of compatible application allocations. This is not the same as claiming that every GPU must use an identical virtual address for its multicast mapping.

## 3. Reported Measurements and Their Interpretation

The uploaded report uses NCCL 2.32.3 with eight H100 GPUs, float sum, and two specific configurations: classic NVLS with unregistered buffers (`-R 0`), and a forced symmetric multicast kernel with symmetric-window registration (`-R 2`).

| Measurement | Classic NVLS | Symmetric multicast |
|---|---:|---:|
| CTAs selected | 16 | 2 |
| Threads per CTA | 640 | 512 |
| 32 MiB algorithm bandwidth | 165.5 GB/s | 248.1 GB/s |
| 2 GiB algorithm bandwidth | 270.3 GB/s | 276.2 GB/s |

The results suggest that specialized SymK execution reduces overhead at small and medium payloads, while both approaches converge toward similar large-message throughput.

**Limits of interpretation:** The experiment does not isolate the benefit of the kernel architecture from the benefit of user-buffer registration. The observed large-message ceiling does not, by itself, identify a specific NVSwitch or Multimem hardware bottleneck. These explanations require additional measurement.

A stronger three-way experiment would compare:

1. **Classic NVLS, unregistered buffers:** staging and traditional pipeline.
2. **Classic NVLS, registered buffers:** less staging, traditional pipeline.
3. **Symmetric-memory NVLS:** direct windows and specialized kernel.

This would disentangle memory-registration effects from scheduling and synchronization effects. Additional profiling could evaluate HBM traffic, warp stalls, step-flag overhead, and sensitivity to competing memory workloads.

## 4. Connection to NVSHMEM and Dynamic Expert Parallelism

NCCL's 2.27 symmetric-memory support should not be confused with the first appearance of symmetric-memory concepts or with **NVSHMEM**. NVSHMEM is a separate library and programming model supporting a partitioned global address space and one-sided GPU communication.

The development of NCCL follows three separate layers:

| Layer | Central issue |
|---|---|
| Hardware Multimem | Multicast address mappings, reductions, and multicast stores |
| Memory management | Creating/binding multicast groups, registering eligible symmetric windows |
| Collective software | Scheduling, synchronization, channel pipelines, direct kernels |

For Expert Parallelism, symmetric memory simplifies direct access to registered data but does **not automatically provide per-token dynamic selection of multicast destinations**. Dynamic multicast membership and routing are separate design problems, which is why the existence of a symmetric-memory API alone does not resolve the restrictions faced by dynamic EP workloads.

## 5. Conclusions

1. **Yes:** Classic NVLS existed before NCCL's public symmetric-memory support, using internally configured multicast memory.
2. The evolution was not a one-step transition. **NVLS buffer registration in NCCL 2.19** preceded the full symmetric-window and SymK architecture of 2.27.
3. The two current AllReduce paths reuse similar hardware Multimem capabilities but implement different collective-execution models.
4. **NVLS hardware acceleration** and the **classic NCCL NVLS algorithm** should be distinguished in papers and benchmarks.
5. An informative comparison must include registered classic NVLS to separate zero-copy gains from changes in kernel organization.

## References

1. NVIDIA, **NCCL 2.17.1 Release Notes**: https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2203/release-notes/rel_2-17-1.html
2. NVIDIA, **NCCL User Buffer Registration**: https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/usage/bufferreg.html
3. NVIDIA, **NCCL 2.20 User Buffer Registration (archive)**: https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2203/user-guide/docs/usage/bufferreg.html
4. NVIDIA, **Enabling Fast Inference and Resilient Training with NCCL 2.27**: https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/
5. NVIDIA, **Fusing Communication and Compute with New Device API and Copy Engine Collectives in NVIDIA NCCL 2.28**: https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/
6. NVIDIA, **NCCL 2.28.3 Release Notes**: https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2283/release-notes/rel_2-28-3.html
7. NVIDIA, **NCCL Device-Initiated Communication Documentation**: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html
8. User-provided report, **NVLS AllReduce Call Paths** (2026-11-17), NCCL 2.32.3 source walkthrough and H100 performance measurements.
