// TMA plan 01 harness: an NCCL LSA push made of TMA bulk stores, next to TMA memory co-runners.
//
// Same layout and completion protocol as thread_store/common/src/store_congest.cu (plan 07), with
// these differences:
//   - The copy is a TMA push. Thread 0 of each copy CTA asks the bulk-copy unit (TMA) to store
//     chunks from shared memory into the destination GPU's window (cp.async.bulk, global <- shared).
//     The other copy threads fill each shared-memory slot with the same position-dependent values
//     as the plan-07 copy, so the destination check still catches missing and misplaced data.
//     No source HBM is read, so source memory traffic comes only from the co-runner.
//   - The co-runner ("writer" role) is a TMA bulk-store writer (kind 0), the TMA bulk-read
//     co-runner of plan 04 (kind 1), or the plan-07 thread-store writer (kind 2, a contrast).
//   - The co-runner can run on its own SMs of either GPU (writer_rank), or on the copy's own SMs
//     (writer_colocated): then the copy and the co-runner share each SM's TMA unit.
//   - Both GPUs launch one CTA per SM and take roles from their own role table.
// Copy CTAs synchronise with named barrier 1 and co-runner threads with named barrier 2, so the two
// thread groups of a shared SM never wait for each other.
// JSON keys are the experiment data interface; renaming C++ fields must not rename keys.
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <nccl.h>
#include <nccl_device.h>
#include <pthread.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "hbm_read_pressure.cuh"

#define CUDA_CHECK(call) do { \
  cudaError_t status = (call); \
  if (status != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
    std::exit(2); \
  } \
} while (0)
#define NCCL_CHECK(call) do { \
  ncclResult_t status = (call); \
  if (status != ncclSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, ncclGetErrorString(status)); \
    std::exit(3); \
  } \
} while (0)

constexpr int maxSms = 132;
constexpr int maxThreadsPerCta = 1024;
constexpr int checkpointCount = 8;
constexpr int maxChunksInFlight = 8;
constexpr size_t writerBufferBytes = size_t(1) << 30;
constexpr int coRunnerChunkBytes = 64 * 1024;          // co-runner block or chunk, as in plan 07
constexpr int coRunnerChunkVectors = coRunnerChunkBytes / 16;
constexpr int colocatedGroupThreads = 64;              // the last two warps of a copy CTA host a co-runner
constexpr int allGpcs = 0xff;
enum Role { roleIdle = 0, roleCopy = 1, roleWriter = 2, roleReceiver = 3 };
enum CoRunnerKind { kindTmaStore = 0, kindTmaRead = 1, kindThreadStore = 2 };
static const char* roleNames[] = {"idle", "copy", "writer", "receiver"};

struct alignas(128) Signal { int value; };
struct WindowControl {
  Signal destinationReady;
  Signal acknowledgement;
  Signal completion[maxSms];
};
struct LocalControl {
  int copyRegistered;      // gives each copy CTA its part of the payload, in arrival order
  int writerRegistered;    // gives each co-runner its slice of the co-runner buffer
  int writersReady;
  int finishedCount;
  int stop;
};
struct CtaRecord {
  unsigned long long startNs;          // A2 / A8
  unsigned long long issueEndNs;       // A3: last bulk store issued
  unsigned long long bulkWaitEndNs;    // A17: every bulk store of this CTA completed
  unsigned long long fenceEndNs;       // A4
  unsigned long long publishEndNs;     // A5
  unsigned long long endNs;            // A1 / A6 / A8
  unsigned long long workUnits;        // A9: 64 KiB co-runner chunks or blocks
  unsigned long long slotWaitCycles;   // A15: thread 0 waiting for a free staging slot
  unsigned long long fillCycles;       // A16: thread 0 from a free slot to the bulk store (fill + barriers)
  unsigned long long issueLoopCycles;  // A15/A16 denominator: thread 0's whole issue loop
  unsigned long long checkpointsNs[checkpointCount]; // A7 issue progress, not arrival
  int smId;                            // A10
  int role;                            // A10: Role
  int roleIndex;                       // copy part or co-runner slice
};
struct Config {
  int sourceGpu = -1;
  int destinationGpu = -1;
  // Copy: CTAs, fill threads per CTA, chunk size, staging slots, and where it runs.
  int copyCtas = 16;
  int copyThreads = 512;
  int copyChunkBytes = 64 * 1024;
  int copyChunksInFlight = 2;
  int copyGenerate = 1;                // 1 fill every chunk with its own values; 0 fill each slot once
  int copyGpcMask = allGpcs;
  int copyPlacement = 0;               // 0 spread round-robin over GPCs, 1 packed into the largest GPCs
  // Co-runner: kind, CTAs, where it runs, and the target total rate (0 = no limit).
  int writerKind = kindTmaStore;
  int writerCtas = 0;
  int writerThreads = 1024;            // thread-store writer only
  int chunksInFlight = 1;              // TMA co-runners: 64 KiB chunks in flight per CTA
  int writerGpcMask = allGpcs;
  int writerPlacement = 0;
  int writerColocated = 0;             // 1: one co-runner in each copy CTA (same SM), none elsewhere
  int writerRank = 0;                  // GPU that runs the co-runner: 0 source, 1 destination
  int receiverGpcMask = allGpcs;       // GPCs allowed for the destination's receiver CTA
  int targetGbps = 0;
  int writerIdle = 0;                  // co-runners start and wait for stop without traffic (the reference)
  int writerAlternate = 0;             // 1 idle first, 2 active first; alternates per launch inside a run
  int useUnmapped = 0;                 // allow roles on SMs the GPC map could not place, after mapped SMs
  int blockThreads = 1024;
  int sharedBytes = 0;
  int iterations = 12;
  int warmup = 3;
  int detailed = 1;
  int communicationEnabled = 1;        // 0: no copy; a timer CTA keeps a window (co-runner-alone runs)
  size_t payloadBytes = size_t(2048) << 20;
  unsigned long long soloDurationNs = 20000000;
};
struct KernelArguments {
  ncclWindow_t window;
  uint8_t* writerBuffer;
  unsigned long long* writerChunks;
  unsigned long long* readChecksums;   // kind 1: per co-runner sum of the first word of every chunk read
  const int* roleOfSm;
  LocalControl* local;
  CtaRecord* records;                  // 2 × grid: [grid, 2 × grid) holds co-runners that share a copy SM
  Config config;
  HbmReadSettings tmaSettings;         // TMA co-runners: buffer, one slice per co-runner, chunks in flight, rate
  int rank;
  int iteration;
  int writerIdle;
  int gridCtas;
  int localWriterCtas;                 // co-runners on this GPU; 0 on the GPU without them
  unsigned long long writerIntervalPs;
};

__device__ __forceinline__ unsigned long long readTimerNs() {
  unsigned long long timestampNs;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestampNs));
  return timestampNs;
}
__device__ __forceinline__ int readSmId() {
  int smId;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smId));
  return smId;
}
__device__ __forceinline__ int acquireSignal(int* address) {
  return cuda::atomic_ref<int, cuda::thread_scope_system>(*address).load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void releaseSignal(int* address, int value) {
  cuda::atomic_ref<int, cuda::thread_scope_system>(*address).store(value, cuda::memory_order_release);
}
__device__ __forceinline__ WindowControl* windowControl(ncclWindow_t window, size_t payloadBytes, int rank) {
  return static_cast<WindowControl*>(ncclGetLsaPointer(window, payloadBytes, rank));
}
// Same value function as plan 07; it depends on the position and the transfer, so the destination
// check catches missing and misplaced writes.
__host__ __device__ __forceinline__ uint32_t expectedValue(size_t offset, int iteration) {
  return uint32_t(offset) * 1664525u ^ uint32_t(iteration + 1) * 1013904223u;
}
__device__ __forceinline__ uint4 writerValue(size_t vectorIndex, int iteration) {
  size_t word = vectorIndex * 4;
  return make_uint4(expectedValue(word, iteration), expectedValue(word + 1, iteration), expectedValue(word + 2, iteration), expectedValue(word + 3, iteration));
}
__host__ __device__ __forceinline__ uint32_t tmaWriterPattern(int iteration) {
  return 0x5a000000u | uint32_t(iteration + 1);
}
// Named barriers: the copy threads and the co-runner threads of one CTA wait only for their own group.
__device__ __forceinline__ void syncCopyThreads(int copyThreads) {
  asm volatile("bar.sync 1, %0;" :: "r"(copyThreads) : "memory");
}
__device__ __forceinline__ void syncCoRunnerThreads(int groupThreads) {
  asm volatile("bar.sync 2, %0;" :: "r"(groupThreads) : "memory");
}
// Ordinary shared-memory writes must be visible to the bulk-copy unit before it reads them.
__device__ __forceinline__ void orderSharedWritesBeforeBulkCopy() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
// The wait counts must be immediate operands, hence the switches.
// Reads: at most `pending` bulk-store groups of this thread still read their shared-memory source.
__device__ __forceinline__ void waitForBulkSourceReads(int pending) {
  switch (pending) {
    case 0: asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); break;
    case 1: asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); break;
    case 2: asm volatile("cp.async.bulk.wait_group.read 2;" ::: "memory"); break;
    case 3: asm volatile("cp.async.bulk.wait_group.read 3;" ::: "memory"); break;
    case 4: asm volatile("cp.async.bulk.wait_group.read 4;" ::: "memory"); break;
    case 5: asm volatile("cp.async.bulk.wait_group.read 5;" ::: "memory"); break;
    case 6: asm volatile("cp.async.bulk.wait_group.read 6;" ::: "memory"); break;
    default: asm volatile("cp.async.bulk.wait_group.read 7;" ::: "memory"); break;
  }
}
// Writes: at most `pending` bulk-store groups of this thread have writes that are not complete.
__device__ __forceinline__ void waitForBulkStores(int pending) {
  switch (pending) {
    case 0: asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); break;
    case 1: asm volatile("cp.async.bulk.wait_group 1;" ::: "memory"); break;
    case 2: asm volatile("cp.async.bulk.wait_group 2;" ::: "memory"); break;
    case 3: asm volatile("cp.async.bulk.wait_group 3;" ::: "memory"); break;
    case 4: asm volatile("cp.async.bulk.wait_group 4;" ::: "memory"); break;
    case 5: asm volatile("cp.async.bulk.wait_group 5;" ::: "memory"); break;
    case 6: asm volatile("cp.async.bulk.wait_group 6;" ::: "memory"); break;
    default: asm volatile("cp.async.bulk.wait_group 7;" ::: "memory"); break;
  }
}
__device__ __forceinline__ void startBulkStore(void* globalDestination, const void* sharedSource, unsigned bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :: "l"(globalDestination), "r"(hbm_read_pressure::sharedAddress(sharedSource)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// Checks every word of the payload. With copy_generate 0 every chunk holds the same values, so only
// missing writes are caught, not misplaced chunks.
__global__ void verifyDestination(const uint32_t* destination, size_t wordCount, size_t chunkWords, int generated, int iteration, unsigned long long* mismatches) {
  unsigned long long localMismatches = 0;
  for (size_t offset = size_t(blockIdx.x) * blockDim.x + threadIdx.x; offset < wordCount; offset += size_t(blockDim.x) * gridDim.x) {
    size_t position = generated ? offset : offset % chunkWords;
    localMismatches += destination[offset] != expectedValue(position, iteration);
  }
  if (localMismatches) {
    atomicAdd(mismatches, localMismatches);
  }
}
// Thread-store writer: every vector written in this launch; a writer that wrapped has rewritten its whole slice.
__global__ void verifyThreadWriterBuffer(const uint4* buffer, int writerCtas, size_t sliceBlocks, const unsigned long long* writerBlocks, int iteration, unsigned long long* mismatches) {
  const size_t sliceVectors = sliceBlocks * coRunnerChunkVectors;
  const size_t totalVectors = sliceVectors * writerCtas;
  unsigned long long localMismatches = 0;
  for (size_t vectorIndex = size_t(blockIdx.x) * blockDim.x + threadIdx.x; vectorIndex < totalVectors; vectorIndex += size_t(blockDim.x) * gridDim.x) {
    size_t writer = vectorIndex / sliceVectors;
    size_t writtenBlocks = writerBlocks[writer] < sliceBlocks ? writerBlocks[writer] : sliceBlocks;
    if (vectorIndex % sliceVectors < writtenBlocks * coRunnerChunkVectors) {
      uint4 value = buffer[vectorIndex];
      uint4 expected = writerValue(vectorIndex, iteration);
      localMismatches += value.x != expected.x || value.y != expected.y || value.z != expected.z || value.w != expected.w;
    }
  }
  if (localMismatches) {
    atomicAdd(mismatches, localMismatches);
  }
}
// TMA store writer: every word of the chunks written in this launch holds the launch's pattern word.
__global__ void verifyTmaWriterBuffer(const uint32_t* buffer, int writerCtas, size_t sliceChunks, const unsigned long long* writerChunks, uint32_t pattern, unsigned long long* mismatches) {
  const size_t chunkWords = coRunnerChunkBytes / 4;
  const size_t sliceWords = sliceChunks * chunkWords;
  unsigned long long localMismatches = 0;
  for (size_t wordIndex = size_t(blockIdx.x) * blockDim.x + threadIdx.x; wordIndex < sliceWords * writerCtas; wordIndex += size_t(blockDim.x) * gridDim.x) {
    size_t writtenChunks = writerChunks[wordIndex / sliceWords] < sliceChunks ? writerChunks[wordIndex / sliceWords] : sliceChunks;
    if (wordIndex % sliceWords < writtenChunks * chunkWords) {
      localMismatches += buffer[wordIndex] != pattern;
    }
  }
  if (localMismatches) {
    atomicAdd(mismatches, localMismatches);
  }
}

// ---------------------------------------------------------------------------------------------
// Copy: TMA push into the destination GPU's window.
// ---------------------------------------------------------------------------------------------

// Fills one staging slot with the values of payload words [firstWord, firstWord + chunk words).
__device__ __forceinline__ void fillStagingSlot(uint8_t* slot, size_t firstWord, int chunkBytes, int copyThreads, int iteration) {
  uint4* vectors = reinterpret_cast<uint4*>(slot);
  for (int vector = threadIdx.x; vector < chunkBytes / 16; vector += copyThreads) {
    size_t word = firstWord + size_t(vector) * 4;
    vectors[vector] = make_uint4(expectedValue(word, iteration), expectedValue(word + 1, iteration), expectedValue(word + 2, iteration), expectedValue(word + 3, iteration));
  }
}
// The copy starts only after every co-runner of this GPU has started, so co-runner traffic covers
// the whole copy, and only after the destination is ready.
__device__ void waitForCopyStart(KernelArguments arguments, CtaRecord& record) {
  WindowControl* localSignals = windowControl(arguments.window, arguments.config.payloadBytes, 0);
  if (threadIdx.x == 0) {
    while (acquireSignal(&arguments.local->writersReady) != arguments.localWriterCtas) {
      __nanosleep(64);
    }
    while (!acquireSignal(&localSignals->destinationReady.value)) {
      __nanosleep(64);
    }
    record.startNs = readTimerNs();
  }
  syncCopyThreads(arguments.config.copyThreads);
}
// Staging slots form a ring of `copyChunksInFlight` chunks. Before chunk j is filled, thread 0
// waits until the bulk store of chunk j - slots has finished reading its slot (wait_group.read);
// then the copy threads fill the slot and thread 0 starts its bulk store. Time spent in that wait
// is the copy's direct view of TMA back-pressure (A15): it grows when the bulk-copy unit cannot
// read slots out because the store path behind it is slow.
// With copy_generate 0 the slots are filled once and only thread 0 runs the loop.
__device__ void issueBulkCopy(KernelArguments arguments, uint8_t* destination, uint8_t* slots, size_t firstChunk, size_t chunkCount, CtaRecord& record) {
  const int chunkBytes = arguments.config.copyChunkBytes;
  const int slotCount = arguments.config.copyChunksInFlight;
  const int copyThreads = arguments.config.copyThreads;
  const size_t chunkWords = chunkBytes / 4;
  const bool generate = arguments.config.copyGenerate;
  const bool detailed = arguments.config.detailed;
  if (!generate) {
    for (int slot = 0; slot < slotCount; ++slot) {
      fillStagingSlot(slots + size_t(slot) * chunkBytes, 0, chunkBytes, copyThreads, arguments.iteration);
    }
    orderSharedWritesBeforeBulkCopy();
    syncCopyThreads(copyThreads);
    if (threadIdx.x != 0) {
      return;
    }
  }
  const size_t checkpointStride = (chunkCount + checkpointCount - 1) / checkpointCount;
  size_t nextCheckpoint = checkpointStride;
  int checkpointIndex = 0;
  unsigned long long slotWaitCycles = 0;
  unsigned long long fillCycles = 0;
  const unsigned long long loopStartCycles = clock64();
  #pragma unroll 1
  for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
    // chunk % slotCount marks the index of the slot in SMEM (ring-based) to be written to
    // `slots` is the pointer to the start of this area. TMA will read data from this area as a buffer
    // `slot` then becomes the new pointer to the slot to be written to.
    uint8_t* slot = slots + size_t(chunk % slotCount) * chunkBytes;
    unsigned long long slotFreeCycles = 0;
    if (threadIdx.x == 0) {
      unsigned long long waitStartCycles = clock64();
      // Wait for the bulk store of the slot that will be overwritten next to finish reading it.
      waitForBulkSourceReads(slotCount - 1);
      slotFreeCycles = clock64();
      slotWaitCycles += slotFreeCycles - waitStartCycles;
    }
    if (generate) {
      syncCopyThreads(copyThreads);
      fillStagingSlot(slot, (firstChunk + chunk) * chunkWords, chunkBytes, copyThreads, arguments.iteration);
      orderSharedWritesBeforeBulkCopy();
      syncCopyThreads(copyThreads);
    }
    if (threadIdx.x == 0) {
      fillCycles += clock64() - slotFreeCycles;
      startBulkStore(destination + (firstChunk + chunk) * chunkBytes, slot, chunkBytes);
      if (detailed && chunk + 1 == nextCheckpoint && checkpointIndex < checkpointCount) {
        record.checkpointsNs[checkpointIndex++] = readTimerNs();
        nextCheckpoint += checkpointStride;
      }
    }
  }
  if (threadIdx.x == 0) {
    record.issueLoopCycles = clock64() - loopStartCycles;
    record.slotWaitCycles = slotWaitCycles;
    record.fillCycles = fillCycles;
  }
}
// Completion: thread 0 waits until its bulk stores have completed, orders them before its own
// later writes (async proxy, then system scope), publishes "done" and waits for the acknowledgement.
// Only thread 0 issued peer writes, so no other copy thread needs a fence.
__device__ void completeBulkCopy(KernelArguments arguments, int copyIndex, CtaRecord& record) {
  WindowControl* localSignals = windowControl(arguments.window, arguments.config.payloadBytes, 0);
  WindowControl* remoteSignals = windowControl(arguments.window, arguments.config.payloadBytes, 1);
  const bool detailed = arguments.config.detailed;
  if (threadIdx.x != 0) {
    return;
  }
  if (detailed) {
    record.issueEndNs = readTimerNs();
  }
  waitForBulkStores(0);
  if (detailed) {
    record.bulkWaitEndNs = readTimerNs();
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __threadfence_system();
  if (detailed) {
    record.fenceEndNs = readTimerNs();
  }
  releaseSignal(&remoteSignals->completion[copyIndex].value, 1);
  if (detailed) {
    record.publishEndNs = readTimerNs();
  }
  while (!acquireSignal(&localSignals->acknowledgement.value)) {
    __nanosleep(64);
  }
  record.endNs = readTimerNs();
  if (atomicAdd(&arguments.local->finishedCount, 1) + 1 == arguments.config.copyCtas) {
    releaseSignal(&arguments.local->stop, 1);
  }
}
// Each copy CTA owns a contiguous run of whole chunks of the payload (chunk-granular CTA-local order).
__device__ void runCopy(KernelArguments arguments, int copyIndex) {
  extern __shared__ __align__(128) uint8_t dynamicShared[];
  CtaRecord& record = arguments.records[blockIdx.x];
  if (threadIdx.x >= arguments.config.copyThreads) {
    return;
  }
  waitForCopyStart(arguments, record);
  auto* destination = static_cast<uint8_t*>(ncclGetLsaPointer(arguments.window, 0, 1));
  const size_t totalChunks = arguments.config.payloadBytes / arguments.config.copyChunkBytes;
  const size_t firstChunk = totalChunks * copyIndex / arguments.config.copyCtas;
  const size_t endChunk = totalChunks * (copyIndex + 1) / arguments.config.copyCtas;
  issueBulkCopy(arguments, destination, dynamicShared, firstChunk, endChunk - firstChunk, record);
  completeBulkCopy(arguments, copyIndex, record);
}

// ---------------------------------------------------------------------------------------------
// Receiver and window timers.
// ---------------------------------------------------------------------------------------------

// The destination orders its co-runners around the transfer with signals, because the two GPUs'
// timers are not compared: co-runners start before "destination ready" and stop after the
// acknowledgement. publishEndNs marks the start of the window that they must cover (A8).
__device__ void runReceiver(KernelArguments arguments) {
  WindowControl* localSignals = windowControl(arguments.window, arguments.config.payloadBytes, 1);
  WindowControl* remoteSignals = windowControl(arguments.window, arguments.config.payloadBytes, 0);
  if (threadIdx.x == 0) {
    CtaRecord& record = arguments.records[blockIdx.x];
    while (acquireSignal(&arguments.local->writersReady) != arguments.localWriterCtas) {
      __nanosleep(64);
    }
    record.publishEndNs = readTimerNs();
    releaseSignal(&remoteSignals->destinationReady.value, 1);
    for (int copyIndex = 0; copyIndex < arguments.config.copyCtas; ++copyIndex) {
      while (!acquireSignal(&localSignals->completion[copyIndex].value)) {
        __nanosleep(64);
      }
    }
    record.issueEndNs = readTimerNs();
    releaseSignal(&remoteSignals->acknowledgement.value, 1);
    record.endNs = readTimerNs();
    releaseSignal(&arguments.local->stop, 1);
  }
}
// Co-runner-alone runs: one CTA of the co-runner's GPU times a fixed window after all co-runners
// started, then stops them (copy CTA 0 on the source, the receiver CTA on the destination).
__device__ void runWindowTimer(KernelArguments arguments) {
  if (threadIdx.x == 0) {
    CtaRecord& record = arguments.records[blockIdx.x];
    while (acquireSignal(&arguments.local->writersReady) != arguments.localWriterCtas) {
      __nanosleep(64);
    }
    record.publishEndNs = readTimerNs();
    while (readTimerNs() - record.publishEndNs < arguments.config.soloDurationNs) {
      __nanosleep(128);
    }
    record.endNs = readTimerNs();
    releaseSignal(&arguments.local->stop, 1);
  }
}

// ---------------------------------------------------------------------------------------------
// Co-runners. Each runs in a thread group [groupBegin, groupBegin + groupThreads) of its CTA:
// the whole CTA on a co-runner SM, the last two warps of a copy CTA when colocated.
// ---------------------------------------------------------------------------------------------
struct CoRunnerGroup {
  int writerIndex;
  int groupBegin;
  int groupThreads;
  uint8_t* sharedBase;
  CtaRecord* record;
};

__device__ __forceinline__ void registerCoRunner(KernelArguments arguments, CtaRecord& record) {
  record.startNs = readTimerNs();
  cuda::atomic_ref<int, cuda::thread_scope_system>(arguments.local->writersReady).fetch_add(1, cuda::memory_order_release);
}
__device__ void waitIdle(volatile int* sharedStopFlag, CtaRecord& record) {
  while (!hbm_read_pressure::stopRequested(sharedStopFlag)) {
    __nanosleep(256);
  }
  record.endNs = readTimerNs();
}

// Kind 0: rate-controlled TMA bulk stores of one 64 KiB shared-memory chunk into this co-runner's
// slice of the local buffer (plan 07 E7.6). The chunk is filled once per launch, so the source
// never changes; `chunks_in_flight` bounds the chunks whose writes have not completed. The end time
// is taken after every write has completed. The issuing thread forwards nothing: a second warp
// watches the global stop flag (one memory round trip per chunk would limit the rate).
__device__ void runTmaStoreWriter(KernelArguments arguments, const CoRunnerGroup& group) {
  __shared__ int sharedStopFlag;
  const int localThread = threadIdx.x - group.groupBegin;
  CtaRecord& record = *group.record;
  const HbmReadSettings& settings = arguments.tmaSettings;
  uint32_t* source = reinterpret_cast<uint32_t*>(group.sharedBase);
  for (int word = localThread; word < coRunnerChunkBytes / 4; word += group.groupThreads) {
    source[word] = tmaWriterPattern(arguments.iteration);
  }
  if (localThread == 0) {
    sharedStopFlag = 0;
  }
  orderSharedWritesBeforeBulkCopy();
  syncCoRunnerThreads(group.groupThreads);
  if (localThread == 32) {
    hbm_read_pressure::forwardStopFlag(&arguments.local->stop, &sharedStopFlag);
  } else if (localThread == 0) {
    registerCoRunner(arguments, record);
    if (arguments.writerIdle) {
      waitIdle(&sharedStopFlag, record);
      return;
    }
    const unsigned long long sliceChunks = settings.workingSetBytes / settings.chunkBytes / settings.pressureCtas;
    uint8_t* slice = const_cast<uint8_t*>(settings.workingSet) + group.writerIndex * sliceChunks * settings.chunkBytes;
    unsigned long long chunks = 0;
    while (!hbm_read_pressure::stopRequested(&sharedStopFlag)) {
      hbm_read_pressure::waitForScheduledStart(settings, record.startNs, chunks);
      startBulkStore(slice + (chunks % sliceChunks) * settings.chunkBytes, source, settings.chunkBytes);
      ++chunks;
      waitForBulkStores(settings.chunksInFlight - 1);
    }
    waitForBulkStores(0);
    record.endNs = readTimerNs();
    record.workUnits = chunks;                   // A9: 64 KiB chunks written
    arguments.writerChunks[group.writerIndex] = chunks;
  }
}
// Kind 1: rate-controlled TMA bulk reads into shared memory (plan 04's read co-runner).
__device__ void runTmaReader(KernelArguments arguments, const CoRunnerGroup& group) {
  __shared__ int sharedStopFlag;
  const int localThread = threadIdx.x - group.groupBegin;
  CtaRecord& record = *group.record;
  if (localThread == 0) {
    sharedStopFlag = 0;
  }
  syncCoRunnerThreads(group.groupThreads);
  if (localThread == 32) {
    hbm_read_pressure::forwardStopFlag(&arguments.local->stop, &sharedStopFlag);
  } else if (localThread == 0) {
    registerCoRunner(arguments, record);
    if (arguments.writerIdle) {
      waitIdle(&sharedStopFlag, record);
      return;
    }
    const HbmReadSettings& settings = arguments.tmaSettings;
    uint64_t* barriers = reinterpret_cast<uint64_t*>(group.sharedBase + size_t(settings.chunksInFlight) * settings.chunkBytes);
    HbmReadResult result;
    hbm_read_pressure::readUntilStopped(settings, group.writerIndex, group.sharedBase, barriers, &sharedStopFlag, result);
    record.endNs = result.endNs;
    record.workUnits = result.chunksRead;        // A9: 64 KiB chunks read
    arguments.writerChunks[group.writerIndex] = result.chunksRead;
    arguments.readChecksums[group.writerIndex] = result.checksum;
  }
}
// Rate limit for the thread-store writer: block k of this CTA starts at startNs + k × interval, never earlier.
__device__ __forceinline__ void waitForScheduledBlock(unsigned long long startNs, unsigned long long blockIndex, unsigned long long intervalPs) {
  if (intervalPs == 0) {
    return;
  }
  unsigned long long scheduledNs = startNs + blockIndex * intervalPs / 1000;
  while (readTimerNs() < scheduledNs) {
  }
}
// Kind 2: the plan-07 thread-store writer (16-byte stores, one 64 KiB block per group barrier). The
// stop flag is loaded while a block is written and passed on through shared memory one block later,
// so writers stop at most two blocks after the copy finishes; those blocks are counted.
__device__ void runThreadStoreWriter(KernelArguments arguments, const CoRunnerGroup& group) {
  __shared__ int stopSlots[2];
  const int localThread = threadIdx.x - group.groupBegin;
  CtaRecord& record = *group.record;
  const int writerThreads = arguments.config.writerThreads;
  const size_t sliceBlocks = writerBufferBytes / coRunnerChunkBytes / arguments.config.writerCtas;
  const size_t sliceFirstVector = size_t(group.writerIndex) * sliceBlocks * coRunnerChunkVectors;
  uint4* slice = reinterpret_cast<uint4*>(arguments.writerBuffer) + sliceFirstVector;
  if (localThread == 0) {
    stopSlots[0] = 0;
    stopSlots[1] = 0;
    registerCoRunner(arguments, record);
  }
  syncCoRunnerThreads(group.groupThreads);
  if (arguments.writerIdle) {
    if (localThread == 0) {
      while (!acquireSignal(&arguments.local->stop)) {
        __nanosleep(256);
      }
      record.endNs = readTimerNs();
    }
    return;
  }
  const unsigned long long startNs = record.startNs;
  int pendingStop = 0;
  unsigned long long blocks = 0;
  for (;; ++blocks) {
    if (localThread == 0) {
      waitForScheduledBlock(startNs, blocks, arguments.writerIntervalPs);
    }
    syncCoRunnerThreads(group.groupThreads);
    if (stopSlots[(blocks + 1) & 1]) {
      break;
    }
    if (localThread == 0) {
      stopSlots[blocks & 1] = pendingStop;
      pendingStop = *reinterpret_cast<volatile int*>(&arguments.local->stop);
    }
    if (localThread < writerThreads) {
      const size_t blockFirstVector = (blocks % sliceBlocks) * coRunnerChunkVectors;
      for (int vector = localThread; vector < coRunnerChunkVectors; vector += writerThreads) {
        slice[blockFirstVector + vector] = writerValue(sliceFirstVector + blockFirstVector + vector, arguments.iteration);
      }
    }
  }
  if (localThread == 0) {
    record.endNs = readTimerNs();
    record.workUnits = blocks;                   // A9: 64 KiB blocks written
    arguments.writerChunks[group.writerIndex] = blocks;
  }
}
__device__ void runCoRunner(KernelArguments arguments, const CoRunnerGroup& group) {
  if (arguments.config.writerKind == kindTmaStore) {
    runTmaStoreWriter(arguments, group);
  } else if (arguments.config.writerKind == kindTmaRead) {
    runTmaReader(arguments, group);
  } else {
    runThreadStoreWriter(arguments, group);
  }
}

// ---------------------------------------------------------------------------------------------
// Fused kernel: one CTA per SM on both GPUs; each CTA takes the role its SM has in this GPU's table.
// ---------------------------------------------------------------------------------------------

// Thread 0 registers the CTA's role (and the co-runner of a colocated copy CTA) in arrival order.
__device__ void registerRoles(KernelArguments arguments, int role, int smId, int* roleIndex, int* colocatedIndex) {
  CtaRecord& record = arguments.records[blockIdx.x];
  record.smId = smId;
  record.role = role;
  record.startNs = readTimerNs();
  *colocatedIndex = -1;
  if (role == roleCopy) {
    *roleIndex = atomicAdd(&arguments.local->copyRegistered, 1);
    if (arguments.config.writerColocated && arguments.localWriterCtas) {
      *colocatedIndex = atomicAdd(&arguments.local->writerRegistered, 1);
      CtaRecord& coRunnerRecord = arguments.records[arguments.gridCtas + blockIdx.x];
      coRunnerRecord.smId = smId;
      coRunnerRecord.role = roleWriter;
      coRunnerRecord.roleIndex = *colocatedIndex;
    }
  } else if (role == roleWriter) {
    *roleIndex = atomicAdd(&arguments.local->writerRegistered, 1);
  } else if (role == roleReceiver) {
    *roleIndex = 0;
  } else {
    *roleIndex = -1;
    record.endNs = record.startNs;
  }
  record.roleIndex = *roleIndex;
}
__global__ void __launch_bounds__(maxThreadsPerCta, 1) tmaCongest(KernelArguments arguments) {
  extern __shared__ __align__(128) uint8_t dynamicShared[];
  __shared__ int roleIndex;
  __shared__ int colocatedIndex;
  const int smId = readSmId();
  const int role = arguments.roleOfSm[smId];
  const Config& config = arguments.config;
  if (threadIdx.x == 0) {
    registerRoles(arguments, role, smId, &roleIndex, &colocatedIndex);
  }
  __syncthreads();
  CtaRecord& record = arguments.records[blockIdx.x];
  if (role == roleReceiver) {
    if (config.communicationEnabled) {
      runReceiver(arguments);
    } else if (config.writerRank == 1) {
      runWindowTimer(arguments);
    }
    return;
  }
  if (role == roleWriter) {
    runCoRunner(arguments, CoRunnerGroup{roleIndex, 0, int(blockDim.x), dynamicShared, &record});
    return;
  }
  if (role != roleCopy) {
    return;
  }
  // A colocated co-runner uses the last two warps and the shared memory after the copy's slots.
  if (colocatedIndex >= 0 && int(threadIdx.x) >= int(blockDim.x) - colocatedGroupThreads) {
    uint8_t* coRunnerShared = dynamicShared + size_t(config.copyChunksInFlight) * config.copyChunkBytes;
    runCoRunner(arguments, CoRunnerGroup{colocatedIndex, int(blockDim.x) - colocatedGroupThreads, colocatedGroupThreads, coRunnerShared,
                                         &arguments.records[arguments.gridCtas + blockIdx.x]});
    return;
  }
  if (config.communicationEnabled) {
    runCopy(arguments, roleIndex);
  } else if (roleIndex == 0 && config.writerRank == 0) {
    runWindowTimer(arguments);
  } else if (threadIdx.x == 0) {
    record.endNs = record.startNs;
  }
}

// ---------------------------------------------------------------------------------------------
// Host: configuration.
// ---------------------------------------------------------------------------------------------
struct RankState {
  ncclComm_t communicator;
  ncclDevComm deviceCommunicator;
  ncclWindow_t window;
  void* windowMemory;
  uint8_t* writerBuffer;
  unsigned long long* writerChunks;
  unsigned long long* readChecksums;
  int* roleOfSm;
  LocalControl* local;
  CtaRecord* deviceRecords;
  unsigned long long* mismatches;
  cudaStream_t stream;
  int gridCtas;
  int localWriterCtas;
  size_t dynamicSharedBytes;
  cudaDeviceProp properties;
  std::vector<std::vector<CtaRecord>> records;
  std::vector<int> writerIdleByIteration;   // retained launches only
};
struct Placement {
  std::string mapPath;
  std::string mapUuid;
  std::vector<int> gpcOfSm;   // A12; -1 = not placed by cluster launches
  std::vector<int> roleOfSm;
};

// User allocation for TMA plan 01: physical GPUs 0, 1, 4, 5 from 2026-11-10 (pairs 0 -> 1 and 4 -> 5).
static bool authorizedGpu(int gpu) {
  return gpu == 0 || gpu == 1 || gpu == 4 || gpu == 5;
}

static Config parseConfiguration(const char* path) {
  Config config;
  std::ifstream input(path);
  if (!input) {
    fprintf(stderr, "Cannot open configuration %s\n", path);
    std::exit(1);
  }
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    auto separator = line.find('=');
    std::string key = line.substr(0, separator);
    long long value = std::stoll(line.substr(separator + 1));
    if (key == "copy_ctas") {
      config.copyCtas = value;
    } else if (key == "copy_threads") {
      config.copyThreads = value;
    } else if (key == "copy_chunk_kib") {
      config.copyChunkBytes = value * 1024;
    } else if (key == "copy_chunks_in_flight") {
      config.copyChunksInFlight = value;
    } else if (key == "copy_generate") {
      config.copyGenerate = value;
    } else if (key == "copy_gpcs") {
      config.copyGpcMask = value;
    } else if (key == "copy_placement") {
      config.copyPlacement = value;
    } else if (key == "writer_kind") {
      config.writerKind = value;
    } else if (key == "writer_ctas") {
      config.writerCtas = value;
    } else if (key == "writer_threads") {
      config.writerThreads = value;
    } else if (key == "chunks_in_flight") {
      config.chunksInFlight = value;
    } else if (key == "writer_gpcs") {
      config.writerGpcMask = value;
    } else if (key == "writer_placement") {
      config.writerPlacement = value;
    } else if (key == "writer_colocated") {
      config.writerColocated = value;
    } else if (key == "writer_rank") {
      config.writerRank = value;
    } else if (key == "receiver_gpcs") {
      config.receiverGpcMask = value;
    } else if (key == "target_gbps") {
      config.targetGbps = value;
    } else if (key == "writer_idle") {
      config.writerIdle = value;
    } else if (key == "writer_alternate") {
      config.writerAlternate = value;
    } else if (key == "use_unmapped") {
      config.useUnmapped = value;
    } else if (key == "block_threads") {
      config.blockThreads = value;
    } else if (key == "shared_bytes") {
      config.sharedBytes = value;
    } else if (key == "iterations") {
      config.iterations = value;
    } else if (key == "warmup") {
      config.warmup = value;
    } else if (key == "detailed") {
      config.detailed = value;
    } else if (key == "communication") {
      config.communicationEnabled = value;
    } else if (key == "payload_mib") {
      config.payloadBytes = size_t(value) << 20;
    } else if (key == "solo_ns") {
      config.soloDurationNs = value;
    } else {
      fprintf(stderr, "Unknown config key %s\n", key.c_str());
      std::exit(1);
    }
  }
  return config;
}
static void validateConfiguration(const Config& config) {
  const int chunk = config.copyChunkBytes;
  const int copyThreadLimit = config.blockThreads - (config.writerColocated ? colocatedGroupThreads : 0);
  bool copyValid = config.copyCtas >= 1 && config.copyCtas <= maxSms && config.copyThreads >= 32 && config.copyThreads % 32 == 0 && config.copyThreads <= copyThreadLimit &&
                   chunk >= 4096 && chunk <= 128 * 1024 && (chunk & (chunk - 1)) == 0 && config.payloadBytes % chunk == 0 && config.payloadBytes / chunk >= size_t(config.copyCtas) &&
                   config.copyChunksInFlight >= 1 && config.copyChunksInFlight <= maxChunksInFlight && config.copyGenerate >= 0 && config.copyGenerate <= 1;
  bool writerValid = config.writerCtas >= 0 && config.writerKind >= kindTmaStore && config.writerKind <= kindThreadStore && config.targetGbps >= 0 &&
                     config.chunksInFlight >= 1 && config.chunksInFlight <= maxChunksInFlight && config.writerRank >= 0 && config.writerRank <= 1 &&
                     config.writerThreads >= 32 && config.writerThreads % 32 == 0 && coRunnerChunkVectors % config.writerThreads == 0 && config.writerThreads <= config.blockThreads;
  // A colocated co-runner is one per copy CTA, on the source GPU, in two warps. A colocated
  // thread-store writer (added 2026-11-10, same-SM follow-up) stores with exactly those 64 threads.
  bool colocationValid = config.writerColocated == 0 ||
                         (config.writerColocated == 1 && config.writerCtas == config.copyCtas && config.writerRank == 0 &&
                          (config.writerKind != kindThreadStore || config.writerThreads == colocatedGroupThreads));
  bool budgetValid = config.writerColocated || config.writerRank == 1 ? config.copyCtas <= maxSms : config.copyCtas + config.writerCtas <= maxSms;
  bool alternationValid = config.writerAlternate >= 0 && config.writerAlternate <= 2 && !(config.writerAlternate && (config.writerIdle || config.writerCtas == 0));
  bool runValid = config.blockThreads <= maxThreadsPerCta && config.blockThreads % 32 == 0 && config.blockThreads >= 128 && config.iterations >= 1 && config.warmup >= 0 &&
                  config.soloDurationNs > 0 && config.payloadBytes >= (size_t(1) << 20) && config.payloadBytes <= (size_t(4096) << 20) &&
                  config.copyPlacement >= 0 && config.copyPlacement <= 1 && config.writerPlacement >= 0 && config.writerPlacement <= 1;
  if (!copyValid || !writerValid || !colocationValid || !budgetValid || !alternationValid || !runValid) {
    fprintf(stderr, "Invalid configuration (copy %d, writer %d, colocation %d, SM budget %d, alternation %d, run %d)\n",
            copyValid, writerValid, colocationValid, budgetValid, alternationValid, runValid);
    std::exit(1);
  }
}
// Shared memory one co-runner needs: kind 0 one 64 KiB chunk; kind 1 its chunks plus one barrier each.
static size_t coRunnerSharedBytes(const Config& config) {
  if (config.writerKind == kindTmaStore) {
    return coRunnerChunkBytes;
  }
  if (config.writerKind == kindTmaRead) {
    return size_t(config.chunksInFlight) * (coRunnerChunkBytes + sizeof(uint64_t));
  }
  return 0;
}
// Shared memory the busiest CTA of this rank needs. Every CTA of a launch gets the same reservation,
// so the runner sets shared_bytes to one value for every arm of a comparison.
static size_t requiredSharedBytes(const Config& config, int rank) {
  size_t coRunner = config.writerCtas ? coRunnerSharedBytes(config) : 0;
  if (rank == 1) {
    return config.writerRank == 1 ? coRunner : 0;
  }
  size_t copy = size_t(config.copyChunksInFlight) * config.copyChunkBytes;
  if (config.writerColocated) {
    return copy + coRunner;
  }
  return std::max(copy, config.writerRank == 0 ? coRunner : size_t(0));
}

// ---------------------------------------------------------------------------------------------
// Host: SM-to-GPC map and role tables (same rules as plan 07).
// ---------------------------------------------------------------------------------------------
static std::string uuidText(const cudaUUID_t& uuid) {
  char text[64];
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(uuid.bytes);
  snprintf(text, sizeof(text), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
           bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
  return text;
}
// Reads the map written by thread_store's build_gpc_map.py: {"uuid": "...", ..., "gpc_of_sm": [g0, g1, ...]}.
static void loadGpcMap(Placement& placement) {
  std::ifstream input(placement.mapPath);
  if (!input) {
    fprintf(stderr, "Cannot open GPC map %s\n", placement.mapPath.c_str());
    std::exit(1);
  }
  std::stringstream content;
  content << input.rdbuf();
  std::string text = content.str();
  size_t uuidKey = text.find("\"uuid\": \"");
  size_t listKey = text.find("\"gpc_of_sm\": [");
  if (uuidKey == std::string::npos || listKey == std::string::npos) {
    fprintf(stderr, "GPC map lacks uuid or gpc_of_sm\n");
    std::exit(1);
  }
  size_t uuidBegin = uuidKey + strlen("\"uuid\": \"");
  placement.mapUuid = text.substr(uuidBegin, text.find('"', uuidBegin) - uuidBegin);
  std::stringstream list(text.substr(listKey + strlen("\"gpc_of_sm\": ["), text.find(']', listKey) - listKey - strlen("\"gpc_of_sm\": [")));
  std::string entry;
  while (std::getline(list, entry, ',')) {
    placement.gpcOfSm.push_back(std::stoi(entry));
  }
}
// Gives `count` free SMs the role. Spread: one SM per allowed GPC in turn (lowest SM IDs first).
// Packed: all free SMs of the largest allowed GPC, then the next largest. Unmapped SMs are used
// only if the config allows it and mapped SMs ran out.
static void assignRole(Placement& placement, int role, int count, int gpcMask, int packed, int useUnmapped) {
  int gpcCount = *std::max_element(placement.gpcOfSm.begin(), placement.gpcOfSm.end()) + 1;
  std::vector<std::vector<int>> freeSmsByGpc(gpcCount);
  std::vector<int> unmappedSms;
  for (size_t smId = 0; smId < placement.gpcOfSm.size(); ++smId) {
    int gpc = placement.gpcOfSm[smId];
    if (placement.roleOfSm[smId] != roleIdle) {
      continue;
    }
    if (gpc < 0) {
      unmappedSms.push_back(smId);
    } else if (gpcMask >> gpc & 1) {
      freeSmsByGpc[gpc].push_back(smId);
    }
  }
  std::vector<int> gpcOrder;
  for (int gpc = 0; gpc < gpcCount; ++gpc) {
    gpcOrder.push_back(gpc);
  }
  if (packed) {
    std::stable_sort(gpcOrder.begin(), gpcOrder.end(), [&](int left, int right) { return freeSmsByGpc[left].size() > freeSmsByGpc[right].size(); });
  }
  std::vector<int> chosen;
  if (packed) {
    for (int gpc : gpcOrder) {
      for (int smId : freeSmsByGpc[gpc]) {
        chosen.push_back(smId);
      }
    }
  } else {
    for (size_t round = 0; chosen.size() < placement.gpcOfSm.size(); ++round) {
      bool tookAny = false;
      for (int gpc : gpcOrder) {
        if (round < freeSmsByGpc[gpc].size()) {
          chosen.push_back(freeSmsByGpc[gpc][round]);
          tookAny = true;
        }
      }
      if (!tookAny) {
        break;
      }
    }
  }
  if (useUnmapped) {
    chosen.insert(chosen.end(), unmappedSms.begin(), unmappedSms.end());
  }
  if (int(chosen.size()) < count) {
    fprintf(stderr, "Only %zu free SMs for %d %s CTAs\n", chosen.size(), count, roleNames[role]);
    std::exit(1);
  }
  for (int index = 0; index < count; ++index) {
    placement.roleOfSm[chosen[index]] = role;
  }
}
// Source: copy CTAs first, then co-runner CTAs (unless colocated). Destination: the receiver,
// then co-runner CTAs when writer_rank is 1. The writer's GPC mask gives the same SM IDs on either
// GPU, because the maps of the pairs measured so far are identical.
static void buildRoleTable(Placement& placement, const Config& config, const cudaDeviceProp& properties, int rank) {
  loadGpcMap(placement);
  if (placement.mapUuid != uuidText(properties.uuid) || int(placement.gpcOfSm.size()) != properties.multiProcessorCount || properties.multiProcessorCount > maxSms) {
    fprintf(stderr, "GPC map %s does not belong to this GPU (%s, %d SMs)\n", placement.mapPath.c_str(), uuidText(properties.uuid).c_str(), properties.multiProcessorCount);
    std::exit(1);
  }
  placement.roleOfSm.assign(properties.multiProcessorCount, roleIdle);
  if (rank == 0) {
    assignRole(placement, roleCopy, config.copyCtas, config.copyGpcMask, config.copyPlacement, config.useUnmapped);
  } else {
    assignRole(placement, roleReceiver, 1, config.receiverGpcMask, 0, 0);
  }
  if (rank == config.writerRank && !config.writerColocated) {
    assignRole(placement, roleWriter, config.writerCtas, config.writerGpcMask, config.writerPlacement, config.useUnmapped);
  }
}

// ---------------------------------------------------------------------------------------------
// Host: per-rank setup, launches, checks.
// ---------------------------------------------------------------------------------------------
static void chooseSharedReservation(RankState& state, const Config& config, int rank) {
  // More than half an SM's shared memory ensures no two CTAs share an SM.
  size_t oneCtaPerSm = ((state.properties.sharedMemPerMultiprocessor / 2 + 255) / 256) * 256;
  state.dynamicSharedBytes = config.sharedBytes ? size_t(config.sharedBytes) : oneCtaPerSm;
  if (state.dynamicSharedBytes < oneCtaPerSm || state.dynamicSharedBytes > state.properties.sharedMemPerBlockOptin) {
    fprintf(stderr, "shared_bytes must keep one CTA per SM and fit one CTA's limit (%zu-%zu)\n", oneCtaPerSm, state.properties.sharedMemPerBlockOptin);
    std::exit(1);
  }
  size_t required = requiredSharedBytes(config, rank);
  if (state.dynamicSharedBytes < required) {
    fprintf(stderr, "shared_bytes %zu too small for this rank's roles (%zu needed)\n", state.dynamicSharedBytes, required);
    std::exit(1);
  }
}
static void allocateBuffers(RankState& state, const Config& config, const Placement& placement, int rank) {
  CUDA_CHECK(cudaMalloc(&state.local, sizeof(LocalControl)));
  CUDA_CHECK(cudaMalloc(&state.deviceRecords, 2 * state.gridCtas * sizeof(CtaRecord)));
  CUDA_CHECK(cudaMalloc(&state.mismatches, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&state.roleOfSm, maxSms * sizeof(int)));
  CUDA_CHECK(cudaMemset(state.roleOfSm, 0, maxSms * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(state.roleOfSm, placement.roleOfSm.data(), placement.roleOfSm.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&state.writerChunks, maxSms * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&state.readChecksums, maxSms * sizeof(unsigned long long)));
  state.writerBuffer = nullptr;
  if (state.localWriterCtas) {
    CUDA_CHECK(cudaMalloc(&state.writerBuffer, writerBufferBytes));
    // The read co-runner reads byte 0x01, so every chunk's first word adds 0x01010101 to its checksum.
    CUDA_CHECK(cudaMemset(state.writerBuffer, config.writerKind == kindTmaRead ? 0x01 : 0, writerBufferBytes));
  }
}
static void setupRank(RankState& state, const Config& config, Placement& placement, int rank, ncclUniqueId identifier) {
  int device = rank == 0 ? config.sourceGpu : config.destinationGpu;
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaGetDeviceProperties(&state.properties, device));
  buildRoleTable(placement, config, state.properties, rank);
  NCCL_CHECK(ncclCommInitRank(&state.communicator, 2, identifier, rank));
  ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_CHECK(ncclCommQueryProperties(state.communicator, &properties));
  if (!properties.deviceApiSupport || properties.nLsaTeams != 1) {
    fprintf(stderr, "Requires a single two-GPU LSA team\n");
    std::exit(1);
  }
  size_t windowBytes = config.payloadBytes + sizeof(WindowControl);
  NCCL_CHECK(ncclMemAlloc(&state.windowMemory, windowBytes));
  NCCL_CHECK(ncclCommWindowRegister(state.communicator, state.windowMemory, windowBytes, &state.window, NCCL_WIN_COLL_SYMMETRIC));
  ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  NCCL_CHECK(ncclDevCommCreate(state.communicator, &requirements, &state.deviceCommunicator));
  state.gridCtas = state.properties.multiProcessorCount;
  state.localWriterCtas = rank == config.writerRank ? config.writerCtas : 0;
  CUDA_CHECK(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
  allocateBuffers(state, config, placement, rank);
  chooseSharedReservation(state, config, rank);
  CUDA_CHECK(cudaFuncSetAttribute(tmaCongest, cudaFuncAttributeMaxDynamicSharedMemorySize, state.dynamicSharedBytes));
  int residentCtas = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residentCtas, tmaCongest, config.blockThreads, state.dynamicSharedBytes));
  if (residentCtas != 1) {
    fprintf(stderr, "Expected exactly one resident CTA per SM, got %d\n", residentCtas);
    std::exit(1);
  }
  // Preload the measured function before concurrent launches (avoids lazy-load waits).
  cudaFuncAttributes attributes;
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, tmaCongest));
  fprintf(stderr, "rank=%d gpu=%d SMs=%d grid=%d block=%d copy=%dx%d chunk=%d slots=%d generate=%d writer_kind=%d writer=%d colocated=%d writer_rank=%d target_gbps=%d shared=%zu registers=%d local_bytes=%zu\n",
          rank, device, state.properties.multiProcessorCount, state.gridCtas, config.blockThreads, config.copyCtas, config.copyThreads, config.copyChunkBytes, config.copyChunksInFlight,
          config.copyGenerate, config.writerKind, config.writerCtas, config.writerColocated, config.writerRank, config.targetGbps, state.dynamicSharedBytes, attributes.numRegs, attributes.localSizeBytes);
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
static void prepareIteration(RankState& state, const Config& config, int rank) {
  CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(state.windowMemory) + config.payloadBytes, 0, sizeof(WindowControl), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.local, 0, sizeof(LocalControl), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.deviceRecords, 0, 2 * state.gridCtas * sizeof(CtaRecord), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.mismatches, 0, sizeof(unsigned long long), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.writerChunks, 0, maxSms * sizeof(unsigned long long), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.readChecksums, 0, maxSms * sizeof(unsigned long long), state.stream));
  if (rank == 1 && config.communicationEnabled) {
    CUDA_CHECK(cudaMemsetAsync(state.windowMemory, 0xff, config.payloadBytes, state.stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
// Whether the co-runners are idle in this launch; counts warm-ups so the order is fixed per run.
static int launchIsIdle(const Config& config, int iteration) {
  if (!config.writerAlternate) {
    return config.writerIdle;
  }
  return (iteration % 2 == 0) == (config.writerAlternate == 1);
}
// Interval between one co-runner's 64 KiB chunk starts, so that all co-runners together meet the target.
static unsigned long long writerIntervalPs(const Config& config) {
  if (!config.targetGbps || !config.writerCtas) {
    return 0;
  }
  return (unsigned long long)(double(coRunnerChunkBytes) * config.writerCtas * 1000.0 / config.targetGbps);
}
static unsigned long long readMismatches(RankState& state) {
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
  unsigned long long mismatches = 0;
  CUDA_CHECK(cudaMemcpy(&mismatches, state.mismatches, sizeof(mismatches), cudaMemcpyDeviceToHost));
  return mismatches;
}
// A10: every SM once in the first half; role counts as in this rank's table; active CTAs have time;
// active co-runners did work. Co-runners of shared copy SMs are in the second half of the records.
static void checkRoles(const std::vector<CtaRecord>& records, const Config& config, int gridCtas, int rank, int localWriterCtas, int writerIdle) {
  std::set<int> usedSms;
  std::set<int> copyIndexes;
  std::set<int> writerIndexes;
  int receivers = 0;
  for (int index = 0; index < 2 * gridCtas; ++index) {
    const CtaRecord& record = records[index];
    bool colocatedHalf = index >= gridCtas;
    if (colocatedHalf && record.role != roleWriter) {
      continue;
    }
    if (!colocatedHalf && !usedSms.insert(record.smId).second) {
      fprintf(stderr, "Two CTAs on SM %d\n", record.smId);
      std::exit(1);
    }
    bool copyActive = record.role == roleCopy && config.communicationEnabled;
    bool timerActive = (record.role == roleCopy && !config.communicationEnabled && record.roleIndex == 0 && config.writerRank == 0) ||
                       (record.role == roleReceiver && (config.communicationEnabled || config.writerRank == 1));
    bool active = record.role == roleWriter || copyActive || timerActive;
    if (active && record.endNs <= record.startNs && !(record.role == roleReceiver)) {
      fprintf(stderr, "Invalid CTA timing (role %d, SM %d)\n", record.role, record.smId);
      std::exit(1);
    }
    receivers += record.role == roleReceiver;
    if (record.role == roleCopy) {
      copyIndexes.insert(record.roleIndex);
    }
    if (record.role == roleWriter) {
      writerIndexes.insert(record.roleIndex);
      if (!writerIdle && record.workUnits == 0) {
        fprintf(stderr, "Co-runner did no work\n");
        std::exit(4);
      }
    }
  }
  if (int(copyIndexes.size()) != (rank == 0 ? config.copyCtas : 0) || receivers != (rank == 1 ? 1 : 0) || int(writerIndexes.size()) != localWriterCtas) {
    fprintf(stderr, "Role counts differ from the role table\n");
    std::exit(1);
  }
}
// A9/A11 for the co-runner: its buffer or checksums on the GPU that runs it.
static void checkCoRunnerWork(RankState& state, const Config& config, int iteration) {
  const int writers = state.localWriterCtas;
  const size_t sliceChunks = writerBufferBytes / coRunnerChunkBytes / std::max(writers, 1);
  unsigned long long mismatches = 0;
  if (config.writerKind == kindTmaRead) {
    std::vector<unsigned long long> chunks(writers);
    std::vector<unsigned long long> checksums(writers);
    CUDA_CHECK(cudaMemcpy(chunks.data(), state.writerChunks, writers * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(checksums.data(), state.readChecksums, writers * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    for (int reader = 0; reader < writers; ++reader) {
      if (chunks[reader] == 0 || checksums[reader] != chunks[reader] * 0x01010101ull) {
        fprintf(stderr, "READ CHECKSUM FAILURE: reader %d chunks=%llu checksum=%llu\n", reader, chunks[reader], checksums[reader]);
        std::exit(4);
      }
    }
    return;
  }
  if (config.writerKind == kindTmaStore) {
    verifyTmaWriterBuffer<<<528, 256, 0, state.stream>>>(reinterpret_cast<const uint32_t*>(state.writerBuffer), writers, sliceChunks, state.writerChunks, tmaWriterPattern(iteration), state.mismatches);
  } else {
    verifyThreadWriterBuffer<<<528, 256, 0, state.stream>>>(reinterpret_cast<const uint4*>(state.writerBuffer), writers, sliceChunks, state.writerChunks, iteration, state.mismatches);
  }
  CUDA_CHECK(cudaGetLastError());
  mismatches = readMismatches(state);
  if (mismatches) {
    fprintf(stderr, "CO-RUNNER CHECK FAILURE: %llu mismatching words or vectors\n", mismatches);
    std::exit(4);
  }
}
// Checks placement, timing, co-runner work and the payload; any failure ends the run with no record.
static void collectIteration(RankState& state, const Config& config, int rank, int iteration) {
  const int writerIdle = launchIsIdle(config, iteration);
  std::vector<CtaRecord> records(2 * state.gridCtas);
  CUDA_CHECK(cudaMemcpy(records.data(), state.deviceRecords, records.size() * sizeof(CtaRecord), cudaMemcpyDeviceToHost));
  checkRoles(records, config, state.gridCtas, rank, state.localWriterCtas, writerIdle);
  if (state.localWriterCtas && !writerIdle) {
    checkCoRunnerWork(state, config, iteration);
  }
  if (rank == 1 && config.communicationEnabled) {
    verifyDestination<<<528, 256, 0, state.stream>>>(static_cast<uint32_t*>(state.windowMemory), config.payloadBytes / sizeof(uint32_t), config.copyChunkBytes / sizeof(uint32_t),
                                                     config.copyGenerate, iteration, state.mismatches);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long mismatches = readMismatches(state);
    if (mismatches) {
      fprintf(stderr, "CORRECTNESS FAILURE: %llu mismatches\n", mismatches);
      std::exit(4);
    }
  }
  if (iteration >= config.warmup) {
    state.records.push_back(std::move(records));
    state.writerIdleByIteration.push_back(writerIdle);
  }
}
static void cleanupRank(RankState& state) {
  NCCL_CHECK(ncclDevCommDestroy(state.communicator, &state.deviceCommunicator));
  NCCL_CHECK(ncclCommWindowDeregister(state.communicator, state.window));
  NCCL_CHECK(ncclMemFree(state.windowMemory));
  CUDA_CHECK(cudaFree(state.writerBuffer));
  CUDA_CHECK(cudaFree(state.writerChunks));
  CUDA_CHECK(cudaFree(state.readChecksums));
  CUDA_CHECK(cudaFree(state.roleOfSm));
  CUDA_CHECK(cudaFree(state.local));
  CUDA_CHECK(cudaFree(state.deviceRecords));
  CUDA_CHECK(cudaFree(state.mismatches));
  CUDA_CHECK(cudaStreamDestroy(state.stream));
  NCCL_CHECK(ncclCommDestroy(state.communicator));
}
static KernelArguments kernelArguments(RankState& state, const Config& config, int rank, int iteration) {
  KernelArguments arguments{};
  arguments.window = state.window;
  arguments.writerBuffer = state.writerBuffer;
  arguments.writerChunks = state.writerChunks;
  arguments.readChecksums = state.readChecksums;
  arguments.roleOfSm = state.roleOfSm;
  arguments.local = state.local;
  arguments.records = state.deviceRecords;
  arguments.config = config;
  arguments.tmaSettings = HbmReadSettings{state.writerBuffer, writerBufferBytes, std::max(state.localWriterCtas, 1), unsigned(coRunnerChunkBytes), config.chunksInFlight, writerIntervalPs(config)};
  arguments.rank = rank;
  arguments.iteration = iteration;
  arguments.writerIdle = launchIsIdle(config, iteration);
  arguments.gridCtas = state.gridCtas;
  arguments.localWriterCtas = state.localWriterCtas;
  arguments.writerIntervalPs = writerIntervalPs(config);
  return arguments;
}
static void runRank(RankState& state, Config config, Placement& placement, int rank, ncclUniqueId identifier, pthread_barrier_t* hostBarrier) {
  setupRank(state, config, placement, rank, identifier);
  pthread_barrier_wait(hostBarrier);
  for (int iteration = 0; iteration < config.warmup + config.iterations; ++iteration) {
    prepareIteration(state, config, rank);
    pthread_barrier_wait(hostBarrier);
    KernelArguments arguments = kernelArguments(state, config, rank, iteration);
    void* launchArguments[] = {&arguments};
    // Cooperative launch: every CTA is resident at once, so waits between roles cannot deadlock.
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(tmaCongest), state.gridCtas, config.blockThreads, launchArguments, state.dynamicSharedBytes, state.stream));
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
    pthread_barrier_wait(hostBarrier);
    collectIteration(state, config, rank, iteration);
    pthread_barrier_wait(hostBarrier);
  }
  cleanupRank(state);
}

// ---------------------------------------------------------------------------------------------
// Host: output.
// ---------------------------------------------------------------------------------------------
static void writeIntegerList(FILE* output, const std::vector<int>& values) {
  fputc('[', output);
  for (size_t index = 0; index < values.size(); ++index) {
    fprintf(output, "%s%d", index ? "," : "", values[index]);
  }
  fputc(']', output);
}
static void writeRecord(FILE* output, const CtaRecord& record, int cta, bool colocated, bool first) {
  fprintf(output, "%s{\"cta\":%d,\"role\":\"%s\",\"role_index\":%d,\"sm_id\":%d,\"colocated\":%d,\"start_ns\":%llu,\"issue_end_ns\":%llu,\"bulk_wait_end_ns\":%llu,"
          "\"fence_end_ns\":%llu,\"publish_end_ns\":%llu,\"end_ns\":%llu,\"work_units\":%llu,\"slot_wait_cycles\":%llu,\"fill_cycles\":%llu,\"issue_loop_cycles\":%llu,\"checkpoints_ns\":[",
          first ? "" : ",", cta, roleNames[record.role], record.roleIndex, record.smId, colocated ? 1 : 0, record.startNs, record.issueEndNs, record.bulkWaitEndNs,
          record.fenceEndNs, record.publishEndNs, record.endNs, record.workUnits, record.slotWaitCycles, record.fillCycles, record.issueLoopCycles);
  for (int checkpoint = 0; checkpoint < checkpointCount; ++checkpoint) {
    fprintf(output, "%s%llu", checkpoint ? "," : "", record.checkpointsNs[checkpoint]);
  }
  fprintf(output, "]}");
}
static void writeJson(const char* path, const char* configPath, const Config& config, const std::array<Placement, 2>& placements, const std::array<RankState, 2>& states) {
  FILE* output = fopen(path, "w");
  if (!output) {
    perror(path);
    std::exit(1);
  }
  // Output keys are stable and consumed by run_tma_congest.py and the analysis scripts; renaming a
  // C++ field never renames a key.
  fprintf(output, "{\"schema_version\":1,\"harness\":\"tma_congest\",\"config_path\":\"%s\",\"source_gpu\":%d,\"destination_gpu\":%d,"
          "\"copy_ctas\":%d,\"copy_threads\":%d,\"copy_chunk_bytes\":%d,\"copy_chunks_in_flight\":%d,\"copy_generate\":%d,\"copy_gpcs\":%d,\"copy_placement\":%d,"
          "\"writer_kind\":%d,\"writer_ctas\":%d,\"writer_threads\":%d,\"chunks_in_flight\":%d,\"writer_gpcs\":%d,\"writer_placement\":%d,\"writer_colocated\":%d,"
          "\"writer_rank\":%d,\"receiver_gpcs\":%d,\"target_gbps\":%d,\"writer_idle\":%d,\"writer_alternate\":%d,"
          "\"use_unmapped\":%d,\"block_threads\":%d,\"detailed\":%d,\"communication\":%d,\"solo_ns\":%llu,\"bytes\":%zu,\"writer_block_bytes\":%d,\"writer_buffer_bytes\":%zu,\"verified\":true",
          configPath, config.sourceGpu, config.destinationGpu,
          config.copyCtas, config.copyThreads, config.copyChunkBytes, config.copyChunksInFlight, config.copyGenerate, config.copyGpcMask, config.copyPlacement,
          config.writerKind, config.writerCtas, config.writerThreads, config.chunksInFlight, config.writerGpcMask, config.writerPlacement, config.writerColocated,
          config.writerRank, config.receiverGpcMask, config.targetGbps, config.writerIdle, config.writerAlternate,
          config.useUnmapped, config.blockThreads, config.detailed, config.communicationEnabled, config.soloDurationNs, config.payloadBytes, coRunnerChunkBytes, writerBufferBytes);
  fprintf(output, ",\"gpc_map\":\"%s\",\"gpc_map_uuid\":\"%s\",\"gpc_of_sm\":", placements[0].mapPath.c_str(), placements[0].mapUuid.c_str());
  writeIntegerList(output, placements[0].gpcOfSm);
  fprintf(output, ",\"role_of_sm\":");
  writeIntegerList(output, placements[0].roleOfSm);
  fprintf(output, ",\"destination_gpc_map\":\"%s\",\"destination_gpc_of_sm\":", placements[1].mapPath.c_str());
  writeIntegerList(output, placements[1].gpcOfSm);
  fprintf(output, ",\"destination_role_of_sm\":");
  writeIntegerList(output, placements[1].roleOfSm);
  fprintf(output, ",\"ranks\":[");
  for (int rank = 0; rank < 2; ++rank) {
    const auto& state = states[rank];
    fprintf(output, "%s{\"rank\":%d,\"sm_count\":%d,\"shared_bytes\":%zu,\"writer_idle_by_iteration\":", rank ? "," : "", rank, state.properties.multiProcessorCount, state.dynamicSharedBytes);
    writeIntegerList(output, state.writerIdleByIteration);
    fprintf(output, ",\"iterations\":[");
    for (size_t iteration = 0; iteration < state.records.size(); ++iteration) {
      fprintf(output, "%s[", iteration ? "," : "");
      const auto& records = state.records[iteration];
      bool first = true;
      // Idle CTAs only prove the SM was occupied; they are listed with their SM for A10.
      for (int cta = 0; cta < state.gridCtas; ++cta) {
        writeRecord(output, records[cta], cta, false, first);
        first = false;
      }
      for (int cta = 0; cta < state.gridCtas; ++cta) {
        if (records[state.gridCtas + cta].role == roleWriter) {
          writeRecord(output, records[state.gridCtas + cta], cta, true, false);
        }
      }
      fprintf(output, "]");
    }
    fprintf(output, "]}");
  }
  fprintf(output, "]}\n");
  fclose(output);
}

// Usage: tma_congest CONFIG OUTPUT.json --gpus SOURCE,DESTINATION --gpc-map SOURCE_MAP.json --destination-gpc-map DESTINATION_MAP.json
// Configs carry no GPU numbers; the pair and both GPUs' maps are given per invocation.
int main(int argumentCount, char** arguments) {
  if (argumentCount < 3) {
    fprintf(stderr, "Usage: tma_congest CONFIG OUTPUT.json --gpus S,D --gpc-map MAP.json --destination-gpc-map MAP.json\n");
    return 1;
  }
  Config config = parseConfiguration(arguments[1]);
  std::array<Placement, 2> placements;
  for (int index = 3; index < argumentCount; ++index) {
    std::string option = arguments[index];
    if (option == "--gpus" && index + 1 < argumentCount && sscanf(arguments[index + 1], "%d,%d", &config.sourceGpu, &config.destinationGpu) == 2) {
      ++index;
    } else if (option == "--gpc-map" && index + 1 < argumentCount) {
      placements[0].mapPath = arguments[++index];
    } else if (option == "--destination-gpc-map" && index + 1 < argumentCount) {
      placements[1].mapPath = arguments[++index];
    } else {
      fprintf(stderr, "Unknown option %s\n", arguments[index]);
      return 1;
    }
  }
  validateConfiguration(config);
  if (!authorizedGpu(config.sourceGpu) || !authorizedGpu(config.destinationGpu) || config.sourceGpu == config.destinationGpu ||
      placements[0].mapPath.empty() || placements[1].mapPath.empty()) {
    fprintf(stderr, "Need --gpus with a pair inside the authorized allocation, --gpc-map and --destination-gpc-map\n");
    return 1;
  }
  ncclUniqueId identifier;
  NCCL_CHECK(ncclGetUniqueId(&identifier));
  pthread_barrier_t hostBarrier;
  pthread_barrier_init(&hostBarrier, nullptr, 2);
  std::array<RankState, 2> states;
  std::thread sourceThread(runRank, std::ref(states[0]), config, std::ref(placements[0]), 0, identifier, &hostBarrier);
  std::thread destinationThread(runRank, std::ref(states[1]), config, std::ref(placements[1]), 1, identifier, &hostBarrier);
  sourceThread.join();
  destinationThread.join();
  pthread_barrier_destroy(&hostBarrier);
  writeJson(arguments[2], arguments[1], config, placements, states);
  return 0;
}
