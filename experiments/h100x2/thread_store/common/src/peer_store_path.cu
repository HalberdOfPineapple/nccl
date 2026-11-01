// Plan 08 harness: does a peer store over NVLink pass through the destination GPU's L2?
//
// One transfer is a fixed sequence of kernel launches on the destination GPU. Every phase is a
// separately named kernel, because ncu reports hardware counters per launch: the kernel name in its
// table says which phase a row belongs to.
//   phaseClean    reads a clean buffer several times larger than L2, to push out earlier data
//   phaseQuiet    does nothing for a fixed time; HBM writes here are L2 writing back by itself
//   phasePrepare  sets the L2 state under test (E8.4, E8.5)
//   phaseMain     the measured stores: peer stores from the source GPU, or local stores
//   phaseChase    one thread follows a linked list written by the stores (E8.3)
//   phaseReadback reads and compares every word right after the stores (E8.4)
//   phaseEvict    reads the clean buffer again; HBM writes here are the dirty data L2 was holding
//   phaseCheck    compares every word with the last value stored
// The stored values change on every pass, so nothing can skip an unchanged store. The copy loop is
// plan 07's store-only copy (4-byte stores, CTA-local stride, parts on 128-byte boundaries).
// JSON keys are the experiment data interface; renaming C++ fields must not rename keys.
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <nccl.h>
#include <nccl_device.h>
#include <pthread.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

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

constexpr int maxCtas = 132;
constexpr int blockThreads = 1024;
constexpr size_t wordsPerLine = 32;                  // 128-byte line
constexpr size_t mebibyte = size_t(1) << 20;
constexpr unsigned long long noWrongOffset = ~0ull;
enum StoreSource { storeNone = 0, storePeer = 1, storeLocal = 2 };
enum StoreShape { shapeWords = 0, shapeVectors = 1, shapeOneWordPerLine = 2, shapeLinkedList = 3 };
enum PrepareKind { prepareNone = 0, prepareRead = 1, prepareLocalStore = 2 };
enum ChaseKind { chaseNone = 0, chaseAfterStores = 1, chaseAfterEvict = 2 };

struct alignas(128) Signal { int value; };
struct WindowControl {
  Signal destinationReady;
  Signal acknowledgement;
  Signal completion[maxCtas];
};
struct LocalControl {
  int residentReady;      // resident reader CTAs that have started (E8.5)
  int finishedWriters;    // local writer CTAs that have finished
  int storesDone;         // releases the resident readers and the eviction readers in the same launch
};
struct CtaRecord {
  unsigned long long startNs;       // A2
  unsigned long long issueEndNs;    // A3
  unsigned long long fenceEndNs;    // A4
  unsigned long long publishEndNs;  // A5
  unsigned long long endNs;         // A1 / A6
  unsigned long long workUnits;     // A9: resident reader passes
  int smId;                         // A10
  int roleCode;                     // A10: 0 store, 1 receiver or timer, 2 resident reader, 3 eviction reader
};
struct ProbeRecord {
  unsigned long long chaseNs;              // A13: whole pass, destination timer
  unsigned long long chaseHops;            // A13
  unsigned long long chaseWrong;           // A14: hops whose next index was not the expected one
  unsigned long long readbackWrong;        // A14: words read right after the stores
  unsigned long long checkWrong;           // A11: words at the end of the transfer
  unsigned long long firstWrongOffset;     // A14: smallest word offset found wrong
  unsigned long long wrongValueSample;     // A14: a value found at a wrong word
};
struct Config {
  int sourceGpu = -1;
  int destinationGpu = -1;
  int profiledRank = 1;           // this rank launches second, so a blocking profiler cannot stop the other
  int storeSource = storePeer;
  int storeShape = shapeWords;
  int writerCtas = 16;            // peer or local writer CTAs
  int writerThreads = 512;        // active threads per writer CTA
  int footprintMib = 2048;
  int passes = 1;
  int fenceEachPass = 0;          // E8.2c: a system fence by every writing thread after each pass
  int evictionMib = 256;
  int evictionPasses = 1;
  int readerCtas = 64;            // clean, evict and check readers
  int prepareKind = prepareNone;
  int chaseKind = chaseNone;
  int readback = 0;               // E8.4
  int quietAfterStores = 1;       // 0 skips phaseQuiet after the stores, so a read comes at once
  int evictInMainLaunch = 0;      // E8.0: phaseEvict's reads run at the end of phaseMain
  int residentCtas = 0;           // E8.5: resident reader CTAs on the destination
  int residentMib = 16;
  int warmup = 2;
  int iterations = 3;
  int detailed = 1;
  unsigned long long quietNs = 5000000;
  unsigned long long noStoreWindowNs = 23000000;   // length of phaseMain when nothing stores
  unsigned long long chaseWaitNs = 0;              // E8.3: wait before the first hop
};
struct KernelArguments {
  ncclWindow_t window;
  uint4* evictionBuffer;
  uint4* residentBuffer;
  LocalControl* local;
  CtaRecord* records;
  ProbeRecord* probe;
  Config config;
  int rank;
  int transfer;
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
// Keeps a computed value without any global memory traffic, so the reads behind it stay in the
// program. An empty inline-asm input did not keep them: the compiler removed every eviction read,
// and phaseEvict ran for 2 microseconds (first counter trial, 2026-10-31).
__device__ __forceinline__ void keepValue(uint32_t value) {
  extern __shared__ __align__(128) uint32_t dynamicShared[];
  dynamicShared[1 + threadIdx.x] = value;
}
__device__ __forceinline__ WindowControl* windowControl(const KernelArguments& arguments, int rank) {
  return static_cast<WindowControl*>(ncclGetLsaPointer(arguments.window, size_t(arguments.config.footprintMib) * mebibyte, rank));
}
// The destination's payload, through the same NCCL LSA call on both ranks.
__device__ __forceinline__ uint32_t* payloadPointer(const KernelArguments& arguments) {
  return static_cast<uint32_t*>(ncclGetLsaPointer(arguments.window, 0, 1));
}

__host__ __device__ __forceinline__ uint32_t passStamp(int transfer, int pass) {
  return uint32_t(transfer * 65536 + pass + 1) * 1013904223u;
}
__host__ __device__ __forceinline__ uint32_t wordValue(size_t offset, uint32_t stamp) {
  return uint32_t(offset) * 1664525u ^ stamp;
}
// The list visits every line once: a full-period generator modulo the line count (a power of two),
// with a multiplier of 1 modulo 4 and an odd increment. The increment changes per transfer, so a
// reader that finds the previous transfer's list reports wrong hops.
__host__ __device__ __forceinline__ uint32_t nextLine(uint32_t line, uint32_t lineCount, int transfer) {
  const uint32_t increment = uint32_t(2 * transfer + 1) * 0x2545F491u;
  return (line * 0x9E3779B1u + increment) & (lineCount - 1);
}

// One pass per thread over its stride. Only a thread's own later passes overwrite a word, so the
// last value of a word is fixed, however the warps drift apart. The compiler barrier keeps a later
// pass from replacing an earlier one.
__device__ void storePasses(const KernelArguments& arguments, uint32_t* payload, int ctaIndex) {
  const Config& config = arguments.config;
  const size_t elementCount = size_t(config.footprintMib) * mebibyte / sizeof(uint32_t);
  const size_t regionBegin = elementCount * ctaIndex / config.writerCtas / wordsPerLine * wordsPerLine;
  const size_t regionEnd = elementCount * (ctaIndex + 1) / config.writerCtas / wordsPerLine * wordsPerLine;
  if (threadIdx.x >= config.writerThreads) {
    return;
  }
  const size_t threads = config.writerThreads;
  for (int pass = 0; pass < config.passes; ++pass) {
    const uint32_t stamp = passStamp(arguments.transfer, pass);
    if (config.storeShape == shapeWords) {
      #pragma unroll 1
      for (size_t offset = regionBegin + threadIdx.x; offset < regionEnd; offset += threads) {
        payload[offset] = wordValue(offset, stamp);
      }
    } else if (config.storeShape == shapeVectors) {
      uint4* vectors = reinterpret_cast<uint4*>(payload);
      #pragma unroll 1
      for (size_t vector = regionBegin / 4 + threadIdx.x; vector < regionEnd / 4; vector += threads) {
        const size_t word = vector * 4;
        vectors[vector] = make_uint4(wordValue(word, stamp), wordValue(word + 1, stamp), wordValue(word + 2, stamp), wordValue(word + 3, stamp));
      }
    } else if (config.storeShape == shapeOneWordPerLine) {
      #pragma unroll 1
      for (size_t offset = regionBegin + threadIdx.x * wordsPerLine; offset < regionEnd; offset += threads * wordsPerLine) {
        payload[offset] = wordValue(offset, stamp);
      }
    } else {
      // Every byte of a line is written: a partly written line could force an HBM read and look like a miss.
      const uint32_t lineCount = uint32_t(elementCount / wordsPerLine);
      #pragma unroll 1
      for (size_t offset = regionBegin + threadIdx.x; offset < regionEnd; offset += threads) {
        payload[offset] = offset % wordsPerLine == 0 ? nextLine(uint32_t(offset / wordsPerLine), lineCount, arguments.transfer) : wordValue(offset, stamp);
      }
    }
    asm volatile("" ::: "memory");
    if (config.fenceEachPass) {
      __threadfence_system();
    }
  }
}

// Reads a range with 16-byte loads, one CTA's share per call. Used to push L2 contents out, and to
// load lines into L2. Reads add no store, so HBM writes during them come only from evicted dirty data.
__device__ void readRange(const uint4* range, size_t vectorCount, int ctaIndex, int ctaCount, int passes) {
  const size_t stride = size_t(blockThreads) * ctaCount;
  uint32_t combined = 0;
  for (int pass = 0; pass < passes; ++pass) {
    #pragma unroll 1
    for (size_t vector = size_t(ctaIndex) * blockThreads + threadIdx.x; vector < vectorCount; vector += stride) {
      const uint4 value = range[vector];
      combined ^= value.x ^ value.y ^ value.z ^ value.w;
    }
  }
  keepValue(combined);
}

__host__ __device__ __forceinline__ bool shapeWritesWord(size_t offset, int storeShape) {
  return storeShape != shapeOneWordPerLine || offset % wordsPerLine == 0;
}
__device__ __forceinline__ uint32_t expectedWord(const KernelArguments& arguments, size_t offset, size_t elementCount) {
  const Config& config = arguments.config;
  const uint32_t stamp = passStamp(arguments.transfer, config.passes - 1);
  if (!shapeWritesWord(offset, config.storeShape)) {
    return 0xffffffffu;   // poison, left alone by these stores
  }
  if (config.storeShape == shapeLinkedList && offset % wordsPerLine == 0) {
    return nextLine(uint32_t(offset / wordsPerLine), uint32_t(elementCount / wordsPerLine), arguments.transfer);
  }
  return wordValue(offset, stamp);
}
// Compares every word; wrong words are counted and the lowest offset is kept. The caller's fresh
// launch guarantees an empty L1, so the words come from L2 or below.
__device__ void compareWords(const KernelArguments& arguments, unsigned long long* wrongCount) {
  const size_t elementCount = size_t(arguments.config.footprintMib) * mebibyte / sizeof(uint32_t);
  const uint32_t* payload = payloadPointer(arguments);
  const size_t stride = size_t(blockDim.x) * gridDim.x;
  unsigned long long wrongWords = 0;
  for (size_t offset = size_t(blockIdx.x) * blockDim.x + threadIdx.x; offset < elementCount; offset += stride) {
    const uint32_t value = payload[offset];
    if (value != expectedWord(arguments, offset, elementCount)) {
      ++wrongWords;
      atomicMin(&arguments.probe->firstWrongOffset, (unsigned long long)offset);
      arguments.probe->wrongValueSample = value;
    }
  }
  if (wrongWords) {
    atomicAdd(wrongCount, wrongWords);
  }
}

// Same protocol and timers as store_congest's runCopy: all writers fence, each CTA publishes "done",
// the receiver acknowledges.
__device__ void runPeerStores(const KernelArguments& arguments, int ctaIndex) {
  WindowControl* localSignals = windowControl(arguments, 0);
  WindowControl* remoteSignals = windowControl(arguments, 1);
  CtaRecord& record = arguments.records[ctaIndex];
  if (threadIdx.x == 0) {
    record.smId = readSmId();
    record.roleCode = 0;
    while (!acquireSignal(&localSignals->destinationReady.value)) {
      __nanosleep(64);
    }
    record.startNs = readTimerNs();
  }
  __syncthreads();
  storePasses(arguments, payloadPointer(arguments), ctaIndex);
  __syncthreads();
  if (arguments.config.detailed && threadIdx.x == 0) {
    record.issueEndNs = readTimerNs();
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    if (arguments.config.detailed) {
      record.fenceEndNs = readTimerNs();
    }
    releaseSignal(&remoteSignals->completion[ctaIndex].value, 1);
    if (arguments.config.detailed) {
      record.publishEndNs = readTimerNs();
    }
    while (!acquireSignal(&localSignals->acknowledgement.value)) {
      __nanosleep(64);
    }
    record.endNs = readTimerNs();
  }
}
__device__ void waitForResidentReaders(const KernelArguments& arguments) {
  while (acquireSignal(&arguments.local->residentReady) != arguments.config.residentCtas) {
    __nanosleep(64);
  }
}
__device__ void runReceiver(const KernelArguments& arguments) {
  if (threadIdx.x != 0) {
    return;
  }
  WindowControl* localSignals = windowControl(arguments, 1);
  WindowControl* remoteSignals = windowControl(arguments, 0);
  CtaRecord& record = arguments.records[0];
  record.smId = readSmId();
  record.roleCode = 1;
  record.startNs = readTimerNs();
  waitForResidentReaders(arguments);
  releaseSignal(&remoteSignals->destinationReady.value, 1);
  for (int ctaIndex = 0; ctaIndex < arguments.config.writerCtas; ++ctaIndex) {
    while (!acquireSignal(&localSignals->completion[ctaIndex].value)) {
      __nanosleep(64);
    }
  }
  record.issueEndNs = readTimerNs();
  releaseSignal(&remoteSignals->acknowledgement.value, 1);
  record.endNs = readTimerNs();
  releaseSignal(&arguments.local->storesDone, 1);
}
__device__ void runLocalStores(const KernelArguments& arguments, int ctaIndex) {
  CtaRecord& record = arguments.records[ctaIndex];
  if (threadIdx.x == 0) {
    record.smId = readSmId();
    record.roleCode = 0;
    waitForResidentReaders(arguments);
    record.startNs = readTimerNs();
  }
  __syncthreads();
  storePasses(arguments, payloadPointer(arguments), ctaIndex);
  __syncthreads();
  if (arguments.config.detailed && threadIdx.x == 0) {
    record.issueEndNs = readTimerNs();
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    record.endNs = readTimerNs();
    if (arguments.config.detailed) {
      record.fenceEndNs = record.endNs;
    }
    if (atomicAdd(&arguments.local->finishedWriters, 1) + 1 == arguments.config.writerCtas) {
      releaseSignal(&arguments.local->storesDone, 1);
    }
  }
}
__device__ void runNoStoreTimer(const KernelArguments& arguments) {
  if (threadIdx.x != 0) {
    return;
  }
  CtaRecord& record = arguments.records[0];
  record.smId = readSmId();
  record.roleCode = 1;
  waitForResidentReaders(arguments);
  record.startNs = readTimerNs();
  while (readTimerNs() - record.startNs < arguments.config.noStoreWindowNs) {
    __nanosleep(1024);
  }
  record.endNs = readTimerNs();
  releaseSignal(&arguments.local->storesDone, 1);
}
// E8.5: repeatedly reads this CTA's slice until the stores are done. Lane 0 checks the stop flag and
// passes it on through shared memory, so the flag check does not add a memory load per lane.
__device__ void runResidentReader(const KernelArguments& arguments, int residentIndex, int recordIndex) {
  extern __shared__ __align__(128) uint32_t dynamicShared[];
  CtaRecord& record = arguments.records[recordIndex];
  const size_t bufferVectors = size_t(arguments.config.residentMib) * mebibyte / sizeof(uint4);
  if (threadIdx.x == 0) {
    dynamicShared[0] = 0;
    record.smId = readSmId();
    record.roleCode = 2;
    record.startNs = readTimerNs();
    cuda::atomic_ref<int, cuda::thread_scope_system>(arguments.local->residentReady).fetch_add(1, cuda::memory_order_release);
  }
  __syncthreads();
  unsigned long long passesRead = 0;
  for (;;) {
    readRange(arguments.residentBuffer, bufferVectors, residentIndex, arguments.config.residentCtas, 1);
    if (threadIdx.x == 0) {
      dynamicShared[0] = *reinterpret_cast<volatile int*>(&arguments.local->storesDone);
    }
    __syncthreads();
    ++passesRead;
    if (dynamicShared[0]) {
      break;
    }
  }
  if (threadIdx.x == 0) {
    record.endNs = readTimerNs();
    record.workUnits = passesRead;
  }
}
// E8.0 launch-boundary test: the eviction reads run after the stores, inside the same launch.
__device__ void runEvictionReader(const KernelArguments& arguments, int readerIndex, int recordIndex) {
  CtaRecord& record = arguments.records[recordIndex];
  if (threadIdx.x == 0) {
    record.smId = readSmId();
    record.roleCode = 3;
    while (!acquireSignal(&arguments.local->storesDone)) {
      __nanosleep(256);
    }
    record.startNs = readTimerNs();
  }
  __syncthreads();
  readRange(arguments.evictionBuffer, size_t(arguments.config.evictionMib) * mebibyte / sizeof(uint4), readerIndex, arguments.config.readerCtas, arguments.config.evictionPasses);
  __syncthreads();
  if (threadIdx.x == 0) {
    record.endNs = readTimerNs();
  }
}

// CTA order on the destination: store role (receiver, local writers or timer), then resident readers,
// then eviction readers. Each CTA has its own SM, so all roles fit at once.
__host__ __device__ __forceinline__ int destinationStoreRoleCtas(const Config& config) {
  return config.storeSource == storeLocal ? config.writerCtas : 1;
}
__global__ void __launch_bounds__(blockThreads, 1) phaseMain(KernelArguments arguments) {
  const Config& config = arguments.config;
  const int ctaIndex = blockIdx.x;
  if (arguments.rank == 0) {
    runPeerStores(arguments, ctaIndex);
    return;
  }
  const int storeRoleCtas = destinationStoreRoleCtas(config);
  if (ctaIndex >= storeRoleCtas + config.residentCtas) {
    runEvictionReader(arguments, ctaIndex - storeRoleCtas - config.residentCtas, ctaIndex);
  } else if (ctaIndex >= storeRoleCtas) {
    runResidentReader(arguments, ctaIndex - storeRoleCtas, ctaIndex);
  } else if (config.storeSource == storePeer) {
    runReceiver(arguments);
  } else if (config.storeSource == storeLocal) {
    runLocalStores(arguments, ctaIndex);
  } else {
    runNoStoreTimer(arguments);
  }
}
__global__ void __launch_bounds__(blockThreads, 1) phaseClean(KernelArguments arguments) {
  readRange(arguments.evictionBuffer, size_t(arguments.config.evictionMib) * mebibyte / sizeof(uint4), blockIdx.x, gridDim.x, arguments.config.evictionPasses);
}
__global__ void __launch_bounds__(blockThreads, 1) phaseEvict(KernelArguments arguments) {
  readRange(arguments.evictionBuffer, size_t(arguments.config.evictionMib) * mebibyte / sizeof(uint4), blockIdx.x, gridDim.x, arguments.config.evictionPasses);
}
__global__ void __launch_bounds__(blockThreads, 1) phaseQuiet(KernelArguments arguments) {
  if (threadIdx.x == 0) {
    const unsigned long long startNs = readTimerNs();
    while (readTimerNs() - startNs < arguments.config.quietNs) {
      __nanosleep(1024);
    }
  }
}
// Sets the L2 state of the footprint (E8.4) and loads the resident buffer (E8.5).
__global__ void __launch_bounds__(blockThreads, 1) phasePrepare(KernelArguments arguments) {
  const Config& config = arguments.config;
  if (config.prepareKind == prepareRead) {
    readRange(reinterpret_cast<const uint4*>(payloadPointer(arguments)), size_t(config.footprintMib) * mebibyte / sizeof(uint4), blockIdx.x, gridDim.x, 2);
  } else if (config.prepareKind == prepareLocalStore) {
    KernelArguments otherValues = arguments;
    otherValues.transfer = arguments.transfer + 16384;
    otherValues.config.passes = 1;
    otherValues.config.writerCtas = gridDim.x;
    otherValues.config.storeShape = shapeWords;
    storePasses(otherValues, payloadPointer(arguments), blockIdx.x);
  }
  if (config.residentCtas) {
    readRange(arguments.residentBuffer, size_t(config.residentMib) * mebibyte / sizeof(uint4), blockIdx.x, gridDim.x, 2);
  }
}
// A13/A14: one thread follows the list once. Each load depends on the last, so loads cannot overlap.
__global__ void __launch_bounds__(blockThreads, 1) phaseChase(KernelArguments arguments) {
  if (threadIdx.x != 0) {
    return;
  }
  const size_t elementCount = size_t(arguments.config.footprintMib) * mebibyte / sizeof(uint32_t);
  const uint32_t lineCount = uint32_t(elementCount / wordsPerLine);
  const volatile uint32_t* payload = payloadPointer(arguments);
  const unsigned long long waitStartNs = readTimerNs();
  while (readTimerNs() - waitStartNs < arguments.config.chaseWaitNs) {
    __nanosleep(1024);
  }
  uint32_t line = 0;
  unsigned long long wrongHops = 0;
  const unsigned long long startNs = readTimerNs();
  for (uint32_t hop = 0; hop < lineCount; ++hop) {
    const uint32_t next = payload[size_t(line) * wordsPerLine];
    if (next != nextLine(line, lineCount, arguments.transfer)) {
      ++wrongHops;
    }
    // A wrong index stays in range, so the pass always ends and its time stays comparable.
    line = next & (lineCount - 1);
  }
  const unsigned long long endNs = readTimerNs();
  arguments.probe->chaseNs = endNs - startNs;
  arguments.probe->chaseHops = lineCount;
  arguments.probe->chaseWrong = wrongHops;
}
__global__ void __launch_bounds__(blockThreads, 1) phaseReadback(KernelArguments arguments) {
  compareWords(arguments, &arguments.probe->readbackWrong);
}
__global__ void __launch_bounds__(blockThreads, 1) phaseCheck(KernelArguments arguments) {
  compareWords(arguments, &arguments.probe->checkWrong);
}

struct RankState {
  ncclComm_t communicator;
  ncclDevComm deviceCommunicator;
  ncclWindow_t window;
  void* windowMemory;
  uint4* evictionBuffer;
  uint4* residentBuffer;
  LocalControl* local;
  CtaRecord* deviceRecords;
  ProbeRecord* probe;
  cudaStream_t stream;
  int mainGridCtas;
  size_t dynamicSharedBytes;
  cudaDeviceProp properties;
  std::vector<std::vector<CtaRecord>> records;
  std::vector<ProbeRecord> probes;
};
struct HostSync {
  pthread_barrier_t readyForMain;
  pthread_barrier_t firstLaunchIssued;
  pthread_barrier_t transferFinished;
};

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
    const auto separator = line.find('=');
    const std::string key = line.substr(0, separator);
    const long long value = std::stoll(line.substr(separator + 1));
    if (key == "store_source") {
      config.storeSource = value;
    } else if (key == "store_shape") {
      config.storeShape = value;
    } else if (key == "writer_ctas") {
      config.writerCtas = value;
    } else if (key == "writer_threads") {
      config.writerThreads = value;
    } else if (key == "footprint_mib") {
      config.footprintMib = value;
    } else if (key == "passes") {
      config.passes = value;
    } else if (key == "fence_each_pass") {
      config.fenceEachPass = value;
    } else if (key == "eviction_mib") {
      config.evictionMib = value;
    } else if (key == "eviction_passes") {
      config.evictionPasses = value;
    } else if (key == "reader_ctas") {
      config.readerCtas = value;
    } else if (key == "prepare_kind") {
      config.prepareKind = value;
    } else if (key == "chase_kind") {
      config.chaseKind = value;
    } else if (key == "readback") {
      config.readback = value;
    } else if (key == "quiet_after_stores") {
      config.quietAfterStores = value;
    } else if (key == "evict_in_main_launch") {
      config.evictInMainLaunch = value;
    } else if (key == "resident_ctas") {
      config.residentCtas = value;
    } else if (key == "resident_mib") {
      config.residentMib = value;
    } else if (key == "warmup") {
      config.warmup = value;
    } else if (key == "iterations") {
      config.iterations = value;
    } else if (key == "detailed") {
      config.detailed = value;
    } else if (key == "quiet_ns") {
      config.quietNs = value;
    } else if (key == "no_store_window_ns") {
      config.noStoreWindowNs = value;
    } else if (key == "chase_wait_ns") {
      config.chaseWaitNs = value;
    } else {
      fprintf(stderr, "Unknown config key %s\n", key.c_str());
      std::exit(1);
    }
  }
  return config;
}
static void validateConfiguration(const Config& config) {
  const size_t lineCount = size_t(config.footprintMib) * mebibyte / (wordsPerLine * sizeof(uint32_t));
  const bool listValid = config.storeShape != shapeLinkedList || (lineCount & (lineCount - 1)) == 0;
  const bool chaseValid = config.chaseKind == chaseNone || config.storeShape == shapeLinkedList;
  const bool storesValid = config.writerCtas >= 1 && config.writerCtas <= maxCtas && config.writerThreads >= 32 && config.writerThreads % 32 == 0 && config.writerThreads <= blockThreads
                           && config.footprintMib >= 1 && config.footprintMib <= 4096 && config.passes >= 1;
  const bool rolesValid = destinationStoreRoleCtas(config) + config.residentCtas + (config.evictInMainLaunch ? config.readerCtas : 0) <= maxCtas;
  if (!listValid || !chaseValid || !storesValid || !rolesValid || (config.residentCtas && (config.storeSource == storeLocal))) {
    fprintf(stderr, "Invalid configuration (list %d, chase %d, stores %d, roles %d)\n", listValid, chaseValid, storesValid, rolesValid);
    std::exit(1);
  }
}

template <typename Kernel>
static void prepareKernel(Kernel kernel, const RankState& state) {
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, state.dynamicSharedBytes));
  int residentCtas = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residentCtas, kernel, blockThreads, state.dynamicSharedBytes));
  if (residentCtas != 1) {
    fprintf(stderr, "Expected exactly one resident CTA per SM, got %d\n", residentCtas);
    std::exit(1);
  }
  // Loads the kernel now, so a later launch does not wait for lazy loading during measurement.
  cudaFuncAttributes attributes;
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
}
static void setupRank(RankState& state, const Config& config, int rank, ncclUniqueId identifier) {
  const int device = rank == 0 ? config.sourceGpu : config.destinationGpu;
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaGetDeviceProperties(&state.properties, device));
  NCCL_CHECK(ncclCommInitRank(&state.communicator, 2, identifier, rank));
  ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_CHECK(ncclCommQueryProperties(state.communicator, &properties));
  if (!properties.deviceApiSupport || properties.nLsaTeams != 1) {
    fprintf(stderr, "Requires a single two-GPU LSA team\n");
    std::exit(1);
  }
  const size_t windowBytes = size_t(config.footprintMib) * mebibyte + sizeof(WindowControl);
  NCCL_CHECK(ncclMemAlloc(&state.windowMemory, windowBytes));
  NCCL_CHECK(ncclCommWindowRegister(state.communicator, state.windowMemory, windowBytes, &state.window, NCCL_WIN_COLL_SYMMETRIC));
  ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  NCCL_CHECK(ncclDevCommCreate(state.communicator, &requirements, &state.deviceCommunicator));
  CUDA_CHECK(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&state.local, sizeof(LocalControl)));
  state.mainGridCtas = rank == 0 ? config.writerCtas : destinationStoreRoleCtas(config) + config.residentCtas + (config.evictInMainLaunch ? config.readerCtas : 0);
  CUDA_CHECK(cudaMalloc(&state.deviceRecords, maxCtas * sizeof(CtaRecord)));
  CUDA_CHECK(cudaMalloc(&state.probe, sizeof(ProbeRecord)));
  state.evictionBuffer = nullptr;
  state.residentBuffer = nullptr;
  if (rank == 1) {
    CUDA_CHECK(cudaMalloc(&state.evictionBuffer, size_t(config.evictionMib) * mebibyte));
    CUDA_CHECK(cudaMemset(state.evictionBuffer, 0x5a, size_t(config.evictionMib) * mebibyte));
    if (config.residentCtas) {
      CUDA_CHECK(cudaMalloc(&state.residentBuffer, size_t(config.residentMib) * mebibyte));
      CUDA_CHECK(cudaMemset(state.residentBuffer, 0x3c, size_t(config.residentMib) * mebibyte));
    }
  }
  // More than half an SM's shared memory ensures no two CTAs share an SM.
  state.dynamicSharedBytes = ((state.properties.sharedMemPerMultiprocessor / 2 + 255) / 256) * 256;
  prepareKernel(phaseMain, state);
  prepareKernel(phaseClean, state);
  prepareKernel(phaseEvict, state);
  prepareKernel(phaseQuiet, state);
  prepareKernel(phasePrepare, state);
  prepareKernel(phaseChase, state);
  prepareKernel(phaseReadback, state);
  prepareKernel(phaseCheck, state);
}
// The poison write uses the copy engine, outside every measured launch; phaseClean follows it.
static void prepareTransfer(RankState& state, const Config& config, int rank) {
  const size_t payloadBytes = size_t(config.footprintMib) * mebibyte;
  CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(state.windowMemory) + payloadBytes, 0, sizeof(WindowControl), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.local, 0, sizeof(LocalControl), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.deviceRecords, 0, maxCtas * sizeof(CtaRecord), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.probe, 0, sizeof(ProbeRecord), state.stream));
  CUDA_CHECK(cudaMemsetAsync(&state.probe->firstWrongOffset, 0xff, sizeof(state.probe->firstWrongOffset), state.stream));
  if (rank == 1) {
    CUDA_CHECK(cudaMemsetAsync(state.windowMemory, 0xff, payloadBytes, state.stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
template <typename Kernel>
static void launchPhase(RankState& state, Kernel kernel, int gridCtas, KernelArguments& arguments) {
  void* kernelArguments[] = {&arguments};
  // Cooperative launch: every CTA is resident at once, so waits between roles cannot deadlock.
  CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), gridCtas, blockThreads, kernelArguments, state.dynamicSharedBytes, state.stream));
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
// phaseMain on both GPUs. The rank that is not profiled launches first; its kernel waits on a signal
// from the other GPU, so a profiler that blocks the second launch cannot hold up the first.
static void launchMain(RankState& state, const Config& config, int rank, KernelArguments& arguments, HostSync& sync) {
  const bool sourceStores = config.storeSource == storePeer;
  pthread_barrier_wait(&sync.readyForMain);
  const bool launchesFirst = rank != config.profiledRank;
  if (!launchesFirst) {
    pthread_barrier_wait(&sync.firstLaunchIssued);
  }
  if (rank == 1 || sourceStores) {
    void* kernelArguments[] = {&arguments};
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(phaseMain), state.mainGridCtas, blockThreads, kernelArguments, state.dynamicSharedBytes, state.stream));
  }
  if (launchesFirst) {
    pthread_barrier_wait(&sync.firstLaunchIssued);
  }
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
// The destination's measured sequence before phaseMain.
static void runPhasesBeforeMain(RankState& state, const Config& config, KernelArguments& arguments) {
  launchPhase(state, phaseClean, config.readerCtas, arguments);
  launchPhase(state, phaseQuiet, 1, arguments);
  if (config.prepareKind != prepareNone || config.residentCtas) {
    launchPhase(state, phasePrepare, config.prepareKind == prepareLocalStore ? config.writerCtas : config.readerCtas, arguments);
  }
}
// The destination's measured sequence after phaseMain.
static void runPhasesAfterMain(RankState& state, const Config& config, KernelArguments& arguments) {
  if (config.chaseKind == chaseAfterStores) {
    launchPhase(state, phaseChase, 1, arguments);
  }
  if (config.readback) {
    launchPhase(state, phaseReadback, config.readerCtas, arguments);
  }
  if (config.quietAfterStores) {
    launchPhase(state, phaseQuiet, 1, arguments);
  }
  if (!config.evictInMainLaunch) {
    launchPhase(state, phaseEvict, config.readerCtas, arguments);
  }
  if (config.chaseKind == chaseAfterEvict) {
    launchPhase(state, phaseChase, 1, arguments);
  }
  // With no stores the payload still holds the poison, so there is nothing to compare.
  if (config.storeSource != storeNone) {
    launchPhase(state, phaseCheck, config.readerCtas, arguments);
  }
}
// Any wrong word at the end of a transfer ends the run, unless a read-back test is looking for them.
static void collectTransfer(RankState& state, const Config& config, int rank, int transfer) {
  std::vector<CtaRecord> records(state.mainGridCtas);
  CUDA_CHECK(cudaMemcpy(records.data(), state.deviceRecords, state.mainGridCtas * sizeof(CtaRecord), cudaMemcpyDeviceToHost));
  ProbeRecord probe;
  CUDA_CHECK(cudaMemcpy(&probe, state.probe, sizeof(probe), cudaMemcpyDeviceToHost));
  std::set<int> usedSms;
  if (rank == 1 || config.storeSource == storePeer) {
    for (const auto& record : records) {
      if (record.endNs && !usedSms.insert(record.smId).second) {
        fprintf(stderr, "Two CTAs on SM %d\n", record.smId);
        std::exit(4);
      }
    }
  }
  if (rank == 1 && probe.checkWrong && !config.readback && !config.chaseKind) {
    fprintf(stderr, "CORRECTNESS FAILURE: %llu wrong words, first at offset %llu\n", probe.checkWrong, probe.firstWrongOffset);
    std::exit(4);
  }
  if (transfer >= config.warmup) {
    state.records.push_back(std::move(records));
    state.probes.push_back(probe);
  }
}
static void cleanupRank(RankState& state) {
  NCCL_CHECK(ncclDevCommDestroy(state.communicator, &state.deviceCommunicator));
  NCCL_CHECK(ncclCommWindowDeregister(state.communicator, state.window));
  NCCL_CHECK(ncclMemFree(state.windowMemory));
  CUDA_CHECK(cudaFree(state.evictionBuffer));
  CUDA_CHECK(cudaFree(state.residentBuffer));
  CUDA_CHECK(cudaFree(state.local));
  CUDA_CHECK(cudaFree(state.deviceRecords));
  CUDA_CHECK(cudaFree(state.probe));
  CUDA_CHECK(cudaStreamDestroy(state.stream));
  NCCL_CHECK(ncclCommDestroy(state.communicator));
}
static void runRank(RankState& state, Config config, int rank, ncclUniqueId identifier, HostSync& sync) {
  setupRank(state, config, rank, identifier);
  pthread_barrier_wait(&sync.transferFinished);
  for (int transfer = 0; transfer < config.warmup + config.iterations; ++transfer) {
    prepareTransfer(state, config, rank);
    KernelArguments arguments{state.window, state.evictionBuffer, state.residentBuffer, state.local, state.deviceRecords, state.probe, config, rank, transfer};
    if (rank == 1) {
      runPhasesBeforeMain(state, config, arguments);
    }
    launchMain(state, config, rank, arguments, sync);
    if (rank == 1) {
      runPhasesAfterMain(state, config, arguments);
    }
    collectTransfer(state, config, rank, transfer);
    pthread_barrier_wait(&sync.transferFinished);
  }
  cleanupRank(state);
}

static unsigned long long storedBytesPerTransfer(const Config& config) {
  const size_t elementCount = size_t(config.footprintMib) * mebibyte / sizeof(uint32_t);
  size_t wordsPerPass = 0;
  for (int ctaIndex = 0; ctaIndex < config.writerCtas; ++ctaIndex) {
    wordsPerPass += elementCount * (ctaIndex + 1) / config.writerCtas / wordsPerLine * wordsPerLine - elementCount * ctaIndex / config.writerCtas / wordsPerLine * wordsPerLine;
  }
  if (config.storeShape == shapeOneWordPerLine) {
    wordsPerPass /= wordsPerLine;
  }
  return (unsigned long long)(wordsPerPass) * sizeof(uint32_t) * config.passes;
}
static void writeJson(const char* path, const char* configPath, const Config& config, const std::array<RankState, 2>& states) {
  FILE* output = fopen(path, "w");
  if (!output) {
    perror(path);
    std::exit(1);
  }
  // Output keys are stable and consumed by analyze_peer_store_path.py.
  fprintf(output, "{\"schema_version\":1,\"harness\":\"peer_store_path\",\"config_path\":\"%s\",\"source_gpu\":%d,\"destination_gpu\":%d,\"profiled_rank\":%d,"
          "\"store_source\":%d,\"store_shape\":%d,\"writer_ctas\":%d,\"writer_threads\":%d,\"block_threads\":%d,\"footprint_mib\":%d,\"passes\":%d,\"fence_each_pass\":%d,"
          "\"stored_bytes\":%llu,\"eviction_mib\":%d,\"eviction_passes\":%d,\"reader_ctas\":%d,\"prepare_kind\":%d,\"chase_kind\":%d,\"readback\":%d,\"quiet_after_stores\":%d,"
          "\"evict_in_main_launch\":%d,\"resident_ctas\":%d,\"resident_mib\":%d,\"quiet_ns\":%llu,\"no_store_window_ns\":%llu,\"chase_wait_ns\":%llu,\"ranks\":[",
          configPath, config.sourceGpu, config.destinationGpu, config.profiledRank, config.storeSource, config.storeShape, config.writerCtas, config.writerThreads, blockThreads,
          config.footprintMib, config.passes, config.fenceEachPass, storedBytesPerTransfer(config), config.evictionMib, config.evictionPasses, config.readerCtas, config.prepareKind,
          config.chaseKind, config.readback, config.quietAfterStores, config.evictInMainLaunch, config.residentCtas, config.residentMib, config.quietNs, config.noStoreWindowNs, config.chaseWaitNs);
  static const char* roleNames[] = {"store", "receiver_or_timer", "resident_reader", "eviction_reader"};
  for (int rank = 0; rank < 2; ++rank) {
    const auto& state = states[rank];
    fprintf(output, "%s{\"rank\":%d,\"sm_count\":%d,\"shared_bytes\":%zu,\"transfers\":[", rank ? "," : "", rank, state.properties.multiProcessorCount, state.dynamicSharedBytes);
    for (size_t transfer = 0; transfer < state.records.size(); ++transfer) {
      const ProbeRecord& probe = state.probes[transfer];
      fprintf(output, "%s{\"chase_ns\":%llu,\"chase_hops\":%llu,\"chase_wrong\":%llu,\"readback_wrong\":%llu,\"check_wrong\":%llu,\"first_wrong_offset\":%lld,\"wrong_value_sample\":%llu,\"ctas\":[",
              transfer ? "," : "", probe.chaseNs, probe.chaseHops, probe.chaseWrong, probe.readbackWrong, probe.checkWrong,
              probe.firstWrongOffset == noWrongOffset ? -1ll : (long long)(probe.firstWrongOffset), probe.wrongValueSample);
      bool firstRecord = true;
      for (const auto& record : state.records[transfer]) {
        if (!record.endNs) {
          continue;
        }
        fprintf(output, "%s{\"role\":\"%s\",\"sm_id\":%d,\"start_ns\":%llu,\"issue_end_ns\":%llu,\"fence_end_ns\":%llu,\"publish_end_ns\":%llu,\"end_ns\":%llu,\"work_units\":%llu}",
                firstRecord ? "" : ",", roleNames[record.roleCode], record.smId, record.startNs, record.issueEndNs, record.fenceEndNs, record.publishEndNs, record.endNs, record.workUnits);
        firstRecord = false;
      }
      fprintf(output, "]}");
    }
    fprintf(output, "]}");
  }
  fprintf(output, "]}\n");
  fclose(output);
}

// User allocation for plan 08: physical GPUs 6 (source) and 7 (destination).
static bool authorizedGpu(int gpu) {
  return gpu == 6 || gpu == 7;
}

// Usage: peer_store_path CONFIG OUTPUT.json --gpus SOURCE,DESTINATION [--profiled-rank RANK]
int main(int argumentCount, char** arguments) {
  if (argumentCount < 3) {
    fprintf(stderr, "Usage: peer_store_path CONFIG OUTPUT.json --gpus S,D [--profiled-rank RANK]\n");
    return 1;
  }
  Config config = parseConfiguration(arguments[1]);
  for (int index = 3; index < argumentCount; ++index) {
    const std::string option = arguments[index];
    if (option == "--gpus" && index + 1 < argumentCount && sscanf(arguments[index + 1], "%d,%d", &config.sourceGpu, &config.destinationGpu) == 2) {
      ++index;
    } else if (option == "--profiled-rank" && index + 1 < argumentCount) {
      config.profiledRank = std::stoi(arguments[++index]);
    } else {
      fprintf(stderr, "Unknown option %s\n", arguments[index]);
      return 1;
    }
  }
  validateConfiguration(config);
  if (!authorizedGpu(config.sourceGpu) || !authorizedGpu(config.destinationGpu) || config.sourceGpu == config.destinationGpu) {
    fprintf(stderr, "Need --gpus with a pair inside the authorized allocation\n");
    return 1;
  }
  ncclUniqueId identifier;
  NCCL_CHECK(ncclGetUniqueId(&identifier));
  HostSync sync;
  pthread_barrier_init(&sync.readyForMain, nullptr, 2);
  pthread_barrier_init(&sync.firstLaunchIssued, nullptr, 2);
  pthread_barrier_init(&sync.transferFinished, nullptr, 2);
  std::array<RankState, 2> states;
  std::thread sourceThread(runRank, std::ref(states[0]), config, 0, identifier, std::ref(sync));
  std::thread destinationThread(runRank, std::ref(states[1]), config, 1, identifier, std::ref(sync));
  sourceThread.join();
  destinationThread.join();
  pthread_barrier_destroy(&sync.readyForMain);
  pthread_barrier_destroy(&sync.firstLaunchIssued);
  pthread_barrier_destroy(&sync.transferFinished);
  writeJson(arguments[2], arguments[1], config, states);
  return 0;
}
