// Plan 07 harness: store-only NCCL LSA push next to a local thread-store writer on the source GPU.
//
// Differences from push_harness.cu (plan 04):
//   - The copy computes each value from its index in registers and never loads source data,
//     so its issue loop is pure store issue.
//   - The writer stores 16-byte vectors to a local 1 GiB buffer with a target-rate knob.
//   - Roles are chosen by SM, not by CTA index: one CTA runs on every SM, reads its SM ID and
//     takes the role the host's role table gives that SM (copy, writer, or idle). The table is
//     built from the measured SM-to-GPC map (metric A12), so a config fixes which GPCs and SMs
//     each role uses in every run.
// The completion protocol (all writers fence, each copy CTA publishes "done", the receiver
// acknowledges) and its timers are the same as in push_harness.cu.
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
constexpr size_t writerBufferBytes = size_t(1) << 30;
constexpr int writerBlockBytes = 64 * 1024;
constexpr int writerBlockVectors = writerBlockBytes / 16;
constexpr int allGpcs = 0xff;
enum Role { roleIdle = 0, roleCopy = 1, roleWriter = 2 };
static const char* roleNames[] = {"idle", "copy", "writer"};

struct alignas(128) Signal { int value; };
struct WindowControl {
  Signal destinationReady;
  Signal acknowledgement;
  Signal completion[maxSms];
};
struct LocalControl {
  int copyRegistered;      // gives each copy CTA its part of the array, in arrival order
  int writerRegistered;    // gives each writer CTA its slice of the writer buffer
  int writersReady;
  int finishedCount;
  int stop;
};
struct CtaRecord {
  unsigned long long startNs;       // A2 / A8
  unsigned long long issueEndNs;    // A3
  unsigned long long fenceEndNs;    // A4
  unsigned long long publishEndNs;  // A5
  unsigned long long endNs;         // A1 / A6 / A8
  unsigned long long workUnits;     // A9: 64 KiB writer blocks
  unsigned long long checkpointsNs[checkpointCount]; // A7 issue progress, not arrival
  int smId;                         // A10
  int role;                         // A10: Role
  int roleIndex;                    // copy part or writer slice
};
struct Config {
  int sourceGpu = -1;
  int destinationGpu = -1;
  // Copy: CTAs × active threads and where it runs.
  int copyCtas = 64;
  int copyThreads = 512;
  int copyGpcMask = allGpcs;
  int copyPlacement = 0;            // 0 spread round-robin over GPCs, 1 packed into the largest GPCs
  // Writer: CTAs × active threads, where it runs, and the target total write rate (0 = no limit).
  int writerCtas = 0;
  int writerThreads = 1024;
  int writerGpcMask = allGpcs;
  int writerPlacement = 0;
  int targetGbps = 0;
  int writerIdle = 0;               // writer CTAs start and wait for stop without writing (the reference)
  int writerAlternate = 0;          // 1 idle first, 2 writing first; alternates per launch inside a run
  int useUnmapped = 0;              // allow roles on SMs the GPC map could not place, after mapped SMs
  int blockThreads = 1024;
  int sharedBytes = 0;
  int iterations = 12;
  int warmup = 3;
  int detailed = 1;
  int communicationEnabled = 1;     // 0: copy CTA 0 only times a window (writer-alone runs)
  size_t payloadBytes = size_t(2048) << 20;
  unsigned long long soloDurationNs = 20000000;
};
struct KernelArguments {
  ncclWindow_t window;
  uint4* writerBuffer;
  unsigned long long* writerBlocks;
  const int* roleOfSm;
  size_t elementCount;
  LocalControl* local;
  CtaRecord* records;
  Config config;
  int rank;
  int iteration;
  int writerIdle;
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
// Same value function as push_harness.cu; it depends on the position and the transfer, so the
// destination check catches missing and misplaced writes.
__device__ __forceinline__ uint32_t expectedValue(size_t offset, int iteration) {
  return uint32_t(offset) * 1664525u ^ uint32_t(iteration + 1) * 1013904223u;
}
__device__ __forceinline__ uint4 writerValue(size_t vectorIndex, int iteration) {
  size_t word = vectorIndex * 4;
  return make_uint4(expectedValue(word, iteration), expectedValue(word + 1, iteration), expectedValue(word + 2, iteration), expectedValue(word + 3, iteration));
}

__global__ void verifyDestination(const uint32_t* destination, size_t elementCount, int iteration, unsigned long long* mismatches) {
  unsigned long long localMismatches = 0;
  for (size_t offset = size_t(blockIdx.x) * blockDim.x + threadIdx.x; offset < elementCount; offset += size_t(blockDim.x) * gridDim.x) {
    localMismatches += destination[offset] != expectedValue(offset, iteration);
  }
  if (localMismatches) {
    atomicAdd(mismatches, localMismatches);
  }
}
// Checks every vector the writer wrote in this launch: a CTA that wrote more blocks than its slice
// holds has rewritten the whole slice with this launch's values.
__global__ void verifyWriterBuffer(const uint4* buffer, int writerCtas, size_t sliceBlocks, const unsigned long long* writerBlocks, int iteration, unsigned long long* mismatches) {
  const size_t sliceVectors = sliceBlocks * writerBlockVectors;
  const size_t totalVectors = sliceVectors * writerCtas;
  unsigned long long localMismatches = 0;
  for (size_t vectorIndex = size_t(blockIdx.x) * blockDim.x + threadIdx.x; vectorIndex < totalVectors; vectorIndex += size_t(blockDim.x) * gridDim.x) {
    size_t writer = vectorIndex / sliceVectors;
    size_t positionInSlice = vectorIndex % sliceVectors;
    size_t writtenBlocks = writerBlocks[writer] < sliceBlocks ? writerBlocks[writer] : sliceBlocks;
    size_t writtenVectors = writtenBlocks * writerBlockVectors;
    if (positionInSlice < writtenVectors) {
      uint4 value = buffer[vectorIndex];
      uint4 expected = writerValue(vectorIndex, iteration);
      localMismatches += value.x != expected.x || value.y != expected.y || value.z != expected.z || value.w != expected.w;
    }
  }
  if (localMismatches) {
    atomicAdd(mismatches, localMismatches);
  }
}

// The copy starts only after every writer CTA has started, so writing covers the whole copy.
__device__ void waitForWriters(KernelArguments arguments) {
  if (threadIdx.x == 0) {
    while (acquireSignal(&arguments.local->writersReady) != arguments.config.writerCtas) {
      __nanosleep(64);
    }
  }
  __syncthreads();
}
// One 4-byte peer store per loop step; the value comes from registers. A "stores per loop step"
// knob (copy_batch) was removed on 2026-10-30: stores do not wait for a reply, so it could not
// change the stores in flight, and the compiler kept one bounds-check branch per store, which made
// every value above 1 cost more instructions per store (SASS); it never raised throughput by more
// than 0.3% and lowered it by up to 41% where the copy SMs limit throughput (E7.1).
__device__ void issueStoreOnlyCopy(KernelArguments arguments, uint32_t* destination, size_t laneOffset, size_t laneStride, size_t regionEnd, CtaRecord& record) {
  size_t laneIterations = laneOffset < regionEnd ? (regionEnd - laneOffset + laneStride - 1) / laneStride : 0;
  size_t checkpointStride = (laneIterations + checkpointCount - 1) / checkpointCount;
  int checkpointIndex = 0;
  size_t progress = 0;
  size_t nextCheckpoint = checkpointStride;
  #pragma unroll 1
  for (size_t offset = laneOffset; offset < regionEnd; offset += laneStride) {
    destination[offset] = expectedValue(offset, arguments.iteration);
    ++progress;
    if (arguments.config.detailed && threadIdx.x == 0 && progress == nextCheckpoint && checkpointIndex < checkpointCount) {
      record.checkpointsNs[checkpointIndex++] = readTimerNs();
      nextCheckpoint += checkpointStride;
    }
  }
}

__device__ void runCopy(KernelArguments arguments, int copyIndex) {
  WindowControl* localSignals = windowControl(arguments.window, arguments.config.payloadBytes, 0);
  WindowControl* remoteSignals = windowControl(arguments.window, arguments.config.payloadBytes, 1);
  auto* destination = static_cast<uint32_t*>(ncclGetLsaPointer(arguments.window, 0, 1));
  CtaRecord& record = arguments.records[blockIdx.x];
  waitForWriters(arguments);
  if (threadIdx.x == 0) {
    while (!acquireSignal(&localSignals->destinationReady.value)) {
      __nanosleep(64);
    }
    record.startNs = readTimerNs();
  }
  __syncthreads();
  // CTA-local stride; lanes beyond the copy thread count issue nothing but join every barrier.
  // Part boundaries are rounded down to 32 elements (128 bytes): otherwise, for CTA counts such as
  // 48 or 96, every warp store crossed two 128-byte lines and the copy lost about 30% (first sweep).
  const size_t copyThreads = arguments.config.copyThreads;
  size_t regionBegin = arguments.elementCount * copyIndex / arguments.config.copyCtas / 32 * 32;
  size_t regionEnd = arguments.elementCount * (copyIndex + 1) / arguments.config.copyCtas / 32 * 32;
  size_t laneOffset = regionBegin + threadIdx.x;
  if (threadIdx.x >= copyThreads) {
    regionEnd = 0;
  }
  issueStoreOnlyCopy(arguments, destination, laneOffset, copyThreads, regionEnd, record);
  __syncthreads();
  if (arguments.config.detailed && threadIdx.x == 0) {
    record.issueEndNs = readTimerNs();
  }
  // Every writer orders its own peer writes; CTA sync precedes publication by lane 0.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    if (arguments.config.detailed) {
      record.fenceEndNs = readTimerNs();
    }
    releaseSignal(&remoteSignals->completion[copyIndex].value, 1);
    if (arguments.config.detailed) {
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
}
__device__ void runReceiver(KernelArguments arguments) {
  WindowControl* localSignals = windowControl(arguments.window, arguments.config.payloadBytes, 1);
  WindowControl* remoteSignals = windowControl(arguments.window, arguments.config.payloadBytes, 0);
  if (threadIdx.x == 0) {
    CtaRecord& record = arguments.records[0];
    record.startNs = readTimerNs();
    releaseSignal(&remoteSignals->destinationReady.value, 1);
    for (int copyIndex = 0; copyIndex < arguments.config.copyCtas; ++copyIndex) {
      while (!acquireSignal(&localSignals->completion[copyIndex].value)) {
        __nanosleep(64);
      }
    }
    record.issueEndNs = readTimerNs();
    releaseSignal(&remoteSignals->acknowledgement.value, 1);
    record.endNs = readTimerNs();
  }
}
// Writer-alone runs: copy CTA 0 times a fixed window after all writers start, then stops them.
__device__ void runSoloTimer(KernelArguments arguments) {
  waitForWriters(arguments);
  if (threadIdx.x == 0) {
    CtaRecord& record = arguments.records[blockIdx.x];
    record.startNs = readTimerNs();
    while (readTimerNs() - record.startNs < arguments.config.soloDurationNs) {
      __nanosleep(128);
    }
    record.endNs = readTimerNs();
    releaseSignal(&arguments.local->stop, 1);
  }
}

// Rate limit: block k of this CTA may start at startNs + k × interval, never earlier. A late
// block starts at once, so an achieved rate below target shows lasting slowdown.
__device__ __forceinline__ void waitForScheduledBlock(unsigned long long startNs, unsigned long long blockIndex, unsigned long long intervalPs) {
  if (intervalPs == 0) {
    return;
  }
  unsigned long long scheduledNs = startNs + blockIndex * intervalPs / 1000;
  while (readTimerNs() < scheduledNs) {
  }
}
// Each CTA writes one 64 KiB block of its slice with all active threads, then the next block,
// wrapping inside the slice. One CTA barrier per block releases the block after its start time.
// Stop check without a memory round trip per block: thread 0 loads the global stop flag while
// the block is written and passes the value through shared memory one block later (two slots,
// so a slot is never rewritten while a slower warp may still read it). Writers therefore stop
// at most two blocks after the copy finishes; those blocks are counted.
__device__ void runWriter(KernelArguments arguments, int writerIndex) {
  __shared__ int stopSlots[2];
  CtaRecord& record = arguments.records[blockIdx.x];
  const int writerThreads = arguments.config.writerThreads;
  const size_t sliceBlocks = writerBufferBytes / writerBlockBytes / arguments.config.writerCtas;
  uint4* slice = arguments.writerBuffer + size_t(writerIndex) * sliceBlocks * writerBlockVectors;
  const size_t sliceFirstVector = size_t(writerIndex) * sliceBlocks * writerBlockVectors;
  if (threadIdx.x == 0) {
    stopSlots[0] = 0;
    stopSlots[1] = 0;
    record.startNs = readTimerNs();
    cuda::atomic_ref<int, cuda::thread_scope_system>(arguments.local->writersReady).fetch_add(1, cuda::memory_order_release);
  }
  __syncthreads();
  if (arguments.writerIdle) {
    if (threadIdx.x == 0) {
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
    if (threadIdx.x == 0) {
      waitForScheduledBlock(startNs, blocks, arguments.writerIntervalPs);
    }
    __syncthreads();
    if (stopSlots[(blocks + 1) & 1]) {
      break;
    }
    if (threadIdx.x == 0) {
      stopSlots[blocks & 1] = pendingStop;
      pendingStop = *reinterpret_cast<volatile int*>(&arguments.local->stop);
    }
    if (threadIdx.x < writerThreads) {
      const size_t blockFirstVector = (blocks % sliceBlocks) * writerBlockVectors;
      for (int vector = threadIdx.x; vector < writerBlockVectors; vector += writerThreads) {
        slice[blockFirstVector + vector] = writerValue(sliceFirstVector + blockFirstVector + vector, arguments.iteration);
      }
    }
  }
  if (threadIdx.x == 0) {
    record.endNs = readTimerNs();
    record.workUnits = blocks;
    arguments.writerBlocks[writerIndex] = blocks;
  }
}

// Rank 0 runs one CTA per SM; each takes the role of its SM. Rank 1 runs only the receiver.
__global__ void __launch_bounds__(maxThreadsPerCta, 1) storeCongest(KernelArguments arguments) {
  __shared__ int roleIndex;
  if (arguments.rank == 1) {
    if (arguments.config.communicationEnabled) {
      runReceiver(arguments);
    }
    return;
  }
  CtaRecord& record = arguments.records[blockIdx.x];
  const int smId = readSmId();
  const int role = arguments.roleOfSm[smId];
  if (threadIdx.x == 0) {
    record.smId = smId;
    record.role = role;
    record.startNs = readTimerNs();
    if (role == roleCopy) {
      roleIndex = atomicAdd(&arguments.local->copyRegistered, 1);
    } else if (role == roleWriter) {
      roleIndex = atomicAdd(&arguments.local->writerRegistered, 1);
    } else {
      roleIndex = -1;
      record.endNs = record.startNs;
    }
    record.roleIndex = roleIndex;
  }
  __syncthreads();
  if (role == roleWriter) {
    runWriter(arguments, roleIndex);
  } else if (role == roleCopy && arguments.config.communicationEnabled) {
    runCopy(arguments, roleIndex);
  } else if (role == roleCopy && roleIndex == 0) {
    runSoloTimer(arguments);
  } else if (role == roleCopy && threadIdx.x == 0) {
    record.endNs = record.startNs;
  }
}

struct RankState {
  ncclComm_t communicator;
  ncclDevComm deviceCommunicator;
  ncclWindow_t window;
  void* windowMemory;
  uint4* writerBuffer;
  unsigned long long* writerBlocks;
  int* roleOfSm;
  LocalControl* local;
  CtaRecord* deviceRecords;
  unsigned long long* mismatches;
  cudaStream_t stream;
  int gridCtas;
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

// User allocation for plan 07: physical GPUs 4 and 5 from 2026-10-29; GPUs 0 and 1 from 2026-10-30.
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
    } else if (key == "copy_gpcs") {
      config.copyGpcMask = value;
    } else if (key == "copy_placement") {
      config.copyPlacement = value;
    } else if (key == "writer_ctas") {
      config.writerCtas = value;
    } else if (key == "writer_threads") {
      config.writerThreads = value;
    } else if (key == "writer_gpcs") {
      config.writerGpcMask = value;
    } else if (key == "writer_placement") {
      config.writerPlacement = value;
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
  bool copyValid = config.copyCtas >= 1 && config.copyCtas <= maxSms && config.copyThreads >= 32 && config.copyThreads % 32 == 0 && config.copyThreads <= config.blockThreads;
  bool writerValid = config.writerCtas >= 0 && config.writerCtas + config.copyCtas <= maxSms && config.writerThreads >= 32 && config.writerThreads % 32 == 0 && writerBlockVectors % config.writerThreads == 0 && config.writerThreads <= config.blockThreads && config.targetGbps >= 0;
  bool alternationValid = config.writerAlternate >= 0 && config.writerAlternate <= 2 && !(config.writerAlternate && (config.writerIdle || config.writerCtas == 0));
  bool runValid = config.blockThreads <= maxThreadsPerCta && config.blockThreads % 32 == 0 && config.iterations >= 1 && config.warmup >= 0 && config.soloDurationNs > 0 && config.payloadBytes >= (size_t(1) << 20) && config.payloadBytes <= (size_t(4096) << 20) && config.copyPlacement >= 0 && config.copyPlacement <= 1 && config.writerPlacement >= 0 && config.writerPlacement <= 1;
  if (!copyValid || !writerValid || !alternationValid || !runValid) {
    fprintf(stderr, "Invalid configuration (copy %d, writer %d, alternation %d, run %d)\n", copyValid, writerValid, alternationValid, runValid);
    std::exit(1);
  }
}

static std::string uuidText(const cudaUUID_t& uuid) {
  char text[64];
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(uuid.bytes);
  snprintf(text, sizeof(text), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
           bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
  return text;
}
// Reads the map written by build_gpc_map.py: {"uuid": "...", ..., "gpc_of_sm": [g0, g1, ...]}.
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
static void buildRoleTable(Placement& placement, const Config& config, const cudaDeviceProp& properties) {
  loadGpcMap(placement);
  if (placement.mapUuid != uuidText(properties.uuid) || int(placement.gpcOfSm.size()) != properties.multiProcessorCount || properties.multiProcessorCount > maxSms) {
    fprintf(stderr, "GPC map %s does not belong to this GPU (%s, %d SMs)\n", placement.mapPath.c_str(), uuidText(properties.uuid).c_str(), properties.multiProcessorCount);
    std::exit(1);
  }
  placement.roleOfSm.assign(properties.multiProcessorCount, roleIdle);
  assignRole(placement, roleCopy, config.copyCtas, config.copyGpcMask, config.copyPlacement, config.useUnmapped);
  assignRole(placement, roleWriter, config.writerCtas, config.writerGpcMask, config.writerPlacement, config.useUnmapped);
}

static void setupRank(RankState& state, const Config& config, Placement& placement, int rank, ncclUniqueId identifier) {
  int device = rank == 0 ? config.sourceGpu : config.destinationGpu;
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaGetDeviceProperties(&state.properties, device));
  if (rank == 0) {
    buildRoleTable(placement, config, state.properties);
  }
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
  state.gridCtas = rank == 0 ? state.properties.multiProcessorCount : 1;
  CUDA_CHECK(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&state.local, sizeof(LocalControl)));
  CUDA_CHECK(cudaMalloc(&state.deviceRecords, state.gridCtas * sizeof(CtaRecord)));
  CUDA_CHECK(cudaMalloc(&state.mismatches, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&state.roleOfSm, maxSms * sizeof(int)));
  CUDA_CHECK(cudaMemset(state.roleOfSm, 0, maxSms * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&state.writerBlocks, maxSms * sizeof(unsigned long long)));
  state.writerBuffer = nullptr;
  if (rank == 0) {
    CUDA_CHECK(cudaMemcpy(state.roleOfSm, placement.roleOfSm.data(), placement.roleOfSm.size() * sizeof(int), cudaMemcpyHostToDevice));
    if (config.writerCtas) {
      CUDA_CHECK(cudaMalloc(&state.writerBuffer, writerBufferBytes));
      CUDA_CHECK(cudaMemset(state.writerBuffer, 0, writerBufferBytes));
    }
  }
  // More than half an SM's shared memory ensures no two CTAs share an SM.
  state.dynamicSharedBytes = ((state.properties.sharedMemPerMultiprocessor / 2 + 255) / 256) * 256;
  if (config.sharedBytes) {
    if (size_t(config.sharedBytes) < state.dynamicSharedBytes || size_t(config.sharedBytes) > state.properties.sharedMemPerBlockOptin) {
      fprintf(stderr, "shared_bytes must keep one CTA per SM and fit one CTA's limit\n");
      std::exit(1);
    }
    state.dynamicSharedBytes = config.sharedBytes;
  }
  CUDA_CHECK(cudaFuncSetAttribute(storeCongest, cudaFuncAttributeMaxDynamicSharedMemorySize, state.dynamicSharedBytes));
  int residentCtas = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residentCtas, storeCongest, config.blockThreads, state.dynamicSharedBytes));
  if (residentCtas != 1) {
    fprintf(stderr, "Expected exactly one resident CTA per SM, got %d\n", residentCtas);
    std::exit(1);
  }
  // Preload the measured function before concurrent launches (avoids lazy-load waits).
  cudaFuncAttributes attributes;
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, storeCongest));
  fprintf(stderr, "rank=%d gpu=%d SMs=%d grid=%d block=%d copy=%dx%d writer=%dx%d target_gbps=%d shared=%zu registers=%d local_bytes=%zu\n", rank, device, state.properties.multiProcessorCount, state.gridCtas, config.blockThreads, config.copyCtas, config.copyThreads, config.writerCtas, config.writerThreads, config.targetGbps, state.dynamicSharedBytes, attributes.numRegs, attributes.localSizeBytes);
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
static void prepareIteration(RankState& state, const Config& config, int rank) {
  CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(state.windowMemory) + config.payloadBytes, 0, sizeof(WindowControl), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.local, 0, sizeof(LocalControl), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.deviceRecords, 0, state.gridCtas * sizeof(CtaRecord), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.mismatches, 0, sizeof(unsigned long long), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.writerBlocks, 0, maxSms * sizeof(unsigned long long), state.stream));
  if (rank == 1 && config.communicationEnabled) {
    CUDA_CHECK(cudaMemsetAsync(state.windowMemory, 0xff, config.payloadBytes, state.stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
// Whether the writer CTAs write in this launch; counts warm-ups so the order is fixed per run.
static int launchIsIdle(const Config& config, int iteration) {
  if (!config.writerAlternate) {
    return config.writerIdle;
  }
  return (iteration % 2 == 0) == (config.writerAlternate == 1);
}
static unsigned long long writerIntervalPs(const Config& config) {
  if (!config.targetGbps || !config.writerCtas) {
    return 0;
  }
  return (unsigned long long)(double(writerBlockBytes) * config.writerCtas * 1000.0 / config.targetGbps);
}
static unsigned long long readMismatches(RankState& state) {
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
  unsigned long long mismatches = 0;
  CUDA_CHECK(cudaMemcpy(&mismatches, state.mismatches, sizeof(mismatches), cudaMemcpyDeviceToHost));
  return mismatches;
}
// Checks placement, timing, writer work and both buffers; any failure ends the run with no record.
static void collectIteration(RankState& state, const Config& config, int rank, int iteration) {
  const int writerIdle = launchIsIdle(config, iteration);
  std::vector<CtaRecord> records(state.gridCtas);
  CUDA_CHECK(cudaMemcpy(records.data(), state.deviceRecords, state.gridCtas * sizeof(CtaRecord), cudaMemcpyDeviceToHost));
  if (rank == 0) {
    std::set<int> usedSms;
    std::set<int> copyIndexes;
    std::set<int> writerIndexes;
    for (const auto& record : records) {
      bool active = record.role == roleWriter || (record.role == roleCopy && (config.communicationEnabled || record.roleIndex == 0));
      if (!usedSms.insert(record.smId).second || (active && record.endNs <= record.startNs)) {
        fprintf(stderr, "Invalid SM separation or CTA timing\n");
        std::exit(1);
      }
      if (record.role == roleCopy) {
        copyIndexes.insert(record.roleIndex);
      }
      if (record.role == roleWriter) {
        writerIndexes.insert(record.roleIndex);
        if (!writerIdle && record.workUnits == 0) {
          fprintf(stderr, "Writer CTA wrote nothing\n");
          std::exit(4);
        }
      }
    }
    if (int(copyIndexes.size()) != config.copyCtas || int(writerIndexes.size()) != config.writerCtas) {
      fprintf(stderr, "Role counts differ from the role table\n");
      std::exit(1);
    }
    if (config.writerCtas && !writerIdle) {
      size_t sliceBlocks = writerBufferBytes / writerBlockBytes / config.writerCtas;
      verifyWriterBuffer<<<528, 256, 0, state.stream>>>(state.writerBuffer, config.writerCtas, sliceBlocks, state.writerBlocks, iteration, state.mismatches);
      unsigned long long mismatches = readMismatches(state);
      if (mismatches) {
        fprintf(stderr, "WRITER CHECK FAILURE: %llu mismatching vectors\n", mismatches);
        std::exit(4);
      }
    }
  }
  if (rank == 1 && config.communicationEnabled) {
    verifyDestination<<<528, 256, 0, state.stream>>>(static_cast<uint32_t*>(state.windowMemory), config.payloadBytes / sizeof(uint32_t), iteration, state.mismatches);
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
  CUDA_CHECK(cudaFree(state.writerBlocks));
  CUDA_CHECK(cudaFree(state.roleOfSm));
  CUDA_CHECK(cudaFree(state.local));
  CUDA_CHECK(cudaFree(state.deviceRecords));
  CUDA_CHECK(cudaFree(state.mismatches));
  CUDA_CHECK(cudaStreamDestroy(state.stream));
  NCCL_CHECK(ncclCommDestroy(state.communicator));
}
static void runRank(RankState& state, Config config, Placement& placement, int rank, ncclUniqueId identifier, pthread_barrier_t* hostBarrier) {
  setupRank(state, config, placement, rank, identifier);
  pthread_barrier_wait(hostBarrier);
  for (int iteration = 0; iteration < config.warmup + config.iterations; ++iteration) {
    prepareIteration(state, config, rank);
    pthread_barrier_wait(hostBarrier);
    KernelArguments arguments{state.window, state.writerBuffer, state.writerBlocks, state.roleOfSm, config.payloadBytes / sizeof(uint32_t), state.local, state.deviceRecords, config, rank, iteration, launchIsIdle(config, iteration), writerIntervalPs(config)};
    void* kernelArguments[] = {&arguments};
    // Cooperative launch: every CTA is resident at once, so waits between roles cannot deadlock.
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(storeCongest), state.gridCtas, config.blockThreads, kernelArguments, state.dynamicSharedBytes, state.stream));
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
    pthread_barrier_wait(hostBarrier);
    collectIteration(state, config, rank, iteration);
    pthread_barrier_wait(hostBarrier);
  }
  cleanupRank(state);
}

static void writeIntegerList(FILE* output, const std::vector<int>& values) {
  fputc('[', output);
  for (size_t index = 0; index < values.size(); ++index) {
    fprintf(output, "%s%d", index ? "," : "", values[index]);
  }
  fputc(']', output);
}
static void writeJson(const char* path, const char* configPath, const Config& config, const Placement& placement, const std::array<RankState, 2>& states) {
  FILE* output = fopen(path, "w");
  if (!output) {
    perror(path);
    std::exit(1);
  }
  // Output keys are stable and consumed by analyze_store_congest.py. Schema 2 (2026-10-30) drops
  // "copy_batch"; schema-1 records carry it, and their label suffix _b<N> gives the same value.
  fprintf(output, "{\"schema_version\":2,\"harness\":\"store_congest\",\"config_path\":\"%s\",\"source_gpu\":%d,\"destination_gpu\":%d,"
          "\"copy_ctas\":%d,\"copy_threads\":%d,\"copy_gpcs\":%d,\"copy_placement\":%d,"
          "\"writer_ctas\":%d,\"writer_threads\":%d,\"writer_gpcs\":%d,\"writer_placement\":%d,\"target_gbps\":%d,\"writer_idle\":%d,\"writer_alternate\":%d,"
          "\"use_unmapped\":%d,\"block_threads\":%d,\"detailed\":%d,\"communication\":%d,\"solo_ns\":%llu,\"bytes\":%zu,\"writer_block_bytes\":%d,\"writer_buffer_bytes\":%zu,\"verified\":true,"
          "\"gpc_map\":\"%s\",\"gpc_map_uuid\":\"%s\",\"gpc_of_sm\":",
          configPath, config.sourceGpu, config.destinationGpu, config.copyCtas, config.copyThreads, config.copyGpcMask, config.copyPlacement,
          config.writerCtas, config.writerThreads, config.writerGpcMask, config.writerPlacement, config.targetGbps, config.writerIdle, config.writerAlternate,
          config.useUnmapped, config.blockThreads, config.detailed, config.communicationEnabled, config.soloDurationNs, config.payloadBytes, writerBlockBytes, writerBufferBytes,
          placement.mapPath.c_str(), placement.mapUuid.c_str());
  writeIntegerList(output, placement.gpcOfSm);
  fprintf(output, ",\"role_of_sm\":");
  writeIntegerList(output, placement.roleOfSm);
  fprintf(output, ",\"ranks\":[");
  for (int rank = 0; rank < 2; ++rank) {
    if (rank) {
      fputc(',', output);
    }
    const auto& state = states[rank];
    fprintf(output, "{\"rank\":%d,\"sm_count\":%d,\"shared_bytes\":%zu,\"writer_idle_by_iteration\":", rank, state.properties.multiProcessorCount, state.dynamicSharedBytes);
    writeIntegerList(output, state.writerIdleByIteration);
    fprintf(output, ",\"iterations\":[");
    for (size_t iteration = 0; iteration < state.records.size(); ++iteration) {
      fprintf(output, "%s[", iteration ? "," : "");
      bool firstRecord = true;
      for (size_t recordIndex = 0; recordIndex < state.records[iteration].size(); ++recordIndex) {
        const auto& record = state.records[iteration][recordIndex];
        // Idle CTAs only prove the SM was occupied; they are listed with their SM for A10.
        fprintf(output, "%s{\"cta\":%zu,\"role\":\"%s\",\"role_index\":%d,\"sm_id\":%d,\"start_ns\":%llu,\"issue_end_ns\":%llu,\"fence_end_ns\":%llu,\"publish_end_ns\":%llu,\"end_ns\":%llu,\"work_units\":%llu,\"checkpoints_ns\":[",
                firstRecord ? "" : ",", recordIndex, rank == 0 ? roleNames[record.role] : "receiver", record.roleIndex, record.smId, record.startNs, record.issueEndNs, record.fenceEndNs, record.publishEndNs, record.endNs, record.workUnits);
        firstRecord = false;
        for (int checkpoint = 0; checkpoint < checkpointCount; ++checkpoint) {
          fprintf(output, "%s%llu", checkpoint ? "," : "", record.checkpointsNs[checkpoint]);
        }
        fprintf(output, "]}");
      }
      fprintf(output, "]");
    }
    fprintf(output, "]}");
  }
  fprintf(output, "]}\n");
  fclose(output);
}

// Usage: store_congest CONFIG OUTPUT.json --gpus SOURCE,DESTINATION --gpc-map SOURCE_MAP.json
// Configs carry no GPU numbers; the pair and the source GPU's map are given per invocation.
int main(int argumentCount, char** arguments) {
  if (argumentCount < 3) {
    fprintf(stderr, "Usage: store_congest CONFIG OUTPUT.json --gpus S,D --gpc-map MAP.json\n");
    return 1;
  }
  Config config = parseConfiguration(arguments[1]);
  Placement placement;
  for (int index = 3; index < argumentCount; ++index) {
    std::string option = arguments[index];
    if (option == "--gpus" && index + 1 < argumentCount && sscanf(arguments[index + 1], "%d,%d", &config.sourceGpu, &config.destinationGpu) == 2) {
      ++index;
    } else if (option == "--gpc-map" && index + 1 < argumentCount) {
      placement.mapPath = arguments[++index];
    } else {
      fprintf(stderr, "Unknown option %s\n", arguments[index]);
      return 1;
    }
  }
  validateConfiguration(config);
  if (!authorizedGpu(config.sourceGpu) || !authorizedGpu(config.destinationGpu) || config.sourceGpu == config.destinationGpu || placement.mapPath.empty()) {
    fprintf(stderr, "Need --gpus with a pair inside the authorized allocation and --gpc-map\n");
    return 1;
  }
  ncclUniqueId identifier;
  NCCL_CHECK(ncclGetUniqueId(&identifier));
  pthread_barrier_t hostBarrier;
  pthread_barrier_init(&hostBarrier, nullptr, 2);
  std::array<RankState, 2> states;
  // Rank 0 builds the role table during setup; rank 1 never reads it.
  std::thread sourceThread(runRank, std::ref(states[0]), config, std::ref(placement), 0, identifier, &hostBarrier);
  std::thread destinationThread(runRank, std::ref(states[1]), config, std::ref(placement), 1, identifier, &hostBarrier);
  sourceThread.join();
  destinationThread.join();
  pthread_barrier_destroy(&hostBarrier);
  writeJson(arguments[2], arguments[1], config, placement, states);
  return 0;
}
