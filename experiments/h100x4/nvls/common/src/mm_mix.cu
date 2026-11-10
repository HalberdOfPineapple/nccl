// Plan 01 harness: st_multicast, ld_reduce and plain ld/st roles on four GPUs, over a fixed time window.
//
// One CTA runs on every SM of every GPU and takes the role that the host's table gives its SM, so
// all arms have the same launch and the same shared-memory reservation. Each role loop has the shape
// of the NCCL symmetric kernels (16-byte packs, 8 packs per thread step, warp-contiguous addresses).
// All roles run for a fixed time, not a fixed byte count: with fixed bytes the faster role would
// finish first and the slower role would then run alone, which hides the interaction under test.
// JSON keys are the experiment data interface; renaming C++ fields must not rename keys.
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <nccl.h>
#include <nccl_device.h>
#include <pthread.h>
#include <algorithm>
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

constexpr int rankCount = 4;
constexpr int maxSms = 132;
constexpr int maxThreadsPerCta = 1024;
constexpr int packsPerStep = 8;              // NCCL's ncclSymkDeepUnrollPacks and allreduceMultimem unroll
constexpr int packBytes = 16;
constexpr int warpThreads = 32;
constexpr size_t chunkBytes = size_t(1) << 20;   // one CTA barrier and one stop check per chunk
constexpr size_t rankPartBytes = size_t(2048) << 20;
constexpr size_t regionBytes = rankPartBytes * rankCount;
constexpr size_t rankOffsetUnitBytes = size_t(32) << 20;  // NCCL's ncclSymkMcPerRankOffsetBytes
constexpr int patternPeriodWords = 4096;
enum RoleKind { kindIdle = 0, kindController = 1, kindMulticast = 2, kindReduce = 3, kindAllreduce = 4, kindPlainLoad = 5, kindPlainStore = 6 };
static const char* roleNames[] = {"idle", "controller", "multicast", "reduce", "allreduce", "plain_load", "plain_store"};

struct alignas(128) Signal { int value; };
struct WindowControl {
  Signal rankReady[rankCount];   // on rank 0: each rank's role CTAs have arrived
  Signal startAll;               // on every rank: rank 0 saw all ranks ready
};
struct LocalControl {
  int arrivedCtas;
  int registered[2];             // gives each CTA of role slot A or B its slice, in arrival order
  int startFlag;
  int stopFlag;
  int finishedCtas;
};
struct CtaRecord {
  unsigned long long startNs;       // N1 / N4: role loop starts, after the all-GPU start
  unsigned long long endNs;         // N1 / N4: role loop ends, after the system fence
  unsigned long long chunks;        // N1 / N3: finished chunks of chunkBytes
  int smId;                         // A10
  int kind;                         // A10: RoleKind
  int roleIndex;                    // A10: slice index inside the role
};
struct Config {
  int gpus[rankCount] = {4, 5, 6, 7};
  int kindA = kindIdle;             // role of slot A SMs
  int ctasA = 0;
  int rankMaskA = 0xf;              // ranks that run role A; E1.4 sets 0x3
  int kindB = kindIdle;             // role of slot B SMs
  int ctasB = 0;
  int rankMaskB = 0xf;              // E1.4 sets 0xc
  int threads = 512;
  unsigned long long windowNs = 300000000;
  int iterations = 5;
  int warmup = 1;
  int rankOffset = 0;               // E1.6c: NCCL's per-rank start offset pattern
};
struct KernelArguments {
  ncclWindow_t window;
  ncclDevComm deviceCommunicator;
  const float* localInput;
  float* localOutput;
  const int* kindOfSm;
  const int* slotOfSm;
  LocalControl* local;
  CtaRecord* records;
  int ctasA;
  int ctasB;
  int localActiveCtas;
  int rank;
  unsigned long long windowNs;
  size_t partStartOffset;           // E1.6c: added to every role address inside the rank's part
};

__host__ __device__ __forceinline__ size_t rankPartOffset(int rank) {
  return size_t(rank) * rankPartBytes;
}
__host__ __device__ __forceinline__ size_t startOffsetOfRank(int rank, int enabled) {
  return enabled ? size_t(rank ^ ((rank >> 1) & 1)) * rankOffsetUnitBytes : 0;
}
// Input word values. Whole small numbers stay exact as float, also after a sum over four ranks.
__host__ __device__ __forceinline__ float inputValue(size_t wordOffset, int rank) {
  return float(wordOffset % patternPeriodWords + rank);
}
__host__ __device__ __forceinline__ float sumValue(size_t wordOffset) {
  return float(4 * (wordOffset % patternPeriodWords) + (0 + 1 + 2 + 3));
}
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
__device__ __forceinline__ WindowControl* windowControl(ncclWindow_t window, int rank) {
  return static_cast<WindowControl*>(ncclGetLsaPointer(window, regionBytes * 2, rank));
}

// The four 16-byte memory operations under test. Written as instructions so that the width cannot
// change between roles. The multimem forms are the ones in nccl_device/impl/multimem__funcs.h.
__device__ __forceinline__ float4 loadReducePack(uintptr_t address) {
  float4 value;
  asm volatile("multimem.ld_reduce.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
               : "l"(address)
               : "memory");
  return value;
}
__device__ __forceinline__ void storeMulticastPack(uintptr_t address, float4 value) {
  asm volatile("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};"
               :: "l"(address), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
               : "memory");
}
__device__ __forceinline__ float4 loadPlainPack(uintptr_t address) {
  float4 value;
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
               : "l"(address)
               : "memory");
  return value;
}
__device__ __forceinline__ void storePlainPack(uintptr_t address, float4 value) {
  asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
               :: "l"(address), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
               : "memory");
}

// Addresses for one thread and one step inside a chunk. Each warp covers 8 × 32 × 16 bytes without
// gaps and lanes are 16 bytes apart, as in NCCL's deep loops, so stores can merge the same way.
template <typename Step>
__device__ __forceinline__ void runChunk(size_t chunkOffset, Step const& step) {
  const size_t threadStepBytes = size_t(packsPerStep) * packBytes;
  const size_t stepsInChunk = chunkBytes / (threadStepBytes * blockDim.x);
  const size_t warp = threadIdx.x / warpThreads;
  const size_t lane = threadIdx.x % warpThreads;
  const size_t warpStepBytes = threadStepBytes * warpThreads;
  const size_t threadBase = chunkOffset + warp * warpStepBytes + lane * packBytes;
  const size_t stepStride = warpStepBytes * (blockDim.x / warpThreads);
  #pragma unroll 1
  for (size_t stepIndex = 0; stepIndex < stepsInChunk; ++stepIndex) {
    step(threadBase + stepIndex * stepStride);
  }
}
// Loads run before stores in every role, as NCCL's kernels do; 8 packs are in flight per thread.
__device__ __forceinline__ void stepMulticast(uintptr_t sourceBase, uintptr_t multimemBase, size_t offset) {
  float4 values[packsPerStep];
  #pragma unroll
  for (int pack = 0; pack < packsPerStep; ++pack) {
    values[pack] = loadPlainPack(sourceBase + offset + size_t(pack) * warpThreads * packBytes);
  }
  #pragma unroll
  for (int pack = 0; pack < packsPerStep; ++pack) {
    storeMulticastPack(multimemBase + offset + size_t(pack) * warpThreads * packBytes, values[pack]);
  }
}
__device__ __forceinline__ void stepReduce(uintptr_t multimemBase, uintptr_t destinationBase, size_t offset) {
  float4 values[packsPerStep];
  #pragma unroll
  for (int pack = 0; pack < packsPerStep; ++pack) {
    values[pack] = loadReducePack(multimemBase + offset + size_t(pack) * warpThreads * packBytes);
  }
  #pragma unroll
  for (int pack = 0; pack < packsPerStep; ++pack) {
    storePlainPack(destinationBase + offset + size_t(pack) * warpThreads * packBytes, values[pack]);
  }
}
__device__ __forceinline__ void stepAllreduce(uintptr_t inputBase, uintptr_t outputBase, size_t offset) {
  float4 values[packsPerStep];
  #pragma unroll
  for (int pack = 0; pack < packsPerStep; ++pack) {
    values[pack] = loadReducePack(inputBase + offset + size_t(pack) * warpThreads * packBytes);
  }
  #pragma unroll
  for (int pack = 0; pack < packsPerStep; ++pack) {
    storeMulticastPack(outputBase + offset + size_t(pack) * warpThreads * packBytes, values[pack]);
  }
}
__device__ __forceinline__ void stepPlainCopy(uintptr_t sourceBase, uintptr_t destinationBase, size_t offset) {
  float4 values[packsPerStep];
  #pragma unroll
  for (int pack = 0; pack < packsPerStep; ++pack) {
    values[pack] = loadPlainPack(sourceBase + offset + size_t(pack) * warpThreads * packBytes);
  }
  #pragma unroll
  for (int pack = 0; pack < packsPerStep; ++pack) {
    storePlainPack(destinationBase + offset + size_t(pack) * warpThreads * packBytes, values[pack]);
  }
}

// Controller CTA, one per GPU: joins all role CTAs of all GPUs into one start and stops them after
// the window. The window is timed on this GPU; start signals reach peers within microseconds, which
// is very small against the 300 ms window. Raw timers are never compared across GPUs.
__device__ void runController(KernelArguments const& arguments) {
  if (threadIdx.x != 0) {
    return;
  }
  WindowControl* ownSignals = windowControl(arguments.window, arguments.rank);
  WindowControl* firstRankSignals = windowControl(arguments.window, 0);
  while (acquireSignal(&arguments.local->arrivedCtas) != arguments.localActiveCtas) {
    __nanosleep(64);
  }
  releaseSignal(&firstRankSignals->rankReady[arguments.rank].value, 1);
  if (arguments.rank == 0) {
    for (int rank = 0; rank < rankCount; ++rank) {
      while (!acquireSignal(&ownSignals->rankReady[rank].value)) {
        __nanosleep(64);
      }
    }
    for (int rank = 0; rank < rankCount; ++rank) {
      releaseSignal(&windowControl(arguments.window, rank)->startAll.value, 1);
    }
  }
  while (!acquireSignal(&ownSignals->startAll.value)) {
    __nanosleep(64);
  }
  CtaRecord& record = arguments.records[blockIdx.x];
  record.startNs = readTimerNs();
  releaseSignal(&arguments.local->startFlag, 1);
  while (readTimerNs() - record.startNs < arguments.windowNs) {
    __nanosleep(1024);
  }
  releaseSignal(&arguments.local->stopFlag, 1);
  while (acquireSignal(&arguments.local->finishedCtas) != arguments.localActiveCtas) {
    __nanosleep(64);
  }
  record.endNs = readTimerNs();
}

// One role loop over this CTA's slice of the rank's part. Only thread 0 touches the stop flag, and it
// passes the value through shared memory one chunk later (two slots, so a slot is never rewritten
// while a slower warp may still read it). Thread 0 checks once per chunk, so the check costs no
// memory round trip per pack. A CTA stops at most two chunks after the stop flag.
template <typename Step>
__device__ void runRoleLoop(KernelArguments const& arguments, int ctasInRole, int roleIndex, Step const& step) {
  __shared__ int stopSlots[2];
  CtaRecord& record = arguments.records[blockIdx.x];
  const size_t sliceChunks = rankPartBytes / chunkBytes / ctasInRole;
  const size_t sliceBase = size_t(roleIndex) * sliceChunks * chunkBytes;
  if (threadIdx.x == 0) {
    stopSlots[0] = 0;
    stopSlots[1] = 0;
    cuda::atomic_ref<int, cuda::thread_scope_system>(arguments.local->arrivedCtas).fetch_add(1, cuda::memory_order_release);
    while (!acquireSignal(&arguments.local->startFlag)) {
      __nanosleep(64);
    }
    record.startNs = readTimerNs();
  }
  __syncthreads();
  int pendingStop = 0;
  unsigned long long chunks = 0;
  for (;; ++chunks) {
    __syncthreads();
    if (stopSlots[(chunks + 1) & 1]) {
      break;
    }
    if (threadIdx.x == 0) {
      stopSlots[chunks & 1] = pendingStop;
      pendingStop = *reinterpret_cast<volatile int*>(&arguments.local->stopFlag);
    }
    size_t chunkOffset = (sliceBase + (chunks % sliceChunks) * chunkBytes + arguments.partStartOffset) % rankPartBytes;
    runChunk(chunkOffset, step);
  }
  // Posted stores count only after they are ordered, so the end time follows the system fence.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    record.endNs = readTimerNs();
    record.chunks = chunks;
    cuda::atomic_ref<int, cuda::thread_scope_system>(arguments.local->finishedCtas).fetch_add(1, cuda::memory_order_release);
  }
}

// Every role reads and writes only inside the issuing rank's part, so ranks never share an address.
__device__ void runRole(KernelArguments const& arguments, int kind, int startNsctasInRole, int roleIndex) {
  const size_t part = rankPartOffset(arguments.rank);
  const int nextRank = (arguments.rank + 1) % rankCount;
  const uintptr_t localSource = reinterpret_cast<uintptr_t>(arguments.localInput);
  const uintptr_t localDestination = reinterpret_cast<uintptr_t>(arguments.localOutput);
  if (kind == kindMulticast) {
    const uintptr_t output = reinterpret_cast<uintptr_t>(
      ncclGetLsaMultimemPointer(arguments.window, regionBytes + part, arguments.deviceCommunicator)
    );
    
    runRoleLoop(
      arguments, ctasInRole, roleIndex, [=] __device__(size_t offset) { 
        stepMulticast(localSource, output, offset); 
      }
    );
  } else if (kind == kindReduce) {
    const uintptr_t input = reinterpret_cast<uintptr_t>(
      ncclGetLsaMultimemPointer(arguments.window, part, arguments.deviceCommunicator)
    );
    runRoleLoop(arguments, ctasInRole, roleIndex, [=] __device__(size_t offset) { stepReduce(input, localDestination, offset); });
  } else if (kind == kindAllreduce) {
    const uintptr_t input = reinterpret_cast<uintptr_t>(ncclGetLsaMultimemPointer(arguments.window, part, arguments.deviceCommunicator));
    const uintptr_t output = reinterpret_cast<uintptr_t>(ncclGetLsaMultimemPointer(arguments.window, regionBytes + part, arguments.deviceCommunicator));
    runRoleLoop(arguments, ctasInRole, roleIndex, [=] __device__(size_t offset) { stepAllreduce(input, output, offset); });
  } else if (kind == kindPlainLoad) {
    const uintptr_t peerInput = reinterpret_cast<uintptr_t>(ncclGetLsaPointer(arguments.window, part, nextRank));
    runRoleLoop(arguments, ctasInRole, roleIndex, [=] __device__(size_t offset) { stepPlainCopy(peerInput, localDestination, offset); });
  } else if (kind == kindPlainStore) {
    const uintptr_t peerOutput = reinterpret_cast<uintptr_t>(ncclGetLsaPointer(arguments.window, regionBytes + part, nextRank));
    runRoleLoop(arguments, ctasInRole, roleIndex, [=] __device__(size_t offset) { stepPlainCopy(localSource, peerOutput, offset); });
  }
}

__global__ void __launch_bounds__(maxThreadsPerCta, 1) mmMix(KernelArguments arguments) {
  __shared__ int sharedRoleIndex;
  const int smId = readSmId();
  const int kind = arguments.kindOfSm[smId];
  const int slot = arguments.slotOfSm[smId];
  if (threadIdx.x == 0) {
    CtaRecord& record = arguments.records[blockIdx.x];
    record.smId = smId;
    record.kind = kind;
    sharedRoleIndex = kind >= kindMulticast ? atomicAdd(&arguments.local->registered[slot], 1) : -1;
    record.roleIndex = sharedRoleIndex;
  }
  __syncthreads();

  if (kind == kindController) {
    runController(arguments);
  } else if (kind >= kindMulticast) {
    runRole(arguments, kind, slot == 0 ? arguments.ctasA : arguments.ctasB, sharedRoleIndex);
  }
}

// Fills one region with inputValue. Runs before the first window and is not timed.
__global__ void fillPattern(float* region, size_t wordCount, size_t firstWord, int rank) {
  for (size_t word = size_t(blockIdx.x) * blockDim.x + threadIdx.x; word < wordCount; word += size_t(blockDim.x) * gridDim.x) {
    region[word] = inputValue(firstWord + word, rank);
  }
}
// A11. Checks every word that a role wrote in this window. A slice whose CTA wrote more chunks than
// the slice holds was rewritten fully; otherwise only its first chunks hold data. The value depends
// only on the position and role, so it does not change between passes of the loop.
__global__ void verifyPart(const float* part, const unsigned long long* chunksOfCta, int ctasInRole, size_t partStartOffset, size_t firstWord, int expectSum, int valueRank, unsigned long long* mismatches) {
  const size_t sliceChunks = rankPartBytes / chunkBytes / ctasInRole;
  const size_t wordsInPart = rankPartBytes / sizeof(float);
  const size_t wordsInChunk = chunkBytes / sizeof(float);
  unsigned long long localMismatches = 0;
  for (size_t logicalWord = size_t(blockIdx.x) * blockDim.x + threadIdx.x; logicalWord < wordsInPart; logicalWord += size_t(blockDim.x) * gridDim.x) {
    size_t chunk = logicalWord / wordsInChunk;
    size_t cta = chunk / sliceChunks;
    if (cta >= size_t(ctasInRole)) {
      continue;
    }
    size_t writtenChunks = chunksOfCta[cta] < sliceChunks ? chunksOfCta[cta] : sliceChunks;
    if (chunk % sliceChunks >= writtenChunks) {
      continue;
    }
    size_t word = (logicalWord + partStartOffset / sizeof(float)) % wordsInPart;
    float expected = expectSum ? sumValue(firstWord + word) : inputValue(firstWord + word, valueRank);
    localMismatches += part[word] != expected;
  }
  if (localMismatches) {
    atomicAdd(mismatches, localMismatches);
  }
}

struct WindowResult {
  std::vector<CtaRecord> records;
  std::vector<unsigned long long> chunksA;
  std::vector<unsigned long long> chunksB;
  unsigned long long mismatches = 0;
};
struct RankState {
  ncclComm_t communicator;
  ncclDevComm deviceCommunicator;
  ncclWindow_t window;
  void* windowMemory;
  float* localInput;
  float* localOutput;
  int* kindOfSm;
  int* slotOfSm;
  LocalControl* local;
  CtaRecord* deviceRecords;
  unsigned long long* deviceChunks;
  unsigned long long* mismatches;
  cudaStream_t stream;
  int gridCtas;
  size_t dynamicSharedBytes;
  cudaDeviceProp properties;
  std::vector<int> kindTable;
  std::vector<int> slotTable;
  std::vector<int> gpcOfSm;
  std::string gpcMapUuid;
  int activeCtas;
  std::vector<WindowResult> windows;
};
struct Shared {
  Config config;
  std::string gpcMapDirectory;
  std::string outputPath;
  ncclUniqueId identifier;
  pthread_barrier_t hostBarrier;
  RankState ranks[rankCount];
};

// User allocation for plan 01: physical GPUs 4-7 from 2026-11-01.
static bool authorizedGpu(int gpu) {
  return gpu >= 4 && gpu <= 7;
}
static int parseKind(const std::string& text) {
  for (int kind = 0; kind < int(sizeof(roleNames) / sizeof(roleNames[0])); ++kind) {
    if (text == roleNames[kind]) {
      return kind;
    }
  }
  fprintf(stderr, "Unknown role %s\n", text.c_str());
  std::exit(1);
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
    std::string value = line.substr(separator + 1);
    if (key == "role_a") {
      config.kindA = parseKind(value);
    } else if (key == "role_b") {
      config.kindB = parseKind(value);
    } else if (key == "ctas_a") {
      config.ctasA = std::stoi(value);
    } else if (key == "ctas_b") {
      config.ctasB = std::stoi(value);
    } else if (key == "rank_mask_a") {
      config.rankMaskA = std::stoi(value);
    } else if (key == "rank_mask_b") {
      config.rankMaskB = std::stoi(value);
    } else if (key == "threads") {
      config.threads = std::stoi(value);
    } else if (key == "window_ms") {
      config.windowNs = std::stoull(value) * 1000000ull;
    } else if (key == "iterations") {
      config.iterations = std::stoi(value);
    } else if (key == "warmup") {
      config.warmup = std::stoi(value);
    } else if (key == "rank_offset") {
      config.rankOffset = std::stoi(value);
    } else {
      fprintf(stderr, "Unknown config key %s\n", key.c_str());
      std::exit(1);
    }
  }
  return config;
}
static void validateConfiguration(const Config& config) {
  bool threadsValid = config.threads >= 32 && config.threads % 32 == 0 && config.threads <= maxThreadsPerCta && chunkBytes % (size_t(packsPerStep) * packBytes * config.threads) == 0;
  // Slot A and slot B take turns over the SMs, so each role can have at most 64 CTAs.
  bool ctasValid = config.ctasA >= 0 && config.ctasA <= 64 && config.ctasB >= 0 && config.ctasB <= 64 && config.ctasA + config.ctasB >= 1;
  bool kindsValid = (config.ctasA == 0 || config.kindA >= kindMulticast) && (config.ctasB == 0 || config.kindB >= kindMulticast);
  bool runValid = config.iterations >= 1 && config.warmup >= 0 && config.windowNs > 0;
  bool gpusValid = true;
  for (int rank = 0; rank < rankCount; ++rank) {
    gpusValid = gpusValid && authorizedGpu(config.gpus[rank]);
  }
  if (!threadsValid || !ctasValid || !kindsValid || !runValid || !gpusValid) {
    fprintf(stderr, "Invalid configuration (threads %d, ctas %d, kinds %d, run %d, gpus %d)\n", threadsValid, ctasValid, kindsValid, runValid, gpusValid);
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
// Reads the map from plan 07 (metric A12): {"uuid": "...", ..., "gpc_of_sm": [g0, g1, ...]}.
static void loadGpcMap(RankState& state, const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    fprintf(stderr, "Cannot open GPC map %s\n", path.c_str());
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
  state.gpcMapUuid = text.substr(uuidBegin, text.find('"', uuidBegin) - uuidBegin);
  std::stringstream list(text.substr(listKey + strlen("\"gpc_of_sm\": ["), text.find(']', listKey) - listKey - strlen("\"gpc_of_sm\": [")));
  std::string entry;
  while (std::getline(list, entry, ',')) {
    state.gpcOfSm.push_back(std::stoi(entry));
  }
}
// SM order: one SM from each GPC in turn, unmapped SMs last. The order does not depend on the arm:
// position 0 is the controller, then odd positions are slot A and even positions slot B. So a role
// keeps the same SMs in every arm, both slots are spread over the GPCs, and one GPC's store limit
// (about 700-800 GB/s, plan 07) is not a limit under test here.
static void buildRoleTable(RankState& state, const Config& config, int rank) {
  const int smCount = int(state.gpcOfSm.size());
  int gpcCount = *std::max_element(state.gpcOfSm.begin(), state.gpcOfSm.end()) + 1;
  std::vector<std::vector<int>> smsOfGpc(gpcCount);
  std::vector<int> unmappedSms;
  for (int smId = 0; smId < smCount; ++smId) {
    if (state.gpcOfSm[smId] < 0) {
      unmappedSms.push_back(smId);
    } else {
      smsOfGpc[state.gpcOfSm[smId]].push_back(smId);
    }
  }
  std::vector<int> order;
  for (size_t round = 0; int(order.size()) + int(unmappedSms.size()) < smCount; ++round) {
    for (int gpc = 0; gpc < gpcCount; ++gpc) {
      if (round < smsOfGpc[gpc].size()) {
        order.push_back(smsOfGpc[gpc][round]);
      }
    }
  }
  order.insert(order.end(), unmappedSms.begin(), unmappedSms.end());
  state.kindTable.assign(maxSms, kindIdle);
  state.slotTable.assign(maxSms, 0);
  state.kindTable[order[0]] = kindController;
  state.activeCtas = 0;
  const int localCtasA = config.rankMaskA >> rank & 1 ? config.ctasA : 0;
  const int localCtasB = config.rankMaskB >> rank & 1 ? config.ctasB : 0;
  for (int index = 0; index < localCtasA; ++index) {
    state.kindTable[order[1 + 2 * index]] = config.kindA;
    state.slotTable[order[1 + 2 * index]] = 0;
  }
  for (int index = 0; index < localCtasB; ++index) {
    state.kindTable[order[2 + 2 * index]] = config.kindB;
    state.slotTable[order[2 + 2 * index]] = 1;
  }
  state.activeCtas = localCtasA + localCtasB;
}
// Runs fillPattern over one region on this rank's stream.
static void fillRegion(RankState& state, float* region, size_t bytes, size_t firstWord, int rank, int blockThreads) {
  const size_t words = bytes / sizeof(float);
  fillPattern<<<state.gridCtas, blockThreads, 0, state.stream>>>(region, words, firstWord, rank);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
static void setupRank(Shared& shared, int rank) {
  RankState& state = shared.ranks[rank];
  const Config& config = shared.config;
  const int device = config.gpus[rank];
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaGetDeviceProperties(&state.properties, device));
  loadGpcMap(state, shared.gpcMapDirectory + "/gpu" + std::to_string(device) + ".json");
  if (state.gpcMapUuid != uuidText(state.properties.uuid) || int(state.gpcOfSm.size()) != state.properties.multiProcessorCount || state.properties.multiProcessorCount > maxSms) {
    fprintf(stderr, "GPC map does not belong to GPU %d (%s)\n", device, uuidText(state.properties.uuid).c_str());
    std::exit(1);
  }
  buildRoleTable(state, config, rank);
  NCCL_CHECK(ncclCommInitRank(&state.communicator, rankCount, shared.identifier, rank));
  ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_CHECK(ncclCommQueryProperties(state.communicator, &properties));
  if (!properties.deviceApiSupport || properties.nLsaTeams != 1 || !properties.multimemSupport) {
    fprintf(stderr, "Requires one LSA team with device API and multimem support\n");
    std::exit(1);
  }

  // Window: input region, output region, then control signals.
  const size_t windowBytes = regionBytes * 2 + sizeof(WindowControl);
  NCCL_CHECK(ncclMemAlloc(&state.windowMemory, windowBytes));
  NCCL_CHECK(ncclCommWindowRegister(state.communicator, state.windowMemory, windowBytes, &state.window, NCCL_WIN_COLL_SYMMETRIC));
  
  ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  // Every arm uses the same device communicator, also arms that issue no multimem operation.
  requirements.lsaMultimem = true;
  NCCL_CHECK(ncclDevCommCreate(state.communicator, &requirements, &state.deviceCommunicator));

  state.gridCtas = state.properties.multiProcessorCount;
  CUDA_CHECK(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&state.local, sizeof(LocalControl)));
  CUDA_CHECK(cudaMalloc(&state.deviceRecords, state.gridCtas * sizeof(CtaRecord)));
  CUDA_CHECK(cudaMalloc(&state.deviceChunks, maxSms * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&state.mismatches, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&state.kindOfSm, maxSms * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&state.slotOfSm, maxSms * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(state.kindOfSm, state.kindTable.data(), maxSms * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(state.slotOfSm, state.slotTable.data(), maxSms * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&state.localInput, rankPartBytes));
  CUDA_CHECK(cudaMalloc(&state.localOutput, rankPartBytes));
  // Input region: this rank's value at every position, for ld_reduce and plain loads from peers.
  fillRegion(state, static_cast<float*>(state.windowMemory), regionBytes, 0, rank, 256);
  // Local input: the values of this rank's part of the input region, for multicast and plain stores.
  fillRegion(state, state.localInput, rankPartBytes, rankPartOffset(rank) / sizeof(float), rank, 256);
  // More than half an SM's shared memory ensures no two CTAs share an SM.
  state.dynamicSharedBytes = ((state.properties.sharedMemPerMultiprocessor / 2 + 255) / 256) * 256;
  CUDA_CHECK(cudaFuncSetAttribute(mmMix, cudaFuncAttributeMaxDynamicSharedMemorySize, state.dynamicSharedBytes));
  int residentCtas = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residentCtas, mmMix, config.threads, state.dynamicSharedBytes));
  if (residentCtas != 1) {
    fprintf(stderr, "Expected exactly one resident CTA per SM, got %d\n", residentCtas);
    std::exit(1);
  }
  cudaFuncAttributes attributes;
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, mmMix));
  fprintf(stderr, "rank=%d gpu=%d SMs=%d A=%sx%d B=%sx%d threads=%d shared=%zu registers=%d local_bytes=%zu\n", rank, device, state.properties.multiProcessorCount, roleNames[config.kindA], config.ctasA, roleNames[config.kindB], config.ctasB, config.threads, state.dynamicSharedBytes, attributes.numRegs, attributes.localSizeBytes);
}
// Poison what roles write, so an unwritten or misplaced word fails the check after the window.
static void prepareWindow(RankState& state) {
  CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(state.windowMemory) + regionBytes, 0xff, regionBytes, state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.localOutput, 0xff, rankPartBytes, state.stream));
  CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(state.windowMemory) + regionBytes * 2, 0, sizeof(WindowControl), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.local, 0, sizeof(LocalControl), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.deviceRecords, 0, state.gridCtas * sizeof(CtaRecord), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.mismatches, 0, sizeof(unsigned long long), state.stream));
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
// CTA chunk counts of one role, ordered by slice index, as the check kernel needs them.
static std::vector<unsigned long long> chunksByRoleIndex(const std::vector<CtaRecord>& records, int kind, int ctasInRole) {
  std::vector<unsigned long long> chunks(maxSms, 0);
  if (kind < kindMulticast) {
    return chunks;
  }
  std::set<int> indexes;
  for (const auto& record : records) {
    if (record.kind == kind) {
      if (record.roleIndex < 0 || record.roleIndex >= ctasInRole || !indexes.insert(record.roleIndex).second || record.chunks == 0) {
        fprintf(stderr, "Invalid role index or a CTA with no finished chunk\n");
        std::exit(4);
      }
      chunks[record.roleIndex] = record.chunks;
    }
  }
  return chunks;
}
static void checkPlacement(const RankState& state, const std::vector<CtaRecord>& records) {
  std::set<int> usedSms;
  for (const auto& record : records) {
    if (!usedSms.insert(record.smId).second) {
      fprintf(stderr, "Two CTAs on one SM\n");
      std::exit(4);
    }
    if (record.kind != state.kindTable[record.smId]) {
      fprintf(stderr, "Role does not match the table\n");
      std::exit(4);
    }
    if (record.kind >= kindMulticast && record.endNs <= record.startNs) {
      fprintf(stderr, "Invalid CTA timing\n");
      std::exit(4);
    }
  }
}
static void runVerify(RankState& state, const float* region, const std::vector<unsigned long long>& chunks, int ctasInRole, size_t partOffsetValue, size_t firstWord, int expectSum, int valueRank) {
  CUDA_CHECK(cudaMemcpy(state.deviceChunks, chunks.data(), maxSms * sizeof(unsigned long long), cudaMemcpyHostToDevice));
  verifyPart<<<state.gridCtas, 256, 0, state.stream>>>(region, state.deviceChunks, ctasInRole, partOffsetValue, firstWord, expectSum, valueRank, state.mismatches);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
// Output region on this GPU, one part per issuing rank that writes here.
static void verifyOutputRole(Shared& shared, RankState& state, int rank, int kind, int ctasInRole, int rankMask, const std::vector<std::vector<unsigned long long>>& chunksOfRank) {
  const char* outputBase = static_cast<const char*>(state.windowMemory) + regionBytes;
  const bool multimemOutput = kind == kindMulticast || kind == kindAllreduce;
  for (int issuer = 0; issuer < rankCount; ++issuer) {
    bool writesHere = multimemOutput || (kind == kindPlainStore && (issuer + 1) % rankCount == rank);
    if (!writesHere || !(rankMask >> issuer & 1)) {
      continue;
    }
    size_t offset = startOffsetOfRank(issuer, shared.config.rankOffset);
    runVerify(state, reinterpret_cast<const float*>(outputBase + rankPartOffset(issuer)), chunksOfRank[issuer], ctasInRole, offset, rankPartOffset(issuer) / sizeof(float), kind == kindAllreduce, issuer);
  }
}
// Local output of this rank: the sum (reduce) or the next rank's value (plain load).
static void verifyLocalRole(Shared& shared, RankState& state, int rank, int kind, int ctasInRole, int rankMask, const std::vector<std::vector<unsigned long long>>& chunksOfRank) {
  if ((kind != kindReduce && kind != kindPlainLoad) || !(rankMask >> rank & 1)) {
    return;
  }
  size_t offset = startOffsetOfRank(rank, shared.config.rankOffset);
  runVerify(state, state.localOutput, chunksOfRank[rank], ctasInRole, offset, rankPartOffset(rank) / sizeof(float), kind == kindReduce, (rank + 1) % rankCount);
}
// Each window: reset, launch, read records, exchange chunk counts between ranks, then check buffers.
static void runWindow(Shared& shared, int rank, int iteration) {
  RankState& state = shared.ranks[rank];
  const Config& config = shared.config;

  prepareWindow(state);
  pthread_barrier_wait(&shared.hostBarrier);

  KernelArguments arguments{
    state.window, state.deviceCommunicator, state.localInput, state.localOutput, 
    state.kindOfSm, state.slotOfSm, state.local, state.deviceRecords, 
    config.ctasA, config.ctasB, 
    state.activeCtas, rank, config.windowNs, startOffsetOfRank(rank, config.rankOffset)
  };
  mmMix<<<state.gridCtas, config.threads, state.dynamicSharedBytes, state.stream>>>(arguments);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(state.stream));

  WindowResult result;
  result.records.resize(state.gridCtas);
  CUDA_CHECK(cudaMemcpy(result.records.data(), state.deviceRecords, state.gridCtas * sizeof(CtaRecord), cudaMemcpyDeviceToHost));
  checkPlacement(state, result.records);

  result.chunksA = chunksByRoleIndex(result.records, config.kindA, config.ctasA);
  result.chunksB = chunksByRoleIndex(result.records, config.kindB, config.ctasB);
  state.windows.push_back(result);
  pthread_barrier_wait(&shared.hostBarrier);

  std::vector<std::vector<unsigned long long>> chunksA(rankCount);
  std::vector<std::vector<unsigned long long>> chunksB(rankCount);
  for (int other = 0; other < rankCount; ++other) {
    chunksA[other] = shared.ranks[other].windows.back().chunksA;
    chunksB[other] = shared.ranks[other].windows.back().chunksB;
  }
  if (config.ctasA) {
    verifyOutputRole(shared, state, rank, config.kindA, config.ctasA, config.rankMaskA, chunksA);
    verifyLocalRole(shared, state, rank, config.kindA, config.ctasA, config.rankMaskA, chunksA);
  }
  if (config.ctasB) {
    verifyOutputRole(shared, state, rank, config.kindB, config.ctasB, config.rankMaskB, chunksB);
    verifyLocalRole(shared, state, rank, config.kindB, config.ctasB, config.rankMaskB, chunksB);
  }
  CUDA_CHECK(cudaMemcpy(&state.windows.back().mismatches, state.mismatches, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  
  if (state.windows.back().mismatches) {
    fprintf(stderr, "rank %d window %d: %llu wrong words\n", rank, iteration, state.windows.back().mismatches);
    std::exit(5);
  }
  pthread_barrier_wait(&shared.hostBarrier);
}
static void rankThread(Shared* shared, int rank) {
  setupRank(*shared, rank);
  pthread_barrier_wait(&shared->hostBarrier);
  for (int iteration = 0; iteration < shared->config.warmup + shared->config.iterations; ++iteration) {
    runWindow(*shared, rank, iteration);
  }
}
// Output keys are an interface: analysis scripts read them, so they must not change with code names.
static void writeRecords(const Shared& shared) {
  std::ofstream output(shared.outputPath);
  const Config& config = shared.config;
  output << "{\"schema\":\"mm_mix_v1\",\"role_a\":\"" << roleNames[config.kindA] << "\",\"ctas_a\":" << config.ctasA << ",\"rank_mask_a\":" << config.rankMaskA;
  output << ",\"role_b\":\"" << roleNames[config.kindB] << "\",\"ctas_b\":" << config.ctasB << ",\"rank_mask_b\":" << config.rankMaskB;
  output << ",\"threads\":" << config.threads << ",\"window_ns\":" << config.windowNs << ",\"chunk_bytes\":" << chunkBytes << ",\"pack_bytes\":" << packBytes << ",\"packs_per_step\":" << packsPerStep << ",\"rank_offset\":" << config.rankOffset << ",\"warmup\":" << config.warmup << ",\"ranks\":[";
  for (int rank = 0; rank < rankCount; ++rank) {
    const RankState& state = shared.ranks[rank];
    output << (rank ? "," : "") << "{\"rank\":" << rank << ",\"gpu\":" << config.gpus[rank] << ",\"uuid\":\"" << uuidText(state.properties.uuid) << "\",\"shared_bytes\":" << state.dynamicSharedBytes << ",\"windows\":[";
    for (size_t window = 0; window < state.windows.size(); ++window) {
      output << (window ? "," : "") << "{\"mismatches\":" << state.windows[window].mismatches << ",\"ctas\":[";
      bool first = true;
      for (const auto& record : state.windows[window].records) {
        if (record.kind == kindIdle) {
          continue;
        }
        output << (first ? "" : ",") << "{\"role\":\"" << roleNames[record.kind] << "\",\"sm_id\":" << record.smId << ",\"role_index\":" << record.roleIndex << ",\"start_ns\":" << record.startNs << ",\"end_ns\":" << record.endNs << ",\"chunks\":" << record.chunks << "}";
        first = false;
      }
      output << "]}";
    }
    output << "]}";
  }
  output << "]}\n";
}
int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s CONFIG OUTPUT_JSON GPC_MAP_DIRECTORY\n", argv[0]);
    return 1;
  }
  static Shared shared;
  shared.config = parseConfiguration(argv[1]);
  validateConfiguration(shared.config);
  shared.outputPath = argv[2];
  shared.gpcMapDirectory = argv[3];
  NCCL_CHECK(ncclGetUniqueId(&shared.identifier));
  pthread_barrier_init(&shared.hostBarrier, nullptr, rankCount);
  std::vector<std::thread> threads;
  for (int rank = 0; rank < rankCount; ++rank) {
    threads.emplace_back(rankThread, &shared, rank);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  writeRecords(shared);
  return 0;
}
