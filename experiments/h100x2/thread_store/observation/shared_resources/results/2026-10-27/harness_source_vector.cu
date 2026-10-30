// Thread-issued NCCL LSA push with spatially separated co-runner roles.
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
#include <fstream>
#include <map>
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

constexpr int maxCommunicationCtas = 64;
constexpr int checkpointCount = 8;
constexpr size_t pressureBytes = size_t(1) << 30;
struct alignas(128) Signal { int value; };
struct WindowControl {
  Signal destinationReady;
  Signal acknowledgement;
  Signal completion[maxCommunicationCtas];
};
struct LocalControl {
  int readyCount;
  int finishedCount;
  int stop;
};
struct CtaRecord {
  unsigned long long startNs;       // A2 / A8
  unsigned long long issueEndNs;    // A3
  unsigned long long fenceEndNs;    // A4
  unsigned long long publishEndNs;  // A5
  unsigned long long endNs;         // A1 / A6 / A8
  unsigned long long workUnits;     // A9 logical 256 KiB bursts or 256 FMA rounds
  unsigned long long checkpointsNs[checkpointCount]; // A7 issue progress, not arrival
  double checksum;
  int smId;                        // A10
  int role;                        // 0 communication/control; 1 co-runner
};
struct Config {
  int sourceGpu = 4;
  int destinationGpu = 5;
  int communicationCtas = 32;
  int threadsPerCta = 256;
  int coRunnerCtas = 0;
  int pressureKind = 0; // 0 none, 1 arithmetic, 2 source reads, 3 destination reads
  int pacingRounds = 0;
  int iterations = 12;
  int warmup = 3;
  int detailed = 1;
  int globalStride = 0;
  int communicationEnabled = 1;
  int vectorizedReads = 0;
  size_t payloadBytes = size_t(256) << 20;
  unsigned long long soloDurationNs = 20000000;
};
struct KernelArguments {
  ncclWindow_t window;
  uint32_t* source;
  uint32_t* pressure;
  size_t elementCount;
  LocalControl* local;
  CtaRecord* records;
  Config config;
  int rank;
  int iteration;
  int localCoRunnerCtas;
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
__device__ __forceinline__ uint32_t expectedValue(size_t offset, int iteration) {
  return uint32_t(offset) * 1664525u ^ uint32_t(iteration + 1) * 1013904223u;
}
__global__ void initializeSource(uint32_t* source, size_t elementCount, int iteration) {
  for (size_t offset = size_t(blockIdx.x) * blockDim.x + threadIdx.x; offset < elementCount; offset += size_t(blockDim.x) * gridDim.x) {
    source[offset] = expectedValue(offset, iteration);
  }
}
__global__ void verifyDestination(uint32_t* destination, size_t elementCount, int iteration, unsigned long long* mismatches) {
  unsigned long long localMismatches = 0;
  for (size_t offset = size_t(blockIdx.x) * blockDim.x + threadIdx.x; offset < elementCount; offset += size_t(blockDim.x) * gridDim.x) {
    localMismatches += destination[offset] != expectedValue(offset, iteration);
  }
  if (localMismatches) {
    atomicAdd(mismatches, localMismatches);
  }
}

// Only one lane polls each control location, bounding control traffic.
__device__ void waitForLocalCoRunners(KernelArguments arguments) {
  if (threadIdx.x == 0) {
    while (acquireSignal(&arguments.local->readyCount) != arguments.localCoRunnerCtas) {
      __nanosleep(64);
    }
  }
  __syncthreads();
}
__device__ void runCommunication(KernelArguments arguments) {
  WindowControl* localSignals = windowControl(arguments.window, arguments.config.payloadBytes, 0);
  WindowControl* remoteSignals = windowControl(arguments.window, arguments.config.payloadBytes, 1);
  auto* destination = static_cast<uint32_t*>(ncclGetLsaPointer(arguments.window, 0, 1));
  CtaRecord& record = arguments.records[blockIdx.x];
  waitForLocalCoRunners(arguments);
  if (threadIdx.x == 0) {
    while (!acquireSignal(&localSignals->destinationReady.value)) {
      __nanosleep(64);
    }
    record.startNs = readTimerNs();
  }
  __syncthreads();
  size_t regionBegin = arguments.elementCount * blockIdx.x / arguments.config.communicationCtas;
  size_t regionEnd = arguments.elementCount * (blockIdx.x + 1) / arguments.config.communicationCtas;
  size_t laneOffset = regionBegin + threadIdx.x;
  size_t laneStride = blockDim.x;
  if (arguments.config.globalStride) {
    laneOffset = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    laneStride = size_t(arguments.config.communicationCtas) * blockDim.x;
    regionEnd = arguments.elementCount;
  }
  size_t laneIterations = laneOffset < regionEnd ? (regionEnd - laneOffset + laneStride - 1) / laneStride : 0;
  size_t checkpointStride = (laneIterations + checkpointCount - 1) / checkpointCount;
  int checkpointIndex = 0;
  size_t progress = 0;
  size_t nextCheckpoint = checkpointStride;
  // Scalar coalesced local loads followed by scalar peer stores, no added chunk fences.
  for (size_t offset = laneOffset; offset < regionEnd; offset += laneStride) {
    destination[offset] = arguments.source[offset];
    ++progress;
    if (arguments.config.detailed && threadIdx.x == 0 && progress == nextCheckpoint && checkpointIndex < checkpointCount) {
      record.checkpointsNs[checkpointIndex++] = readTimerNs();
      nextCheckpoint += checkpointStride;
    }
  }
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
    releaseSignal(&remoteSignals->completion[blockIdx.x].value, 1);
    if (arguments.config.detailed) {
      record.publishEndNs = readTimerNs();
    }
    while (!acquireSignal(&localSignals->acknowledgement.value)) {
      __nanosleep(64);
    }
    record.endNs = readTimerNs();
    if (atomicAdd(&arguments.local->finishedCount, 1) + 1 == arguments.config.communicationCtas) {
      releaseSignal(&arguments.local->stop, 1);
    }
  }
}
__device__ void runReceiver(KernelArguments arguments) {
  WindowControl* localSignals = windowControl(arguments.window, arguments.config.payloadBytes, 1);
  WindowControl* remoteSignals = windowControl(arguments.window, arguments.config.payloadBytes, 0);
  waitForLocalCoRunners(arguments);
  if (threadIdx.x == 0) {
    CtaRecord& record = arguments.records[0];
    record.startNs = readTimerNs();
    releaseSignal(&remoteSignals->destinationReady.value, 1);
    for (int communicationCta = 0; communicationCta < arguments.config.communicationCtas; ++communicationCta) {
      while (!acquireSignal(&localSignals->completion[communicationCta].value)) {
        __nanosleep(64);
      }
    }
    record.issueEndNs = readTimerNs();
    releaseSignal(&remoteSignals->acknowledgement.value, 1);
    record.endNs = readTimerNs();
    releaseSignal(&arguments.local->stop, 1);
  }
}
__device__ __forceinline__ void arithmeticRound(float* accumulators) {
  #pragma unroll
  for (int accumulator = 0; accumulator < 8; ++accumulator) {
    accumulators[accumulator] = fmaf(accumulators[accumulator], 0.99999f, 0.00001f * (accumulator + 1));
  }
}
// Vector reads expose more independent memory requests without changing the copy.
// Four checksum chains and four-way loop unrolling avoid a scalar accumulation bottleneck.
__device__ float readVectorBurst(KernelArguments arguments, int coRunnerIndex, unsigned long long workUnits) {
  const size_t vectorElements = pressureBytes / sizeof(uint4);
  size_t regionBegin = vectorElements * coRunnerIndex / arguments.localCoRunnerCtas;
  size_t regionEnd = vectorElements * (coRunnerIndex + 1) / arguments.localCoRunnerCtas;
  regionBegin = (regionBegin + blockDim.x - 1) / blockDim.x * blockDim.x;
  regionEnd = regionEnd / blockDim.x * blockDim.x;
  size_t burstOffset = regionBegin + (workUnits * 16384) % (regionEnd - regionBegin);
  uint32_t checksumFirst = 0;
  uint32_t checksumSecond = 0;
  uint32_t checksumThird = 0;
  uint32_t checksumFourth = 0;
  const auto* vectors = reinterpret_cast<const uint4*>(arguments.pressure);
  #pragma unroll 4
  for (int vectorIndex = threadIdx.x; vectorIndex < 16384; vectorIndex += blockDim.x) {
    size_t offset = burstOffset + vectorIndex;
    if (offset >= regionEnd) {
      offset = regionBegin + offset - regionEnd;
    }
    uint4 values = vectors[offset];
    checksumFirst += values.x;
    checksumSecond += values.y;
    checksumThird += values.z;
    checksumFourth += values.w;
  }
  return (float(checksumFirst) + float(checksumSecond) + float(checksumThird) + float(checksumFourth)) * 1e-10f;
}

__device__ void runCoRunner(KernelArguments arguments, int coRunnerIndex) {
  CtaRecord& record = arguments.records[blockIdx.x];
  float accumulators[8];
  #pragma unroll
  for (int accumulator = 0; accumulator < 8; ++accumulator) {
    accumulators[accumulator] = float(threadIdx.x + accumulator + 1) * 0.001f;
  }
  __shared__ int stopRequested;
  unsigned long long workUnits = 0;
  const bool arithmeticOnly = arguments.config.pressureKind == 1;
  // Partition one 1 GiB array among active co-runner CTAs; wrap within each slice.
  const size_t pressureElements = pressureBytes / sizeof(uint32_t);
  size_t regionBegin = pressureElements * coRunnerIndex / arguments.localCoRunnerCtas;
  size_t regionEnd = pressureElements * (coRunnerIndex + 1) / arguments.localCoRunnerCtas;
  regionBegin = (regionBegin + blockDim.x - 1) / blockDim.x * blockDim.x;
  regionEnd = regionEnd / blockDim.x * blockDim.x;
  size_t burstOffset = regionBegin;
  if (threadIdx.x == 0) {
    record.startNs = readTimerNs();
    cuda::atomic_ref<int, cuda::thread_scope_system>(arguments.local->readyCount).fetch_add(1, cuda::memory_order_release);
  }
  __syncthreads();
  while (true) {
    if (arithmeticOnly) {
      #pragma unroll 1
      for (int round = 0; round < 256; ++round) {
        arithmeticRound(accumulators);
      }
    } else if (arguments.config.vectorizedReads) {
      accumulators[0] += readVectorBurst(arguments, coRunnerIndex, workUnits);
      for (int round = 0; round < arguments.config.pacingRounds; ++round) {
        arithmeticRound(accumulators);
      }
    } else {
      uint32_t checksum = 0;
      // Volatile scalar reads prevent reusing cached register values across passes.
      for (size_t burstElement = threadIdx.x; burstElement < 65536; burstElement += blockDim.x) {
        size_t offset = burstOffset + burstElement;
        if (offset >= regionEnd) {
          offset = regionBegin + offset - regionEnd;
        }
        checksum += reinterpret_cast<volatile uint32_t*>(arguments.pressure)[offset];
      }
      accumulators[0] += float(checksum) * 1e-10f;
      burstOffset = regionBegin + (burstOffset - regionBegin + 65536) % (regionEnd - regionBegin);
      for (int round = 0; round < arguments.config.pacingRounds; ++round) {
        arithmeticRound(accumulators);
      }
    }
    ++workUnits;
    if (threadIdx.x == 0) {
      stopRequested = acquireSignal(&arguments.local->stop);
    }
    __syncthreads();
    if (stopRequested) {
      break;
    }
  }
  // Every lane's checksum escapes so generated arithmetic and loads remain observable.
  __shared__ float laneChecksums[512];
  float checksum = 0;
  #pragma unroll
  for (int accumulator = 0; accumulator < 8; ++accumulator) {
    checksum += accumulators[accumulator];
  }
  laneChecksums[threadIdx.x] = checksum;
  __syncthreads();
  if (threadIdx.x == 0) {
    record.endNs = readTimerNs();
    record.workUnits = workUnits;
    record.checksum = 0;
    for (int lane = 0; lane < blockDim.x; ++lane) {
      record.checksum += laneChecksums[lane];
    }
  }
}
__device__ void runSoloControl(KernelArguments arguments) {
  waitForLocalCoRunners(arguments);
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
// Both endpoints use the same residency constraint, including matched no-pressure runs.
__global__ void fusedPush(KernelArguments arguments) {
  int controlCtas = arguments.rank == 0 ? arguments.config.communicationCtas : 1;
  if (threadIdx.x == 0) {
    arguments.records[blockIdx.x].smId = readSmId();
    arguments.records[blockIdx.x].role = blockIdx.x >= controlCtas;
  }
  if (blockIdx.x >= controlCtas) {
    runCoRunner(arguments, blockIdx.x - controlCtas);
  } else if (!arguments.config.communicationEnabled) {
    runSoloControl(arguments);
  } else if (arguments.rank == 0) {
    runCommunication(arguments);
  } else {
    runReceiver(arguments);
  }
}

struct RankState {
  ncclComm_t communicator;
  ncclDevComm deviceCommunicator;
  ncclWindow_t window;
  void* windowMemory;
  uint32_t* source;
  uint32_t* pressure;
  LocalControl* local;
  CtaRecord* deviceRecords;
  unsigned long long* mismatches;
  cudaStream_t stream;
  int gridCtas;
  int localCoRunnerCtas;
  size_t dynamicSharedBytes;
  cudaDeviceProp properties;
  std::vector<std::vector<CtaRecord>> records;
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
    auto separator = line.find('=');
    std::string key = line.substr(0, separator);
    int value = std::stoi(line.substr(separator + 1));
    if (key == "source_gpu") {
      config.sourceGpu = value;
    }
    else if (key == "destination_gpu") {
      config.destinationGpu = value;
    }
    else if (key == "ctas") {
      config.communicationCtas = value;
    }
    else if (key == "threads") {
      config.threadsPerCta = value;
    }
    else if (key == "co_ctas") {
      config.coRunnerCtas = value;
    }
    else if (key == "kind") {
      config.pressureKind = value;
    }
    else if (key == "pacing") {
      config.pacingRounds = value;
    }
    else if (key == "iterations") {
      config.iterations = value;
    }
    else if (key == "warmup") {
      config.warmup = value;
    }
    else if (key == "detailed") {
      config.detailed = value;
    }
    else if (key == "global_stride") {
      config.globalStride = value;
    }
    else if (key == "communication") {
      config.communicationEnabled = value;
    }
    else if (key == "vector_reads") {
      config.vectorizedReads = value;
    }
    else {
      fprintf(stderr, "Unknown config key %s\n", key.c_str());
      std::exit(1);
    }
  }
  if (config.sourceGpu < 4 || config.sourceGpu > 7 || config.destinationGpu < 4 || config.destinationGpu > 7 || config.sourceGpu == config.destinationGpu || config.communicationCtas < 1 || config.communicationCtas > maxCommunicationCtas || config.threadsPerCta < 64 || config.threadsPerCta > 512 || config.threadsPerCta % 32 || config.coRunnerCtas < 0 || config.pressureKind < 0 || config.pressureKind > 3 || config.iterations < 1 || config.pacingRounds < 0) {
    fprintf(stderr, "Invalid configuration or GPU outside authorized allocation 4-7\n");
    std::exit(1);
  }
  return config;
}
static void setupRank(RankState& state, Config config, int rank, ncclUniqueId identifier) {
  int device = rank == 0 ? config.sourceGpu : config.destinationGpu;
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaGetDeviceProperties(&state.properties, device));
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
  state.localCoRunnerCtas = ((rank == 0 && (config.pressureKind == 1 || config.pressureKind == 2)) || (rank == 1 && config.pressureKind == 3)) ? config.coRunnerCtas : 0;
  state.gridCtas = (rank == 0 ? config.communicationCtas : 1) + state.localCoRunnerCtas;
  if (state.gridCtas > state.properties.multiProcessorCount) {
    fprintf(stderr, "Grid exceeds one-CTA-per-SM residency\n");
    std::exit(1);
  }
  CUDA_CHECK(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&state.source, config.payloadBytes));
  CUDA_CHECK(cudaMalloc(&state.local, sizeof(LocalControl)));
  CUDA_CHECK(cudaMalloc(&state.deviceRecords, state.gridCtas * sizeof(CtaRecord)));
  CUDA_CHECK(cudaMalloc(&state.mismatches, sizeof(unsigned long long)));
  state.pressure = nullptr;
  if (state.localCoRunnerCtas && config.pressureKind != 1) {
    CUDA_CHECK(cudaMalloc(&state.pressure, pressureBytes));
    CUDA_CHECK(cudaMemsetAsync(state.pressure, 1, pressureBytes, state.stream));
  }
  // More than half an SM's shared memory ensures no two blocks can share it.
  state.dynamicSharedBytes = ((state.properties.sharedMemPerMultiprocessor / 2 + 255) / 256) * 256;
  CUDA_CHECK(cudaFuncSetAttribute(fusedPush, cudaFuncAttributeMaxDynamicSharedMemorySize, state.dynamicSharedBytes));
  int residentCtas = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residentCtas, fusedPush, config.threadsPerCta, state.dynamicSharedBytes));
  if (residentCtas != 1) {
    fprintf(stderr, "Expected exactly one resident CTA per SM, got %d\n", residentCtas);
    std::exit(1);
  }
  // Preload the measured function before concurrent launches (avoids lazy-load waits).
  cudaFuncAttributes attributes;
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, fusedPush));
  fprintf(stderr, "rank=%d gpu=%d SMs=%d grid=%d shared=%zu registers=%d local_bytes=%zu\n", rank, device, state.properties.multiProcessorCount, state.gridCtas, state.dynamicSharedBytes, attributes.numRegs, attributes.localSizeBytes);
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
static void prepareIteration(RankState& state, Config config, int rank, int iteration) {
  CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(state.windowMemory) + config.payloadBytes, 0, sizeof(WindowControl), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.local, 0, sizeof(LocalControl), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.deviceRecords, 0, state.gridCtas * sizeof(CtaRecord), state.stream));
  CUDA_CHECK(cudaMemsetAsync(state.mismatches, 0, sizeof(unsigned long long), state.stream));
  if (rank == 0) {
    initializeSource<<<256, 256, 0, state.stream>>>(state.source, config.payloadBytes / sizeof(uint32_t), iteration);
  } else {
    CUDA_CHECK(cudaMemsetAsync(state.windowMemory, 0xff, config.payloadBytes, state.stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
static void collectIteration(RankState& state, Config config, int rank, int iteration) {
  std::vector<CtaRecord> records(state.gridCtas);
  CUDA_CHECK(cudaMemcpy(records.data(), state.deviceRecords, state.gridCtas * sizeof(CtaRecord), cudaMemcpyDeviceToHost));
  std::set<int> usedSms;
  for (const auto& record : records) {
    if (!usedSms.insert(record.smId).second || record.endNs <= record.startNs) {
      fprintf(stderr, "Invalid SM separation or CTA timing\n");
      std::exit(1);
    }
  }
  if (rank == 1 && config.communicationEnabled) {
    verifyDestination<<<256, 256, 0, state.stream>>>(static_cast<uint32_t*>(state.windowMemory), config.payloadBytes / sizeof(uint32_t), iteration, state.mismatches);
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
    unsigned long long mismatches = 0;
    CUDA_CHECK(cudaMemcpy(&mismatches, state.mismatches, sizeof(mismatches), cudaMemcpyDeviceToHost));
    if (mismatches) {
      fprintf(stderr, "CORRECTNESS FAILURE: %llu mismatches\n", mismatches);
      std::exit(4);
    }
  }
  if (iteration >= config.warmup) {
    state.records.push_back(std::move(records));
  }
}
static void cleanupRank(RankState& state) {
  NCCL_CHECK(ncclDevCommDestroy(state.communicator, &state.deviceCommunicator));
  NCCL_CHECK(ncclCommWindowDeregister(state.communicator, state.window));
  NCCL_CHECK(ncclMemFree(state.windowMemory));
  CUDA_CHECK(cudaFree(state.source));
  CUDA_CHECK(cudaFree(state.pressure));
  CUDA_CHECK(cudaFree(state.local));
  CUDA_CHECK(cudaFree(state.deviceRecords));
  CUDA_CHECK(cudaFree(state.mismatches));
  CUDA_CHECK(cudaStreamDestroy(state.stream));
  NCCL_CHECK(ncclCommDestroy(state.communicator));
}
static void runRank(RankState& state, Config config, int rank, ncclUniqueId identifier, pthread_barrier_t* hostBarrier) {
  setupRank(state, config, rank, identifier);
  pthread_barrier_wait(hostBarrier);
  for (int iteration = 0; iteration < config.warmup + config.iterations; ++iteration) {
    prepareIteration(state, config, rank, iteration);
    pthread_barrier_wait(hostBarrier);
    KernelArguments arguments{state.window, state.source, state.pressure, config.payloadBytes / sizeof(uint32_t), state.local, state.deviceRecords, config, rank, iteration, state.localCoRunnerCtas};
    void* kernelArguments[] = {&arguments};
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(fusedPush), state.gridCtas, config.threadsPerCta, kernelArguments, state.dynamicSharedBytes, state.stream));
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
    pthread_barrier_wait(hostBarrier);
    collectIteration(state, config, rank, iteration);
    pthread_barrier_wait(hostBarrier);
  }
  cleanupRank(state);
}
static void writeJson(const char* path, const char* configPath, Config config, const std::array<RankState, 2>& states) {
  FILE* output = fopen(path, "w");
  if (!output) {
    perror(path);
    std::exit(1);
  }
  // Output keys are stable and consumed by collect.py and make_figures.py.
  fprintf(output, "{\"schema_version\":1,\"config_path\":\"%s\",\"source_gpu\":%d,\"destination_gpu\":%d,\"ctas\":%d,\"threads\":%d,\"kind\":%d,\"co_ctas\":%d,\"pacing\":%d,\"detailed\":%d,\"global_stride\":%d,\"communication\":%d,\"vector_reads\":%d,\"bytes\":%zu,\"verified\":true,\"ranks\":[", configPath, config.sourceGpu, config.destinationGpu, config.communicationCtas, config.threadsPerCta, config.pressureKind, config.coRunnerCtas, config.pacingRounds, config.detailed, config.globalStride, config.communicationEnabled, config.vectorizedReads, config.payloadBytes);
  for (int rank = 0; rank < 2; ++rank) {
    if (rank) fputc(',', output);
    const auto& state = states[rank];
    fprintf(output, "{\"rank\":%d,\"sm_count\":%d,\"shared_bytes\":%zu,\"iterations\":[", rank, state.properties.multiProcessorCount, state.dynamicSharedBytes);
    for (size_t iteration = 0; iteration < state.records.size(); ++iteration) {
      if (iteration) fputc(',', output);
      fputc('[', output);
      for (size_t recordIndex = 0; recordIndex < state.records[iteration].size(); ++recordIndex) {
        if (recordIndex) fputc(',', output);
        const auto& record = state.records[iteration][recordIndex];
        fprintf(output, "{\"cta\":%zu,\"role\":%d,\"sm_id\":%d,\"start_ns\":%llu,\"issue_end_ns\":%llu,\"fence_end_ns\":%llu,\"publish_end_ns\":%llu,\"end_ns\":%llu,\"work_units\":%llu,\"checksum\":%.8g,\"checkpoints_ns\":[", recordIndex, record.role, record.smId, record.startNs, record.issueEndNs, record.fenceEndNs, record.publishEndNs, record.endNs, record.workUnits, record.checksum);
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
int main(int argumentCount, char** arguments) {
  if (argumentCount != 3) {
    fprintf(stderr, "Usage: push_harness CONFIG OUTPUT.json\n");
    return 1;
  }
  Config config = parseConfiguration(arguments[1]);
  ncclUniqueId identifier;
  NCCL_CHECK(ncclGetUniqueId(&identifier));
  pthread_barrier_t hostBarrier;
  pthread_barrier_init(&hostBarrier, nullptr, 2);
  std::array<RankState, 2> states;
  std::thread sourceThread(runRank, std::ref(states[0]), config, 0, identifier, &hostBarrier);
  std::thread destinationThread(runRank, std::ref(states[1]), config, 1, identifier, &hostBarrier);
  sourceThread.join();
  destinationThread.join();
  pthread_barrier_destroy(&hostBarrier);
  writeJson(arguments[2], arguments[1], config, states);
  return 0;
}
