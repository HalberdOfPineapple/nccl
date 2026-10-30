// Standalone calibration of the HBM read-pressure role on one GPU.
// The launch has the same shape as the fused communication kernel: one CTA per SM, a control
// CTA, and pressure CTAs that call hbm_read_pressure::readUntilStopped from one thread.
// JSON keys are the data interface read by the analysis scripts; renaming a C++ field must not rename a key.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "../include/hbm_read_pressure.cuh"

#define CUDA_CHECK(call) do { \
  cudaError_t status = (call); \
  if (status != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
    std::exit(2); \
  } \
} while (0)

constexpr size_t workingSetBytes = size_t(1) << 30;
constexpr uint32_t workingSetWord = 0x01010101u;  // every byte of the buffer is 0x01
constexpr int timerSamples = 256;

struct Config {
  int gpu = 6;
  int pressureCtas = 16;
  int blockThreads = 1024;
  int chunkKib = 16;
  int chunksInFlight = 6;
  int targetGbps = 0;                         // total over all pressure CTAs; 0 = no limit
  unsigned long long windowNs = 20000000;
  int warmup = 2;
  int windows = 5;
};
struct LocalControl {
  int readyCount;
  int stop;
};
struct CtaRecord {
  unsigned long long startNs;
  unsigned long long endNs;
  unsigned long long chunksRead;
  unsigned long long checksum;
  unsigned long long timerStepMinNs;          // control CTA only: smallest timer increment seen
  unsigned long long timerStepMaxNs;
  int smId;
  int role;                                   // 0 control; 1 pressure
};
struct KernelArguments {
  HbmReadSettings settings;
  LocalControl* local;
  CtaRecord* records;
  unsigned long long windowNs;
};

__device__ __forceinline__ int readSmId() {
  int smId;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smId));
  return smId;
}

// Measure how finely %globaltimer advances; it limits how evenly the rate knob can space chunks.
__device__ void measureTimerStep(CtaRecord& record) {
  unsigned long long smallestNs = ~0ull;
  unsigned long long largestNs = 0;
  unsigned long long previousNs = hbm_read_pressure::timerNs();
  for (int sample = 0; sample < timerSamples; ++sample) {
    unsigned long long currentNs = hbm_read_pressure::timerNs();
    while (currentNs == previousNs) {
      currentNs = hbm_read_pressure::timerNs();
    }
    smallestNs = min(smallestNs, currentNs - previousNs);
    largestNs = max(largestNs, currentNs - previousNs);
    previousNs = currentNs;
  }
  record.timerStepMinNs = smallestNs;
  record.timerStepMaxNs = largestNs;
}

// Control CTA: wait until every pressure CTA has started, keep them running for one window, stop them.
__device__ void runControl(KernelArguments arguments) {
  CtaRecord& record = arguments.records[0];
  measureTimerStep(record);
  int pressureCtas = arguments.settings.pressureCtas;
  while (cuda::atomic_ref<int, cuda::thread_scope_device>(arguments.local->readyCount).load(cuda::memory_order_acquire) != pressureCtas) {
    __nanosleep(64);
  }
  record.startNs = hbm_read_pressure::timerNs();
  while (hbm_read_pressure::timerNs() - record.startNs < arguments.windowNs) {
    __nanosleep(256);
  }
  cuda::atomic_ref<int, cuda::thread_scope_system>(arguments.local->stop).store(1, cuda::memory_order_release);
  record.endNs = hbm_read_pressure::timerNs();
}

// Pressure CTA: thread 0 runs the copies and thread 32 watches the stop flag; the other
// threads wait at the final barrier, as idle lanes of a fused kernel's co-runner role would.
__device__ void runPressure(KernelArguments arguments, uint8_t* sharedReservation, volatile int* sharedStopFlag) {
  if (threadIdx.x == 32) {
    hbm_read_pressure::forwardStopFlag(&arguments.local->stop, sharedStopFlag);
    return;
  }
  const HbmReadSettings& settings = arguments.settings;
  uint8_t* slots = sharedReservation;
  uint64_t* barriers = reinterpret_cast<uint64_t*>(sharedReservation + size_t(settings.chunksInFlight) * settings.chunkBytes);
  CtaRecord& record = arguments.records[blockIdx.x];
  cuda::atomic_ref<int, cuda::thread_scope_device>(arguments.local->readyCount).fetch_add(1, cuda::memory_order_release);
  HbmReadResult result;
  hbm_read_pressure::readUntilStopped(settings, blockIdx.x - 1, slots, barriers, sharedStopFlag, result);
  record.startNs = result.startNs;
  record.endNs = result.endNs;
  record.chunksRead = result.chunksRead;
  record.checksum = result.checksum;
}

__global__ void calibrationKernel(KernelArguments arguments) {
  extern __shared__ __align__(128) uint8_t sharedReservation[];
  __shared__ int sharedStopFlag;
  if (threadIdx.x == 0) {
    sharedStopFlag = 0;
    arguments.records[blockIdx.x].smId = readSmId();
    arguments.records[blockIdx.x].role = blockIdx.x > 0;
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    runControl(arguments);
  } else if (blockIdx.x > 0 && (threadIdx.x == 0 || threadIdx.x == 32)) {
    runPressure(arguments, sharedReservation, &sharedStopFlag);
  }
  __syncthreads();
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
    if (key == "gpu") {
      config.gpu = value;
    }
    else if (key == "pressure_ctas") {
      config.pressureCtas = value;
    }
    else if (key == "block_threads") {
      config.blockThreads = value;
    }
    else if (key == "chunk_kib") {
      config.chunkKib = value;
    }
    else if (key == "chunks_in_flight") {
      config.chunksInFlight = value;
    }
    else if (key == "target_gbps") {
      config.targetGbps = value;
    }
    else if (key == "window_ns") {
      config.windowNs = value;
    }
    else if (key == "warmup") {
      config.warmup = value;
    }
    else if (key == "windows") {
      config.windows = value;
    }
    else {
      fprintf(stderr, "Unknown config key %s\n", key.c_str());
      std::exit(1);
    }
  }
  if (config.gpu < 4 || config.gpu > 7 || config.pressureCtas < 1 || config.blockThreads < 64 || config.blockThreads > 1024 || config.blockThreads % 32 || config.chunkKib < 1 || config.chunkKib > 512 || config.chunksInFlight < 1 || config.chunksInFlight > 64 || config.targetGbps < 0 || config.windows < 1 || config.warmup < 0 || config.windowNs == 0) {
    fprintf(stderr, "Invalid configuration or GPU outside authorized allocation 4-7\n");
    std::exit(1);
  }
  return config;
}

// Shared memory per CTA: at least the fused harness's reservation (more than half an SM, so
// only one CTA fits per SM), or more if the requested chunks need it.
static size_t sharedBytesFor(const Config& config, const cudaDeviceProp& properties) {
  size_t fusedReservation = ((properties.sharedMemPerMultiprocessor / 2 + 255) / 256) * 256;
  size_t chunkSpace = size_t(config.chunksInFlight) * config.chunkKib * 1024 + size_t(config.chunksInFlight) * sizeof(uint64_t);
  return std::max(fusedReservation, (chunkSpace + 255) / 256 * 256);
}

static HbmReadSettings buildSettings(const Config& config, const uint8_t* workingSet) {
  HbmReadSettings settings;
  settings.workingSet = workingSet;
  settings.workingSetBytes = workingSetBytes;
  settings.pressureCtas = config.pressureCtas;
  settings.chunkBytes = config.chunkKib * 1024;
  settings.chunksInFlight = config.chunksInFlight;
  // Per-CTA interval between chunk starts, in picoseconds, for the requested total rate.
  settings.chunkIntervalPs = config.targetGbps ? (unsigned long long)(double(settings.chunkBytes) * config.pressureCtas * 1000.0 / config.targetGbps) : 0;
  return settings;
}

static void checkWindow(const Config& config, const std::vector<CtaRecord>& records) {
  std::set<int> usedSms;
  for (const auto& record : records) {
    if (!usedSms.insert(record.smId).second || record.endNs <= record.startNs) {
      fprintf(stderr, "Invalid SM separation or CTA timing\n");
      std::exit(1);
    }
    if (record.role == 1 && (record.chunksRead == 0 || record.checksum != record.chunksRead * workingSetWord)) {
      fprintf(stderr, "CHECKSUM FAILURE: chunks=%llu checksum=%llu\n", record.chunksRead, record.checksum);
      std::exit(4);
    }
  }
}

static void writeJson(const char* path, const char* configPath, const Config& config, const HbmReadSettings& settings, const cudaDeviceProp& properties, int memoryClockKhz, int busWidthBits, size_t sharedBytes, int registers, const std::vector<std::vector<CtaRecord>>& windows) {
  FILE* output = fopen(path, "w");
  if (!output) {
    perror(path);
    std::exit(1);
  }
  // Output keys are stable and consumed by analyze_hbm_read_rate.py.
  double peakGbps = 2.0 * memoryClockKhz * 1e3 * busWidthBits / 8 / 1e9;
  fprintf(output, "{\"schema_version\":1,\"config_path\":\"%s\",\"gpu\":%d,\"pressure_ctas\":%d,\"block_threads\":%d,\"chunk_kib\":%d,\"chunks_in_flight\":%d,\"target_gbps\":%d,\"chunk_interval_ps\":%llu,\"window_ns\":%llu,\"working_set_bytes\":%zu,\"sm_count\":%d,\"shared_bytes\":%zu,\"registers\":%d,\"memory_clock_khz\":%d,\"bus_width_bits\":%d,\"peak_gbps\":%.3f,\"l2_bytes\":%d,\"verified\":true,\"windows\":[",
          configPath, config.gpu, config.pressureCtas, config.blockThreads, config.chunkKib, config.chunksInFlight, config.targetGbps, settings.chunkIntervalPs, config.windowNs, workingSetBytes, properties.multiProcessorCount, sharedBytes, registers, memoryClockKhz, busWidthBits, peakGbps, properties.l2CacheSize);
  for (size_t window = 0; window < windows.size(); ++window) {
    fprintf(output, "%s[", window ? "," : "");
    for (size_t index = 0; index < windows[window].size(); ++index) {
      const CtaRecord& record = windows[window][index];
      fprintf(output, "%s{\"cta\":%zu,\"role\":%d,\"sm_id\":%d,\"start_ns\":%llu,\"end_ns\":%llu,\"chunks_read\":%llu,\"timer_step_min_ns\":%llu,\"timer_step_max_ns\":%llu}",
              index ? "," : "", index, record.role, record.smId, record.startNs, record.endNs, record.chunksRead, record.timerStepMinNs, record.timerStepMaxNs);
    }
    fprintf(output, "]");
  }
  fprintf(output, "]}\n");
  fclose(output);
}

int main(int argumentCount, char** arguments) {
  if (argumentCount != 3) {
    fprintf(stderr, "Usage: hbm_read_calibration CONFIG OUTPUT.json\n");
    return 1;
  }
  Config config = parseConfiguration(arguments[1]);
  CUDA_CHECK(cudaSetDevice(config.gpu));
  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, config.gpu));
  int memoryClockKhz = 0;
  int busWidthBits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&memoryClockKhz, cudaDevAttrMemoryClockRate, config.gpu));
  CUDA_CHECK(cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, config.gpu));
  int gridCtas = config.pressureCtas + 1;
  if (gridCtas > properties.multiProcessorCount) {
    fprintf(stderr, "Grid exceeds one-CTA-per-SM residency\n");
    return 1;
  }
  size_t sharedBytes = sharedBytesFor(config, properties);
  if (sharedBytes > properties.sharedMemPerBlockOptin) {
    fprintf(stderr, "Chunks in flight need more shared memory than one CTA may use\n");
    return 1;
  }
  CUDA_CHECK(cudaFuncSetAttribute(calibrationKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedBytes));
  int residentCtas = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residentCtas, calibrationKernel, config.blockThreads, sharedBytes));
  if (residentCtas != 1) {
    fprintf(stderr, "Expected exactly one resident CTA per SM, got %d\n", residentCtas);
    return 1;
  }
  cudaFuncAttributes attributes;
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, calibrationKernel));
  fprintf(stderr, "gpu=%d grid=%d block=%d shared=%zu registers=%d local_bytes=%zu\n", config.gpu, gridCtas, config.blockThreads, sharedBytes, attributes.numRegs, attributes.localSizeBytes);
  uint8_t* workingSet = nullptr;
  LocalControl* local = nullptr;
  CtaRecord* deviceRecords = nullptr;
  CUDA_CHECK(cudaMalloc(&workingSet, workingSetBytes));
  CUDA_CHECK(cudaMemset(workingSet, 1, workingSetBytes));
  CUDA_CHECK(cudaMalloc(&local, sizeof(LocalControl)));
  CUDA_CHECK(cudaMalloc(&deviceRecords, gridCtas * sizeof(CtaRecord)));
  HbmReadSettings settings = buildSettings(config, workingSet);
  std::vector<std::vector<CtaRecord>> windows;
  for (int window = 0; window < config.warmup + config.windows; ++window) {
    CUDA_CHECK(cudaMemset(local, 0, sizeof(LocalControl)));
    CUDA_CHECK(cudaMemset(deviceRecords, 0, gridCtas * sizeof(CtaRecord)));
    KernelArguments kernelArguments{settings, local, deviceRecords, config.windowNs};
    void* launchArguments[] = {&kernelArguments};
    // Cooperative launch guarantees all CTAs are resident, so the control CTA's wait cannot deadlock.
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(calibrationKernel), gridCtas, config.blockThreads, launchArguments, sharedBytes, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<CtaRecord> records(gridCtas);
    CUDA_CHECK(cudaMemcpy(records.data(), deviceRecords, gridCtas * sizeof(CtaRecord), cudaMemcpyDeviceToHost));
    checkWindow(config, records);
    if (window >= config.warmup) {
      windows.push_back(records);
    }
  }
  writeJson(arguments[2], arguments[1], config, settings, properties, memoryClockKhz, busWidthBits, sharedBytes, attributes.numRegs, windows);
  CUDA_CHECK(cudaFree(workingSet));
  CUDA_CHECK(cudaFree(local));
  CUDA_CHECK(cudaFree(deviceRecords));
  return 0;
}
