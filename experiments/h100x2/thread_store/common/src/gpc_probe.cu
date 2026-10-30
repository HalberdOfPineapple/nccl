// SM-to-GPC probe (plan 07, E7.0, metric A12) for one GPU.
//
// CUDA runs all CTAs of one thread-block cluster in the same GPC. Each CTA records its
// cluster number and its SM ID; two SMs that ever ran CTAs of the same cluster are in the
// same GPC. The grouping itself is done by build_gpc_map.py; this program only records.
// JSON keys are read by build_gpc_map.py; renaming C++ fields must not rename keys.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) do { \
  cudaError_t status = (call); \
  if (status != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
    std::exit(2); \
  } \
} while (0)

struct ClusterRecord {
  int clusterIndex;
  int smId;
};

__device__ __forceinline__ unsigned long long readTimerNs() {
  unsigned long long timestampNs;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestampNs));
  return timestampNs;
}

// Each CTA stays resident for its own pseudo-random time and leaves without waiting for its
// cluster. Freed SMs then pair with SMs that are still idle, so later clusters span different
// SM sets; with a cluster-wide wait, the scheduler reused the same SM groups in every wave and
// four TPCs never shared a cluster (first trial, 2026-10-29).
__global__ void recordClusterPlacement(ClusterRecord* records, unsigned residentNs, unsigned seed) {
  extern __shared__ int reservation[];
  if (threadIdx.x == 0) {
    int smId;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smId));
    cg::cluster_group cluster = cg::this_cluster();
    records[blockIdx.x].clusterIndex = blockIdx.x / cluster.num_blocks();
    records[blockIdx.x].smId = smId;
    reservation[0] = smId;
    unsigned long long startNs = readTimerNs();
    unsigned hash = (blockIdx.x + 1) * 2654435761u ^ seed * 2246822519u;
    hash ^= hash >> 15;
    hash *= 2246822519u;
    hash ^= hash >> 13;
    unsigned stayNs = residentNs / 4 + hash % (2 * residentNs);
    while (readTimerNs() - startNs < stayNs) {
      __nanosleep(256);
    }
  }
}

static bool authorizedGpu(int gpu) {
  // User allocation for plan 07: physical GPUs 4 and 5 from 2026-10-29; GPUs 0 and 1 from 2026-10-30.
  return gpu == 0 || gpu == 1 || gpu == 4 || gpu == 5;
}

static int maximumActiveClusters(int clusterSize, size_t sharedBytes) {
  cudaLaunchConfig_t launch = {};
  cudaLaunchAttribute attribute;
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = clusterSize;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  launch.gridDim = dim3(clusterSize * 64);
  launch.blockDim = dim3(128);
  launch.dynamicSmemBytes = sharedBytes;
  launch.attrs = &attribute;
  launch.numAttrs = 1;
  int clusters = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters, recordClusterPlacement, &launch));
  return clusters;
}

// One launch of clusterCount clusters; returns per-CTA records.
static std::vector<ClusterRecord> launchClusters(int clusterSize, int clusterCount, size_t sharedBytes, unsigned residentNs, unsigned seed) {
  int ctas = clusterSize * clusterCount;
  ClusterRecord* deviceRecords = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceRecords, ctas * sizeof(ClusterRecord)));
  cudaLaunchConfig_t launch = {};
  cudaLaunchAttribute attribute;
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = clusterSize;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  launch.gridDim = dim3(ctas);
  launch.blockDim = dim3(128);
  launch.dynamicSmemBytes = sharedBytes;
  launch.attrs = &attribute;
  launch.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&launch, recordClusterPlacement, deviceRecords, residentNs, seed));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<ClusterRecord> records(ctas);
  CUDA_CHECK(cudaMemcpy(records.data(), deviceRecords, ctas * sizeof(ClusterRecord), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(deviceRecords));
  return records;
}

static std::string uuidText(const cudaUUID_t& uuid) {
  char text[64];
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(uuid.bytes);
  snprintf(text, sizeof(text), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
           bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
  return text;
}

int main(int argumentCount, char** arguments) {
  if (argumentCount != 4) {
    fprintf(stderr, "Usage: gpc_probe GPU REPEATS OUTPUT.json\n");
    return 1;
  }
  int gpu = std::atoi(arguments[1]);
  int repeats = std::atoi(arguments[2]);
  if (!authorizedGpu(gpu) || repeats < 1) {
    fprintf(stderr, "GPU outside the authorized allocation or invalid repeat count\n");
    return 1;
  }
  CUDA_CHECK(cudaSetDevice(gpu));
  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, gpu));
  // More than half an SM's shared memory: at most one CTA per SM, so cluster CTAs use distinct SMs.
  size_t sharedBytes = ((properties.sharedMemPerMultiprocessor / 2 + 255) / 256) * 256;
  CUDA_CHECK(cudaFuncSetAttribute(recordClusterPlacement, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedBytes));
  // Cluster size 16 is above the portable limit of 8.
  CUDA_CHECK(cudaFuncSetAttribute(recordClusterPlacement, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  FILE* output = fopen(arguments[3], "w");
  if (!output) {
    perror(arguments[3]);
    return 1;
  }
  fprintf(output, "{\"schema_version\":1,\"gpu\":%d,\"uuid\":\"%s\",\"pci_bus\":%d,\"sm_count\":%d,\"shared_bytes\":%zu,\"repeats\":%d,\"cluster_sizes\":[",
          gpu, uuidText(properties.uuid).c_str(), properties.pciBusID, properties.multiProcessorCount, sharedBytes, repeats);
  const int clusterSizes[] = {2, 4, 8, 16};
  bool firstSize = true;
  for (int clusterSize : clusterSizes) {
    int activeClusters = maximumActiveClusters(clusterSize, sharedBytes);
    fprintf(stderr, "gpu=%d cluster_size=%d max_active_clusters=%d\n", gpu, clusterSize, activeClusters);
    fprintf(output, "%s{\"cluster_size\":%d,\"max_active_clusters\":%d,\"launches\":[", firstSize ? "" : ",", clusterSize, activeClusters);
    firstSize = false;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      // Eight waves per launch; the seed changes every CTA's stay time between launches.
      unsigned residentNs = 40000;
      std::vector<ClusterRecord> records = launchClusters(clusterSize, std::max(activeClusters, 1) * 8, sharedBytes, residentNs, repeat + 1);
      fprintf(output, "%s[", repeat ? "," : "");
      for (size_t index = 0; index < records.size(); ++index) {
        fprintf(output, "%s[%d,%d]", index ? "," : "", records[index].clusterIndex, records[index].smId);
      }
      fprintf(output, "]");
    }
    fprintf(output, "]}");
  }
  fprintf(output, "]}\n");
  fclose(output);
  return 0;
}
