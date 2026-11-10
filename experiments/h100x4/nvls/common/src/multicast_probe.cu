// Reads the attribute NCCL uses to decide NVLS multicast support, per visible device.
#include <cuda.h>
#include <cstdio>
int main() {
  CUresult initResult = cuInit(0);
  printf("cuInit=%d\n", (int)initResult);
  int deviceCount = 0;
  cuDeviceGetCount(&deviceCount);
  int driverVersion = 0;
  cuDriverGetVersion(&driverVersion);
  printf("deviceCount=%d driverCudaVersion=%d\n", deviceCount, driverVersion);
  for (int ordinal = 0; ordinal < deviceCount; ordinal++) {
    CUdevice device;
    cuDeviceGet(&device, ordinal);
    char name[128];
    cuDeviceGetName(name, sizeof(name), device);
    int multicastSupported = -1;
    CUresult attributeResult = cuDeviceGetAttribute(&multicastSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
    printf("ordinal=%d name=%s attributeResult=%d multicastSupported=%d\n", ordinal, name, (int)attributeResult, multicastSupported);
  }
  return 0;
}
