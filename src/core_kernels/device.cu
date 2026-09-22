#include <cuda.h>
#include <cuda_runtime.h>

#include "microinfer/check.h"
#include "microinfer/kernels.h"

namespace microinfer
{

  int cuda_driver_version()
  {
    // cuInit is required before any other driver API call. It is idempotent.
    driver_check(cuInit(0), "cuInit");
    int version = 0;
    driver_check(cuDriverGetVersion(&version), "cuDriverGetVersion");
    return version;
  }

  MemoryInfo device_memory_info()
  {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    return MemoryInfo{free_bytes, total_bytes};
  }

  std::string device_name()
  {
    cudaDeviceProp props{};
    cuda_check(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
    return props.name;
  }

} // namespace microinfer
