#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

#include "microinfer/check.h"
#include "microinfer/kernels.h"

namespace microinfer
{

  MemoryInfo device_memory_info()
  {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    return MemoryInfo{free_bytes, total_bytes};
  }

  DeviceTensor::DeviceTensor(const float *host, size_t count) : count_(count)
  {
    if (count == 0)
    {
      return;
    }

    // Converted on the host so the device never holds an fp32 copy. On a 6 GiB
    // card a transient fp32 staging buffer would double the peak for no reason.
    std::vector<__half> staged(count);
    for (size_t i = 0; i < count; ++i)
    {
      staged[i] = __float2half(host[i]);
    }

    cuda_check(cudaMalloc(&ptr_, count * sizeof(__half)), "cudaMalloc");
    cuda_check(cudaMemcpy(ptr_, staged.data(), count * sizeof(__half),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy weights host-to-device");
  }

  DeviceTensor::~DeviceTensor()
  {
    if (ptr_ != nullptr)
    {
      cudaFree(ptr_);
    }
  }

  void DeviceTensor::download(float *out) const
  {
    if (count_ == 0)
    {
      return;
    }
    std::vector<__half> staged(count_);
    cuda_check(cudaMemcpy(staged.data(), ptr_, count_ * sizeof(__half),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy weights device-to-host");
    for (size_t i = 0; i < count_; ++i)
    {
      out[i] = __half2float(staged[i]);
    }
  }

} // namespace microinfer
