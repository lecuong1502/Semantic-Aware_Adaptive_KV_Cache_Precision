#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

#include "microinfer/check.h"
#include "microinfer/kernels.h"

namespace microinfer
{

  size_t DeviceTensor::nbytes() const { return count_ * sizeof(__half); }

  DeviceTensor::DeviceTensor(const float *host, size_t count) : count_(count)
  {
    if (count == 0)
    {
      return;
    }

    // Converted on the host so the device never holds an fp32 copy. On a card
    // this small a transient fp32 staging buffer would double the peak for
    // nothing.
    std::vector<__half> staged(count);
    for (size_t i = 0; i < count; ++i)
    {
      staged[i] = __float2half(host[i]);
    }

    cuda_check(cudaMalloc(&ptr_, count * sizeof(__half)), "cudaMalloc");

    // A throw here would abandon the allocation: the destructor does not run
    // for an object whose constructor threw. That leak would be permanent and
    // invisible except as a smaller free-memory reading — which is the exact
    // signal RQ2 depends on, so it would corrupt the measurement rather than
    // merely waste memory.
    try
    {
      cuda_check(cudaMemcpy(ptr_, staged.data(), count * sizeof(__half),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy weights host-to-device");
    }
    catch (...)
    {
      cudaFree(ptr_);
      ptr_ = nullptr;
      count_ = 0;
      throw;
    }
  }

  DeviceTensor::DeviceTensor(size_t count) : count_(count)
  {
    if (count > 0)
    {
      cuda_check(cudaMalloc(&ptr_, count * sizeof(__half)), "cudaMalloc");
    }
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

  DeviceIndex::DeviceIndex(const int32_t *host, size_t count) : count_(count)
  {
    if (count == 0)
    {
      return;
    }
    cuda_check(cudaMalloc(&ptr_, count * sizeof(int32_t)), "cudaMalloc");
    // As in DeviceTensor: a throw after the allocation would otherwise leak it.
    try
    {
      cuda_check(cudaMemcpy(ptr_, host, count * sizeof(int32_t),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy indices host-to-device");
    }
    catch (...)
    {
      cudaFree(ptr_);
      ptr_ = nullptr;
      count_ = 0;
      throw;
    }
  }

  DeviceIndex::~DeviceIndex()
  {
    if (ptr_ != nullptr)
    {
      cudaFree(ptr_);
    }
  }

} // namespace microinfer
