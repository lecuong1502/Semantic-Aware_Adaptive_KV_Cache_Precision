#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "microinfer/check.h"
#include "microinfer/kernels.h"

namespace microinfer
{

  size_t DeviceTensor::nbytes() const { return count_ * sizeof(__half); }

  void DeviceTensor::fill(const float *host)
  {
    if (count_ == 0)
    {
      return;
    }
    // Converted on the host so the device never holds an fp32 copy. On a card
    // this small a transient fp32 staging buffer would double the peak for
    // nothing.
    std::vector<__half> staged(count_);
    for (size_t i = 0; i < count_; ++i)
    {
      staged[i] = __float2half(host[i]);
    }
    cuda_check(cudaMemcpy(ptr_, staged.data(), count_ * sizeof(__half),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy weights host-to-device");
  }

  // The arena is held by storage_ from the moment it exists, so a throw from
  // fill() below, after the allocation, still gives it back: the destructor
  // of a member that was constructed runs even when the constructor throws.
  // A leak here would be permanent and invisible except as a smaller
  // free-memory reading, which is the exact signal RQ2 depends on.
  DeviceTensor::DeviceTensor(const float *host, size_t count)
      : storage_(std::make_shared<DeviceArena>(count * sizeof(__half))),
        ptr_(storage_->base()), count_(count)
  {
    fill(host);
  }

  DeviceTensor::DeviceTensor(size_t count)
      : storage_(std::make_shared<DeviceArena>(count * sizeof(__half))),
        ptr_(storage_->base()), count_(count)
  {
  }

  DeviceTensor::DeviceTensor(std::shared_ptr<DeviceArena> arena,
                             std::size_t offset, const float *host,
                             std::size_t count)
      : storage_(std::move(arena)), count_(count)
  {
    if (storage_ == nullptr)
    {
      throw std::invalid_argument("a tensor in an arena needs the arena");
    }
    if (offset % kWeightAlignment != 0)
    {
      throw std::invalid_argument(
          "offset " + std::to_string(offset) + " is not a multiple of " +
          std::to_string(kWeightAlignment) +
          " bytes, the alignment every weight is given (kernels.h)");
    }
    // Divided rather than multiplied, so that no count overflows the test.
    if (offset > storage_->nbytes() ||
        count > (storage_->nbytes() - offset) / sizeof(__half))
    {
      throw std::invalid_argument(
          std::to_string(count) + " elements from byte " +
          std::to_string(offset) + " run past the arena's " +
          std::to_string(storage_->nbytes()) + " bytes");
    }
    ptr_ = static_cast<char *>(storage_->base()) + offset;
    fill(host);
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

  DeviceFloats::DeviceFloats(size_t count) : count_(count)
  {
    if (count > 0)
    {
      cuda_check(cudaMalloc(&ptr_, count * sizeof(float)), "cudaMalloc");
    }
  }

  DeviceFloats::~DeviceFloats()
  {
    if (ptr_ != nullptr)
    {
      cudaFree(ptr_);
    }
  }

  void DeviceFloats::download(float *out) const
  {
    if (count_ > 0)
    {
      cuda_check(cudaMemcpy(out, ptr_, nbytes(), cudaMemcpyDeviceToHost),
                 "cudaMemcpy fp32 device-to-host");
    }
  }

} // namespace microinfer
