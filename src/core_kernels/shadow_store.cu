#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "microinfer/check.h"
#include "microinfer/shadow_store.h"

namespace microinfer
{

  ShadowStore::ShadowStore(std::size_t page_bytes)
      : page_bytes_(page_bytes),
        // A KVPages makes its store before it checks its own dimensions,
        // so a page of no bytes must not divide by zero here.
        pages_per_allocation_(page_bytes == 0 || page_bytes >= kAllocationBytes
                                  ? 1
                                  : kAllocationBytes / page_bytes)
  {
  }

  ShadowStore::~ShadowStore()
  {
    for (std::uint8_t *allocation : allocations_)
    {
      cudaFreeHost(allocation);
    }
  }

  void ShadowStore::take(PageKey key, const void *device_page)
  {
    if (contains(key))
    {
      throw std::logic_error("page " + std::to_string(key.page_index) +
                             " of layer " + std::to_string(key.layer) +
                             " has a shadow already; a page has one");
    }
    const std::size_t slot = shadows_.size();
    if (slot == allocations_.size() * pages_per_allocation_)
    {
      void *allocation = nullptr;
      cuda_check(cudaHostAlloc(&allocation, pages_per_allocation_ * page_bytes_,
                               cudaHostAllocDefault),
                 "cudaHostAlloc pinned memory for FP16 shadows");
      allocations_.push_back(static_cast<std::uint8_t *>(allocation));
    }
    std::uint8_t *shadow = allocations_[slot / pages_per_allocation_] +
                           (slot % pages_per_allocation_) * page_bytes_;
    // Synchronous: it returns once the bytes are in host memory, after
    // every launch already enqueued on the legacy default stream.
    cuda_check(
        cudaMemcpy(shadow, device_page, page_bytes_, cudaMemcpyDeviceToHost),
        "cudaMemcpy a page to its FP16 shadow");
    shadows_[key] = shadow;
  }

  bool ShadowStore::contains(PageKey key) const
  {
    return shadows_.count(key) != 0;
  }

  const std::uint8_t *ShadowStore::shadow(PageKey key) const
  {
    const auto it = shadows_.find(key);
    if (it == shadows_.end())
    {
      throw PageNotFound(key);
    }
    return it->second;
  }

} // namespace microinfer
