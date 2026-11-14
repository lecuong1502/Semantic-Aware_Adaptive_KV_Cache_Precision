#include <cuda_runtime.h>

#include <algorithm>
#include <iterator>
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
    for (const Allocation &a : allocations_)
    {
      cudaFreeHost(a.host);
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
    reserve(1);
    const Allocation &a = allocations_.back();
    std::uint8_t *shadow = a.host + next_slot() * page_bytes_;
    // Synchronous: it returns once the bytes are in host memory, after
    // every launch already enqueued on the legacy default stream.
    cuda_check(
        cudaMemcpy(shadow, device_page, page_bytes_, cudaMemcpyDeviceToHost),
        "cudaMemcpy a page to its FP16 shadow");
    shadows_[key] = shadow;
  }

  void ShadowStore::reserve(std::size_t count)
  {
    const std::size_t room =
        allocations_.empty()
            ? 0
            : allocations_.back().pages - allocations_.back().used;
    if (count <= room)
    {
      return;
    }
    // What the last allocation has left is given up: shadows fill one
    // allocation at a time.
    const std::size_t pages = std::max(count, pages_per_allocation_);
    // Mapped, so that a batch's kernels read and write shadows in place
    // (#205); under unified addressing the device's address is the host's,
    // but it is asked for, not assumed.
    void *allocation = nullptr;
    cuda_check(
        cudaHostAlloc(&allocation, pages * page_bytes_, cudaHostAllocMapped),
        "cudaHostAlloc pinned memory for FP16 shadows");
    void *on_device = nullptr;
    const cudaError_t mapped =
        cudaHostGetDevicePointer(&on_device, allocation, 0);
    if (mapped != cudaSuccess)
    {
      cudaFreeHost(allocation);
      cuda_check(mapped, "cudaHostGetDevicePointer for FP16 shadows");
    }
    by_address_[static_cast<std::uint8_t *>(allocation)] = allocations_.size();
    allocations_.push_back({static_cast<std::uint8_t *>(allocation),
                            static_cast<std::uint8_t *>(on_device), pages, 0});
    capacity_ += pages;
  }

  std::size_t ShadowStore::next_slot() { return allocations_.back().used++; }

  std::uint8_t *ShadowStore::add(PageKey key)
  {
    if (contains(key))
    {
      throw std::logic_error("page " + std::to_string(key.page_index) +
                             " of layer " + std::to_string(key.layer) +
                             " has a shadow already; a page has one");
    }
    reserve(1);
    const Allocation &a = allocations_.back();
    const std::size_t slot = next_slot();
    shadows_[key] = a.host + slot * page_bytes_;
    return a.device + slot * page_bytes_;
  }

  const std::uint8_t *ShadowStore::device_shadow(PageKey key) const
  {
    const std::uint8_t *host = shadow(key);
    // The allocation at or below the shadow's address holds it.
    const Allocation &a =
        allocations_[std::prev(by_address_.upper_bound(host))->second];
    return a.device + (host - a.host);
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
