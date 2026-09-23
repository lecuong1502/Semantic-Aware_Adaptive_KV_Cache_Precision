#include <cuda.h>

#include <stdexcept>
#include <string>

#include "microinfer/check.h"
#include "microinfer/paged_kv_cache.h"

namespace microinfer
{

  namespace
  {

    std::size_t round_up(std::size_t n, std::size_t multiple)
    {
      return (n + multiple - 1) / multiple * multiple;
    }

    std::string describe(PageKey key)
    {
      return "page (layer " + std::to_string(key.layer) + ", page_index " +
             std::to_string(key.page_index) + ")";
    }

    // The driver API acts on the context current on the calling thread, and a
    // Python caller may arrive on any thread. Every entry point makes the
    // device's primary context current first; it is the same context the
    // runtime API uses, so kernels see the cache's memory.
    class ContextScope
    {
    public:
      explicit ContextScope(CUcontext ctx)
      {
        driver_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
      }
    };

  } // namespace

  PageNotFound::PageNotFound(PageKey key)
      : std::out_of_range(describe(key) + " is not in the page table")
  {
  }

  PagedKVCache::PagedKVCache(
      const std::array<std::size_t, kTierCount> &page_bytes,
      const std::array<std::size_t, kTierCount> &capacity_pages)
  {
    for (int t = 0; t < kTierCount; ++t)
    {
      if (page_bytes[t] == 0)
      {
        throw std::invalid_argument("page_bytes for tier " + std::to_string(t) +
                                    " must be positive");
      }
    }

    driver_check(cuInit(0), "cuInit");
    driver_check(cuDeviceGet(&device_, 0), "cuDeviceGet");
    driver_check(cuDevicePrimaryCtxRetain(&ctx_, device_),
                 "cuDevicePrimaryCtxRetain");
    ContextScope scope(ctx_);

    prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop_.location.id = device_;
    driver_check(cuMemGetAllocationGranularity(
                     &granule_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                 "cuMemGetAllocationGranularity");

    try
    {
      for (int t = 0; t < kTierCount; ++t)
      {
        Range &r = ranges_[t];
        r.page_bytes = page_bytes[t];
        r.capacity_pages = capacity_pages[t];
        r.reserved = round_up(page_bytes[t] * capacity_pages[t], granule_);
        if (r.reserved > 0)
        {
          driver_check(cuMemAddressReserve(&r.base, r.reserved, 0, 0, 0),
                       "cuMemAddressReserve");
        }
      }
    }
    catch (...)
    {
      release_everything();
      throw;
    }
  }

  PagedKVCache::~PagedKVCache() { release_everything(); }

  void PagedKVCache::release_everything() noexcept
  {
    // Errors are ignored here: this runs from a destructor, and the only
    // alternative to carrying on is leaking every other range as well.
    cuCtxSetCurrent(ctx_);
    cuCtxSynchronize();
    for (Range &r : ranges_)
    {
      while (!r.granules.empty())
      {
        const std::size_t last = r.granules.size() - 1;
        cuMemUnmap(r.base + last * granule_, granule_);
        cuMemRelease(r.granules.back());
        r.granules.pop_back();
      }
      if (r.base != 0)
      {
        cuMemAddressFree(r.base, r.reserved);
        r.base = 0;
      }
    }
    if (ctx_ != nullptr)
    {
      cuDevicePrimaryCtxRelease(device_);
      ctx_ = nullptr;
    }
  }

  void PagedKVCache::allocate(PageKey key, Tier tier)
  {
    if (contains(key))
    {
      throw std::invalid_argument(describe(key) + " is already allocated");
    }
    Range &r = range(tier);
    if (r.slots.size() == r.capacity_pages)
    {
      throw std::length_error("tier " + std::to_string(static_cast<int>(tier)) +
                              " is full at " +
                              std::to_string(r.capacity_pages) + " pages");
    }
    ContextScope scope(ctx_);

    r.slots.push_back(key);
    try
    {
      fit_granules(r);
    }
    catch (...)
    {
      r.slots.pop_back();
      fit_granules(r); // Give back any granule mapped before the failure.
      throw;
    }
    table_[key] = PageLocation{tier, r.slots.size() - 1};
    ++generation_;
  }

  void PagedKVCache::free(PageKey key)
  {
    const PageLocation loc = locate(key);
    Range &r = range(loc.tier);
    ContextScope scope(ctx_);

    const std::size_t tail = r.slots.size() - 1;
    if (loc.slot != tail)
    {
      const PageKey moved = r.slots[tail];
      driver_check(cuMemcpyDtoD(r.base + loc.slot * r.page_bytes,
                                r.base + tail * r.page_bytes, r.page_bytes),
                   "cuMemcpyDtoD (tail swap)");
      r.slots[loc.slot] = moved;
      table_[moved].slot = loc.slot;
    }
    r.slots.pop_back();
    table_.erase(key);
    ++generation_;

    // The copy must land before the granule it read from can be unmapped.
    driver_check(cuCtxSynchronize(), "cuCtxSynchronize");
    fit_granules(r);
  }

  void PagedKVCache::fit_granules(Range &r)
  {
    const std::size_t needed =
        round_up(r.slots.size() * r.page_bytes, granule_) / granule_;

    while (r.granules.size() < needed)
    {
      const CUdeviceptr at = r.base + r.granules.size() * granule_;
      CUmemGenericAllocationHandle handle;
      driver_check(cuMemCreate(&handle, granule_, &prop_, 0), "cuMemCreate");
      CUresult mapped = cuMemMap(at, granule_, 0, handle, 0);
      if (mapped != CUDA_SUCCESS)
      {
        cuMemRelease(handle);
        driver_check(mapped, "cuMemMap");
      }
      r.granules.push_back(handle);

      CUmemAccessDesc access{};
      access.location = prop_.location;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      driver_check(cuMemSetAccess(at, granule_, &access, 1), "cuMemSetAccess");
    }

    while (r.granules.size() > needed)
    {
      const std::size_t last = r.granules.size() - 1;
      driver_check(cuMemUnmap(r.base + last * granule_, granule_),
                   "cuMemUnmap");
      // Only now does the memory go back to the driver: releasing the handle
      // after unmapping is what frees the physical allocation.
      driver_check(cuMemRelease(r.granules.back()), "cuMemRelease");
      r.granules.pop_back();
    }
  }

  void PagedKVCache::write(PageKey key, const void *host, std::size_t bytes)
  {
    const CUdeviceptr at = address(key, bytes);
    ContextScope scope(ctx_);
    driver_check(cuMemcpyHtoD(at, host, bytes), "cuMemcpyHtoD");
  }

  void PagedKVCache::read(PageKey key, void *host, std::size_t bytes) const
  {
    const CUdeviceptr at = address(key, bytes);
    ContextScope scope(ctx_);
    driver_check(cuMemcpyDtoH(host, at, bytes), "cuMemcpyDtoH");
  }

  CUdeviceptr PagedKVCache::address(PageKey key, std::size_t bytes) const
  {
    const PageLocation loc = locate(key);
    const Range &r = range(loc.tier);
    if (bytes != r.page_bytes)
    {
      throw std::invalid_argument(describe(key) + " holds " +
                                  std::to_string(r.page_bytes) +
                                  " bytes, not " + std::to_string(bytes));
    }
    return r.base + loc.slot * r.page_bytes;
  }

  PageLocation PagedKVCache::locate(PageKey key) const
  {
    const auto it = table_.find(key);
    if (it == table_.end())
    {
      throw PageNotFound(key);
    }
    return it->second;
  }

  bool PagedKVCache::contains(PageKey key) const
  {
    return table_.count(key) != 0;
  }

  const std::vector<PageKey> &PagedKVCache::pages(Tier tier) const
  {
    return range(tier).slots;
  }

  std::size_t PagedKVCache::page_bytes(Tier tier) const
  {
    return range(tier).page_bytes;
  }

  std::size_t PagedKVCache::reserved_bytes(Tier tier) const
  {
    return range(tier).reserved;
  }

  std::size_t PagedKVCache::mapped_bytes(Tier tier) const
  {
    return range(tier).granules.size() * granule_;
  }

} // namespace microinfer
