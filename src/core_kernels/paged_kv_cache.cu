#include <cuda.h>

#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

#include "microinfer/check.h"
#include "microinfer/device_buffer.h"
#include "microinfer/device_ops.h"
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

  std::size_t allocation_granularity()
  {
    driver_check(cuInit(0), "cuInit");
    CUdevice device = 0;
    driver_check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    std::size_t granule = 0;
    driver_check(cuMemGetAllocationGranularity(
                     &granule, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                 "cuMemGetAllocationGranularity");
    return granule;
  }

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
      throw std::length_error(std::string(tier_name(tier)) + " is full at " +
                              std::to_string(r.capacity_pages) + " pages");
    }
    ContextScope scope(ctx_);

    r.slots.push_back(key);
    try
    {
      fit_granules(r, true);
    }
    catch (const OutOfMemory &e)
    {
      // Read before anything is given back: the headroom the failure met.
      std::size_t free_bytes = 0, total_bytes = 0;
      const bool known =
          cuMemGetInfo(&free_bytes, &total_bytes) == CUDA_SUCCESS;
      roll_back(r);
      throw OutOfMemory(
          "allocating " + describe(key) + " at " + tier_name(tier) +
          " needed a " + std::to_string(granule_) +
          "-byte granule: " + e.what() +
          (known ? "; " + std::to_string(free_bytes) + " bytes were free"
                 : std::string()));
    }
    catch (...)
    {
      roll_back(r);
      throw;
    }
    table_[key] = PageLocation{tier, r.slots.size() - 1};
    ++generation_;
  }

  void PagedKVCache::roll_back(Range &r)
  {
    r.slots.pop_back();
    try
    {
      // Give back any granule mapped before the failure.
      fit_granules(r, false);
    }
    catch (...)
    {
      // The failure being reported is the one that matters; one in giving
      // back must not replace it.
    }
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
    fit_granules(r, false);
  }

  void PagedKVCache::free_pages(const std::vector<PageKey> &keys)
  {
    std::set<PageKey> freeing;
    for (const PageKey &key : keys)
    {
      locate(key); // PageNotFound before anything changes
      if (!freeing.insert(key).second)
      {
        throw std::invalid_argument(describe(key) +
                                    " is named twice in one free");
      }
    }
    if (keys.empty())
    {
      return;
    }
    ContextScope scope(ctx_);
    // Tier by tier: the tier keeps its first `kept` slots. Each freed page
    // below there is a hole, and each kept page at or above it moves into
    // one, the copies reading only slots the tier gives up and writing only
    // slots it keeps, so that none reads what another writes. Every tier's
    // moves are worked out, and the device memory their addresses need is
    // allocated, before anything changes, so that a failure changes nothing.
    struct TierMoves
    {
      std::size_t kept = 0;
      std::vector<std::size_t> holes;
      std::vector<std::size_t> movers;
    };
    std::array<TierMoves, kTierCount> moves;
    std::array<bool, kTierCount> touched{};
    std::size_t copies = 0;
    for (int t = 0; t < kTierCount; ++t)
    {
      const Range &r = ranges_[t];
      std::size_t freed = 0;
      for (const PageKey &slot : r.slots)
      {
        freed += freeing.count(slot);
      }
      if (freed == 0)
      {
        continue;
      }
      touched[t] = true;
      TierMoves &m = moves[t];
      m.kept = r.slots.size() - freed;
      for (std::size_t slot = 0; slot < r.slots.size(); ++slot)
      {
        const bool going = freeing.count(r.slots[slot]) != 0;
        if (slot < m.kept && going)
        {
          m.holes.push_back(slot);
        }
        else if (slot >= m.kept && !going)
        {
          m.movers.push_back(slot);
        }
      }
      copies += m.holes.size();
    }
    // Each tier's sources, then its destinations, one tier after another.
    std::vector<const void *> table;
    table.reserve(2 * copies);
    for (int t = 0; t < kTierCount; ++t)
    {
      const Range &r = ranges_[t];
      for (const std::size_t slot : moves[t].movers)
      {
        table.push_back(
            reinterpret_cast<const void *>(r.base + slot * r.page_bytes));
      }
      for (const std::size_t slot : moves[t].holes)
      {
        table.push_back(
            reinterpret_cast<const void *>(r.base + slot * r.page_bytes));
      }
    }
    std::optional<DeviceBuffer> addresses;
    if (copies > 0)
    {
      addresses.emplace(table.size() * sizeof(void *));
      cuda_check(cudaMemcpy(addresses->raw(), table.data(),
                            table.size() * sizeof(void *),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy a free's tail moves");
    }

    // From here on nothing fails but the device.
    std::size_t at = 0;
    for (int t = 0; t < kTierCount; ++t)
    {
      Range &r = ranges_[t];
      const TierMoves &m = moves[t];
      if (!m.holes.empty())
      {
        const auto *on_device = addresses->as<const void *const>() + at;
        // Slots sit at multiples of the page size from a granule-aligned
        // base.
        device::copy_pages(
            on_device, const_cast<void *const *>(on_device + m.holes.size()),
            r.page_bytes, static_cast<int>(m.holes.size()),
            r.page_bytes % 16 == 0);
        at += 2 * m.holes.size();
      }
      for (std::size_t i = 0; i < m.holes.size(); ++i)
      {
        const PageKey moved = r.slots[m.movers[i]];
        r.slots[m.holes[i]] = moved;
        table_[moved].slot = m.holes[i];
      }
      if (touched[t])
      {
        r.slots.resize(m.kept);
      }
    }
    for (const PageKey &key : keys)
    {
      table_.erase(key);
    }
    ++generation_;
    // The tail moves land, and whatever the device was given before, which
    // may read pages whose granules go now, the freed pages' own among
    // them, finishes, whether or not a tail moved (#205).
    driver_check(cuCtxSynchronize(), "cuCtxSynchronize");
    for (int t = 0; t < kTierCount; ++t)
    {
      if (touched[t])
      {
        fit_granules(ranges_[t], false);
      }
    }
  }

  void PagedKVCache::replace_pages(
      const std::vector<std::pair<PageKey, PageKey>> &pairs)
  {
    std::set<PageKey> staged;
    std::vector<PageKey> keys;
    for (const auto &[key, stage] : pairs)
    {
      if (key == stage)
      {
        throw std::invalid_argument("a page cannot replace itself");
      }
      locate(stage); // PageNotFound before anything changes
      if (!staged.insert(stage).second)
      {
        throw std::invalid_argument(describe(stage) +
                                    " is named twice in one replace");
      }
      keys.push_back(key);
    }
    for (const PageKey &key : keys)
    {
      if (staged.count(key) != 0)
      {
        throw std::invalid_argument(describe(key) +
                                    " is both replaced and staged");
      }
    }
    free_pages(keys);
    // Looked up after the free, which moves a staged page that sat above
    // the slots its tier kept.
    for (const auto &[key, stage] : pairs)
    {
      const PageLocation loc = table_.at(stage);
      table_.erase(stage);
      table_[key] = loc;
      range(loc.tier).slots[loc.slot] = key;
    }
    ++generation_;
  }

  void PagedKVCache::replace(PageKey key, PageKey staged)
  {
    if (key == staged)
    {
      throw std::invalid_argument("a page cannot replace itself");
    }
    locate(staged); // PageNotFound before anything changes
    free(key);
    // Looked up after the free, which moves staged if it was key's tier's
    // tail.
    const PageLocation loc = table_.at(staged);
    table_.erase(staged);
    table_[key] = loc;
    range(loc.tier).slots[loc.slot] = key;
    ++generation_;
  }

  void PagedKVCache::map_granule(Range &r)
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

  void PagedKVCache::fit_granules(Range &r, bool map_spares)
  {
    const std::size_t needed =
        round_up(r.slots.size() * r.page_bytes, granule_) / granule_;
    const std::size_t kept =
        std::min(needed + r.spare_granules, r.reserved / granule_);

    while (r.granules.size() < needed)
    {
      map_granule(r);
    }
    try
    {
      while (map_spares && r.granules.size() < kept)
      {
        map_granule(r);
      }
    }
    catch (const OutOfMemory &)
    {
      // A spare is room kept for later, not room this page needs.
    }

    while (r.granules.size() > kept)
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

  void PagedKVCache::keep_spare_granules(Tier tier, std::size_t granules)
  {
    Range &r = range(tier);
    ContextScope scope(ctx_);
    r.spare_granules = granules;
    fit_granules(r, true);
  }

  std::size_t PagedKVCache::spare_granules(Tier tier) const
  {
    return range(tier).spare_granules;
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
