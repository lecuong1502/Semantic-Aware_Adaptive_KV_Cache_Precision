#pragma once

#include <cuda.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace microinfer
{

  // The four precision tiers (ADR-0008), in descending order of precision.
  enum class Tier : int
  {
    FP16 = 0,
    INT8 = 1,
    INT4 = 2,
    INT2 = 3,
  };

  constexpr int kTierCount = 4;

  // A logical page: the keys and values for one span of token positions in one
  // layer. The page table is keyed by this pair and nothing else (CONTEXT.md).
  struct PageKey
  {
    int layer;
    int page_index;

    bool operator<(const PageKey &o) const
    {
      return std::pair(layer, page_index) < std::pair(o.layer, o.page_index);
    }
  };

  // Asked about a page the page table does not hold.
  class PageNotFound : public std::out_of_range
  {
  public:
    explicit PageNotFound(PageKey key);
  };

  // Where a page lives: its tier, and its slot, the page's ordinal position
  // from the low end of that tier's address range.
  struct PageLocation
  {
    Tier tier;
    std::size_t slot;
  };

  // The KV cache allocator of ADR-0007, built on the CUDA virtual memory
  // management API so that emptied memory returns to the *driver*, where
  // nvmlDeviceGetMemoryInfo can see it. A page recycled inside a pool the
  // engine still held would be invisible there, and the response to pressure
  // would be a no-op that cannot be measured.
  //
  // Each tier owns one reserved virtual address range. Its pages are packed
  // from the low end, slot 0 upwards, with no holes: slot i lives at byte
  // i * page_bytes of the range. Physical granules back the range only as far
  // as the packed pages reach, so memory held is always the whole granules the
  // live pages need and never more.
  //
  // Freeing a page moves the tier's last page into the vacated slot (one
  // device-to-device copy of one page), then retracts the tail. Holes are
  // therefore impossible by construction, and freeing costs the same however
  // many pages a tier holds.
  //
  // Because a page moves when its tier's tail is retracted, its device address
  // is never handed out. Callers name pages by PageKey; the only way to reach a
  // page's bytes from outside is through write() and read(), which resolve the
  // key at the time of the call.
  //
  // Page sizes are parameters, one per tier. The allocator never interprets
  // what a page holds, so it knows nothing of models, head counts or P.
  class PagedKVCache
  {
  public:
    // page_bytes[t] is the size of one page at tier t. capacity_pages[t] is
    // how many pages tier t can ever hold at once; that many pages' worth of
    // address space is reserved, rounded up to whole granules. Reserving
    // address space allocates no device memory.
    PagedKVCache(const std::array<std::size_t, kTierCount> &page_bytes,
                 const std::array<std::size_t, kTierCount> &capacity_pages);
    ~PagedKVCache();

    PagedKVCache(const PagedKVCache &) = delete;
    PagedKVCache &operator=(const PagedKVCache &) = delete;

    // Places a new page at the tail of its tier, mapping another granule if
    // the tail crosses into one. The contents are undefined until written.
    void allocate(PageKey key, Tier tier);

    // Removes a page: the tier's tail page moves into its slot, and any
    // granule the retracted tail leaves empty is unmapped and released.
    void free(PageKey key);

    // Copies a page's bytes in from, or out to, host memory. `bytes` must be
    // exactly page_bytes(tier) of the page's tier.
    void write(PageKey key, const void *host, std::size_t bytes);
    void read(PageKey key, void *host, std::size_t bytes) const;

    PageLocation locate(PageKey key) const;
    bool contains(PageKey key) const;

    // The pages of a tier in slot order: element i occupies slot i.
    const std::vector<PageKey> &pages(Tier tier) const;

    std::size_t page_bytes(Tier tier) const;
    std::size_t reserved_bytes(Tier tier) const;
    // Device memory currently backing the tier: whole granules only.
    std::size_t mapped_bytes(Tier tier) const;
    // From cuMemGetAllocationGranularity, never assumed.
    std::size_t granule_bytes() const { return granule_; }

  private:
    struct Range
    {
      CUdeviceptr base = 0;
      std::size_t reserved = 0;
      std::size_t page_bytes = 0;
      std::size_t capacity_pages = 0;
      std::vector<PageKey> slots;
      std::vector<CUmemGenericAllocationHandle> granules;
    };

    Range &range(Tier tier) { return ranges_[static_cast<int>(tier)]; }
    const Range &range(Tier tier) const
    {
      return ranges_[static_cast<int>(tier)];
    }
    // Private on purpose: see the class comment. Valid only until the next
    // free() of any page in the same tier. Checks that `bytes` is the page's
    // size, since every caller is about to copy that many.
    CUdeviceptr address(PageKey key, std::size_t bytes) const;
    // Maps or releases granules until exactly those the slots reach remain.
    void fit_granules(Range &r);
    void release_everything() noexcept;

    CUdevice device_ = 0;
    CUcontext ctx_ = nullptr;
    CUmemAllocationProp prop_{};
    std::size_t granule_ = 0;
    std::array<Range, kTierCount> ranges_;
    std::map<PageKey, PageLocation> table_;
  };

} // namespace microinfer
