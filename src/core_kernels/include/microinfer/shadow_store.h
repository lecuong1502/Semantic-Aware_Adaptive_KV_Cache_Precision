#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "microinfer/paged_kv_cache.h"

namespace microinfer
{

  // The FP16 shadows of a cache's downgraded pages (#94): each page's FP16
  // bytes, copied to pinned host memory before its first downgrade, and kept
  // for the life of the cache. #88's later tickets will quantise further
  // downgrades from it, and copy it back for an upgrade. About 0.9 GiB for
  // Qwen2.5-1.5B at 32K, against 23 GiB of RAM, so nothing is ever evicted.
  //
  // Shadows are only ever added, so they are packed into pinned allocations
  // of about kAllocationBytes each, made as they fill: page-locking is a
  // driver call too slow to make once per page. Everything is freed when the
  // store goes.
  class ShadowStore
  {
  public:
    // About what one pinned allocation holds: whole pages, at least one. A
    // size for host memory, unrelated to the device's VMM granule.
    static constexpr std::size_t kAllocationBytes = std::size_t{2} << 20;

    explicit ShadowStore(std::size_t page_bytes);
    ~ShadowStore();

    ShadowStore(const ShadowStore &) = delete;
    ShadowStore &operator=(const ShadowStore &) = delete;

    // Copies the page_bytes at `device_page` into a new shadow for `key`, and
    // returns once the copy is complete, so that the page may then be freed.
    // A page has one shadow: taking a second is an error.
    void take(PageKey key, const void *device_page);

    bool contains(PageKey key) const;
    // The shadow's bytes, page_bytes of them; PageNotFound if `key` has none.
    const std::uint8_t *shadow(PageKey key) const;

    std::size_t page_bytes() const { return page_bytes_; }
    std::size_t count() const { return shadows_.size(); }
    // Pinned host memory held: every allocation, whole.
    std::size_t bytes() const
    {
      return allocations_.size() * pages_per_allocation_ * page_bytes_;
    }

  private:
    std::size_t page_bytes_;
    std::size_t pages_per_allocation_;
    std::vector<std::uint8_t *> allocations_;
    std::map<PageKey, std::uint8_t *> shadows_;
  };

} // namespace microinfer
