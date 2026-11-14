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
  // for the life of the cache: a later downgrade quantises from it (#95),
  // and an upgrade restores the page from it (#96). About 0.9 GiB for
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

    // For a batch (#205): room for `count` more shadows, made now, so that
    // as many add()s cannot fail; then a shadow for `key` whose bytes the
    // caller writes, at the address add() returns, which the device can
    // write as it can read device_shadow(). The shadow is taken from then,
    // and its bytes are the caller's to have written before they are read.
    void reserve(std::size_t count);
    std::uint8_t *add(PageKey key);

    bool contains(PageKey key) const;
    // The shadow's bytes, page_bytes of them; PageNotFound if `key` has none.
    const std::uint8_t *shadow(PageKey key) const;
    // The same bytes, at the address the device reads them by (#205): pinned
    // host memory, mapped for the device.
    const std::uint8_t *device_shadow(PageKey key) const;

    std::size_t page_bytes() const { return page_bytes_; }
    std::size_t count() const { return shadows_.size(); }
    // Pinned host memory held: every allocation, whole.
    std::size_t bytes() const { return capacity_ * page_bytes_; }

  private:
    // One pinned allocation: its addresses on the host and on the device,
    // the pages it holds and those it has given out.
    struct Allocation
    {
      std::uint8_t *host;
      std::uint8_t *device;
      std::size_t pages;
      std::size_t used;
    };
    // The next shadow's slot, in the last allocation, once room for it is
    // held: its index there.
    std::size_t next_slot();

    std::size_t page_bytes_;
    std::size_t pages_per_allocation_;
    // Shadows fill the last allocation; a batch's reserve() makes one large
    // enough for all of its own, so that page-locking is one driver call a
    // batch, not one for every kAllocationBytes (#205).
    std::vector<Allocation> allocations_;
    std::size_t capacity_ = 0;
    // Each allocation's index, by its host address.
    std::map<const std::uint8_t *, std::size_t> by_address_;
    std::map<PageKey, std::uint8_t *> shadows_;
  };

} // namespace microinfer
