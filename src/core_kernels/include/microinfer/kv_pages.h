#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <memory>

#include "microinfer/device_buffer.h"
#include "microinfer/paged_kv_cache.h"

// P, the tokens per page, is a build-time parameter (ADR-0004): 32 by
// default, and set with -DMICROINFER_PAGE_TOKENS for Milestone 3's ablation
// over 16, 32 and 64. It is defined by CMake and nowhere else, so no source
// file can quietly assume a value.
#ifndef MICROINFER_PAGE_TOKENS
#error "MICROINFER_PAGE_TOKENS must be defined by the build (CMakeLists.txt)"
#endif

namespace microinfer
{

  constexpr int kPageTokens = MICROINFER_PAGE_TOKENS;

  // The engine's KV cache, on pages (#14): one layer's keys and values for
  // page_tokens consecutive positions per page, allocated from a
  // PagedKVCache as the sequence grows.
  //
  // A page holds page_tokens rows of keys, then page_tokens rows of values,
  // each row (kv_heads, head_dim) in fp16. Page `i` of layer `l` is the page
  // table entry (l, i), and holds positions [i * page_tokens, (i + 1) *
  // page_tokens).
  //
  // The allocator may move any page whenever it frees one (ADR-0007), so this
  // class never keeps a page's address. store() and attention() each resolve
  // the layer's page table immediately before their launch, into device
  // memory the launch alone reads.
  class KVPages
  {
  public:
    // `row` is kv_heads * head_dim. The allocator's page size for `tier` must
    // be exactly what that layout needs, and the allocator must outlive this.
    KVPages(PagedKVCache &allocator, int layers, int page_tokens,
            std::size_t row, Tier tier);
    // Frees every page this cache holds.
    ~KVPages();

    KVPages(const KVPages &) = delete;
    KVPages &operator=(const KVPages &) = delete;

    // Makes room for positions [0, tokens) in every layer, allocating only the
    // pages not already held: on demand, never for a maximum length.
    void reserve(int tokens);

    // Writes n tokens' keys and values, (n, row) each, at positions
    // [start, start + n) of `layer`.
    void store(int layer, const __half *keys, const __half *values, int start,
               int n);

    // Attention over positions [0, seq_k) of `layer`; see attention_paged.
    void attention(int layer, const __half *q, const __half *k_bias,
                   __half *out, int seq_q, int seq_k, int heads, int kv_heads,
                   int head_dim, double theta);

    int page_tokens() const { return page_tokens_; }
    int pages_per_layer() const { return pages_; }
    int capacity_tokens() const { return pages_ * page_tokens_; }
    std::size_t page_bytes() const { return page_bytes_; }

  private:
    // The layer's first `count` page addresses, copied to the device.
    // Valid for the launch that follows and nothing after it.
    const unsigned long long *resolve(int layer, int count);

    PagedKVCache &allocator_;
    int layers_;
    int page_tokens_;
    std::size_t row_;
    Tier tier_;
    std::size_t page_bytes_;
    int pages_ = 0;
    std::unique_ptr<DeviceBuffer> table_;
    int table_capacity_ = 0;
  };

} // namespace microinfer
