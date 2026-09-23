#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "microinfer/device_buffer.h"
#include "microinfer/paged_kv_cache.h"
#include "microinfer/rope_table.h"

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
    // The bytes one page needs: page_tokens rows of keys and as many of
    // values, kv_width fp16 elements each. The one place this layout's size
    // is written down.
    static std::size_t page_bytes_for(int page_tokens, std::size_t kv_width);

    // `kv_width` is kv_heads * head_dim, the fp16 elements in one token's
    // keys (or values). The allocator's page size for `tier` must be
    // page_bytes_for(page_tokens, kv_width), and the allocator must outlive
    // this.
    KVPages(PagedKVCache &allocator, int layers, int page_tokens,
            std::size_t kv_width, Tier tier);
    // Frees every page this cache holds, newest first, so that each free is
    // of a tail page and moves nothing.
    ~KVPages();

    KVPages(const KVPages &) = delete;
    KVPages &operator=(const KVPages &) = delete;

    // Makes room for positions [0, tokens) in every layer, allocating only the
    // pages not already held: on demand, never for a maximum length. If an
    // allocation fails, as it will when contention leaves no memory, the
    // pages taken for that position range are given back before the error
    // propagates, and the cache is as it was.
    void reserve(int tokens);

    // Writes n tokens' keys and values, (n, kv_width) each, at positions
    // [start, start + n) of `layer`.
    void store(int layer, const __half *keys, const __half *values, int start,
               int n);

    // Attention over positions [0, seq_k) of `layer`; see attention_paged.
    void attention(int layer, const __half *q, const __half *k_bias,
                   const RopeTable *rope, __half *out, int seq_q, int seq_k,
                   int heads, int kv_heads, int head_dim);

    int page_tokens() const { return page_tokens_; }
    int pages_per_layer() const { return pages_; }
    int capacity_tokens() const { return pages_ * page_tokens_; }
    std::size_t page_bytes() const { return page_bytes_; }

    std::size_t kv_width() const { return kv_width_; }

  private:
    // The layer's page table on the device, for the pages held now. Every
    // layer's table is resolved at once, and again whenever the allocator's
    // generation has moved since: no address outlives an allocate or a free,
    // anyone's, and a decode step with no allocator operation between its
    // launches uploads nothing.
    const unsigned long long *resolve(int layer);

    PagedKVCache &allocator_;
    int layers_;
    int page_tokens_;
    std::size_t kv_width_;
    Tier tier_;
    std::size_t page_bytes_;
    int pages_ = 0;
    // layers_ tables of table_stride_ entries each.
    std::unique_ptr<DeviceBuffer> tables_;
    int table_stride_ = 0;
    int resolved_pages_ = -1;
    std::uint64_t resolved_generation_ = 0;
  };

} // namespace microinfer
