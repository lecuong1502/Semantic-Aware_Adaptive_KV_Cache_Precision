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

  // The page_index of a layer's open page at a quantised tier: never a
  // position's page, so it cannot collide with one.
  constexpr int kOpenPage = -1;

  // The engine's KV cache, on pages (#14): one layer's keys and values for
  // page_tokens consecutive positions per page, allocated from a
  // PagedKVCache as the sequence grows.
  //
  // An FP16 page holds page_tokens rows of keys, then page_tokens rows of
  // values, each row (kv_heads, head_dim) in fp16. Page `i` of layer `l` is
  // the page table entry (l, i), and holds positions [i * page_tokens,
  // (i + 1) * page_tokens).
  //
  // At a quantised tier (#18) the cache's tier is static: every page (l, i)
  // is allocated at that tier and written once, when its last position
  // arrives, in quant.h's layout. Until then its positions live in the
  // layer's open page, (l, kOpenPage): FP16, allocated once, and reused for
  // each span in turn. A key channel's scale spans a whole page, so a page
  // being filled cannot be quantised (ADR-0005). No page ever changes tier.
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

    // The allocator's page size for `tier` must be what a page at that tier
    // needs: page_bytes_for(page_tokens, kv_heads * head_dim) at FP16, or
    // quantised_page_layout(...).page_bytes. At a quantised tier its FP16
    // pages, which hold the open pages, must be FP16-sized too. The
    // allocator must outlive this.
    KVPages(PagedKVCache &allocator, int layers, int page_tokens, int kv_heads,
            int head_dim, Tier tier);
    // Frees every page this cache holds, newest first, so that each free is
    // of a tail page and moves nothing.
    ~KVPages();

    KVPages(const KVPages &) = delete;
    KVPages &operator=(const KVPages &) = delete;

    // Makes room for positions [0, tokens) in every layer, allocating only the
    // pages not already held: on demand, never for a maximum length. At a
    // quantised tier those are the pages whose every position is below
    // `tokens`, and the open pages, on the first call. If an allocation
    // fails, as it will when contention leaves no memory, the pages taken
    // for that position range are given back before the error propagates,
    // and the cache is as it was.
    void reserve(int tokens);

    // Writes n tokens' keys and values, (n, kv_width) each, at positions
    // [start, start + n) of `layer`. At a quantised tier, a page whose last
    // position this writes is quantised into place, from the rows given if
    // they cover it whole and from the open page if they finish it.
    void store(int layer, const __half *keys, const __half *values, int start,
               int n);

    // Attention over positions [0, seq_k) of `layer`; see attention_paged.
    void attention(int layer, const __half *q, const __half *k_bias,
                   const RopeTable *rope, __half *out, int seq_q, int seq_k,
                   int heads, int kv_heads, int head_dim);

    int page_tokens() const { return page_tokens_; }
    // Pages at the cache's tier in each layer; the open pages are not counted.
    int pages_per_layer() const { return pages_; }
    int capacity_tokens() const
    {
      return (pages_ + (quantised() ? 1 : 0)) * page_tokens_;
    }
    // The size of a page at the cache's tier.
    std::size_t page_bytes() const { return page_bytes_; }
    Tier tier() const { return tier_; }
    bool quantised() const { return tier_ != Tier::FP16; }

    std::size_t kv_width() const { return kv_width_; }

  private:
    // store() at a quantised tier, span by span: a span the rows cover whole
    // is quantised straight into its page; any other part of a span is
    // copied into the open page, which is quantised into the span's page
    // once the rows complete it.
    void store_quantised(int layer, const __half *keys, const __half *values,
                         int start, int n);
    void allocate_open_pages();

    // The layer's page table on the device, for the pages held now: at a
    // quantised tier, the open page's address follows the tier's pages. Every
    // layer's table is resolved at once, and again whenever the allocator's
    // generation has moved since: no address outlives an allocate or a free,
    // anyone's, and a decode step with no allocator operation between its
    // launches uploads nothing.
    const unsigned long long *resolve(int layer);

    PagedKVCache &allocator_;
    int layers_;
    int page_tokens_;
    int kv_heads_;
    int head_dim_;
    std::size_t kv_width_;
    Tier tier_;
    std::size_t page_bytes_;
    int pages_ = 0;
    bool open_pages_ = false;
    // layers_ tables of table_stride_ entries each.
    std::unique_ptr<DeviceBuffer> tables_;
    int table_stride_ = 0;
    int resolved_pages_ = -1;
    std::uint64_t resolved_generation_ = 0;
  };

} // namespace microinfer
