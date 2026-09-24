#pragma once

#include <cuda_fp16.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

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

  // The page_index of a layer's two open pages at a quantised tier: never a
  // position's page, so they cannot collide with one.
  constexpr std::array<int, 2> kOpenPages{-1, -2};

  // Which halves of a page a quantised tier quantises. Both is the tier; the
  // other two are a diagnostic (#18): the page is stored at FP16, and when it
  // is sealed only the named half is replaced by what the tier's quantiser
  // would return for it, so that the cost of a tier can be split between
  // keys and values. The engine never runs them unless asked.
  enum class Halves : int
  {
    Both = 0,
    Keys = 1,
    Values = 2,
  };

  // The engine's KV cache, on pages (#14): one layer's keys and values for
  // page_tokens consecutive positions per page, allocated from a
  // PagedKVCache as the sequence grows.
  //
  // An FP16 page holds page_tokens rows of keys, then page_tokens rows of
  // values, each row (kv_heads, head_dim) in fp16. Page `i` of layer `l` is
  // the page table entry (l, i), and holds positions [i * page_tokens,
  // (i + 1) * page_tokens).
  //
  // At a quantised tier (#18, ADR-0011) the tier is static. A page (l, i) is
  // allocated at the tier once all its positions are reserved, and *sealed*
  // once, when its last position is stored: quantised into place, in
  // quant.h's layout, and never written again. Until then its positions
  // are in one of the layer's two open pages, (l, kOpenPages[k]), FP16 and
  // allocated once. A key channel's scale spans a whole page, so a page
  // being filled cannot be quantised (ADR-0005). No page ever changes tier.
  //
  // Attention at a quantised tier is causal in the sense decode is: a query
  // reads the pages before its own as sealed, and its own page at FP16,
  // whether it arrived alone or in a chunk of hundreds. A chunk that begins
  // mid-page needs that page's earlier positions from the open page they are
  // in, while its own last, partial page needs an open page to go into; the
  // two open pages are for those two, so the second never overwrites the
  // first before attention has read it.
  //
  // The allocator may move any page whenever it frees one (ADR-0007), so this
  // class never keeps a page's address. store() and attention() each resolve
  // what they need immediately before their launches.
  class KVPages
  {
  public:
    // The bytes one FP16 page needs: page_tokens rows of keys and as many of
    // values, kv_width fp16 elements each. The one place this layout's size
    // is written down.
    static std::size_t page_bytes_for(int page_tokens, std::size_t kv_width);

    // The allocator's page size at the tier the pages are stored at must be
    // a page's there: page_bytes_for(page_tokens, kv_heads * head_dim) at
    // FP16, else quantised_page_layout(...).page_bytes. At a quantised tier
    // the open pages are FP16, so its FP16 pages must be FP16-sized too; with
    // Halves other than Both, the pages themselves are stored at FP16. The
    // allocator must outlive this.
    KVPages(PagedKVCache &allocator, int layers, int page_tokens, int kv_heads,
            int head_dim, Tier tier, Halves halves = Halves::Both);
    // Frees every page this cache holds, newest first, so that each free is
    // of a tail page and moves nothing.
    ~KVPages();

    KVPages(const KVPages &) = delete;
    KVPages &operator=(const KVPages &) = delete;

    // Makes room for positions [0, tokens) in every layer, allocating only the
    // pages not already held: on demand, never for a maximum length. At a
    // quantised tier those are the pages whose every position is below
    // `tokens`, and the open pages, on the first call. If an allocation
    // fails, as it will when contention leaves no memory, everything taken
    // in the call is given back before the error propagates, and the cache
    // is as it was.
    void reserve(int tokens);

    // Writes n tokens' keys and values, (n, kv_width) each, at positions
    // [start, start + n) of `layer`. At a quantised tier a page whose last
    // position this writes is sealed.
    void store(int layer, const __half *keys, const __half *values, int start,
               int n);

    // Attention over positions [0, seq_k) of `layer`, which must all have
    // been stored, for queries at the last seq_q of them. At a quantised
    // tier `keys` and `values` are the rows the queries stored, (seq_q,
    // kv_width) each, from which each query reads its own page (see above);
    // at FP16 they are not read and may be null.
    void attention(int layer, const __half *q, const __half *k_bias,
                   const RopeTable *rope, __half *out, int seq_q, int seq_k,
                   int heads, int kv_heads, int head_dim, const __half *keys,
                   const __half *values);

    int page_tokens() const { return page_tokens_; }
    // Pages of positions held in each layer; the open pages are not counted.
    int pages_per_layer() const { return pages_; }
    int capacity_tokens() const
    {
      return (pages_ + (quantised() ? 1 : 0)) * page_tokens_;
    }
    // The size of one page of positions, as it is stored.
    std::size_t page_bytes() const { return page_bytes_; }
    Tier tier() const { return tier_; }
    Halves halves() const { return halves_; }
    bool quantised() const { return tier_ != Tier::FP16; }
    // The tier the pages of positions are allocated at: the cache's own,
    // except under a diagnostic Halves, where it is FP16.
    Tier storage_tier() const { return storage_tier_; }

    std::size_t kv_width() const { return kv_width_; }

  private:
    // store() at a quantised tier, span by span (ADR-0011).
    void store_quantised(int layer, const __half *keys, const __half *values,
                         int start, int n);
    // Quantises the page of positions `span` from rows (page_tokens, kv_width)
    // of keys and of values, into its place.
    void seal(int layer, int span, const __half *keys, const __half *values);
    __half *open_page(int layer, int which);

    // The layer's table of pages of positions on the device, for the pages
    // held now. Every layer's table is resolved at once, and again whenever
    // the allocator's generation has moved since: no address outlives an
    // allocate or a free, anyone's, and a decode step with no allocator
    // operation between its launches uploads nothing.
    const unsigned long long *resolve(int layer);

    PagedKVCache &allocator_;
    int layers_;
    int page_tokens_;
    int kv_heads_;
    int head_dim_;
    std::size_t kv_width_;
    Tier tier_;
    Halves halves_;
    Tier storage_tier_;
    std::size_t page_bytes_;
    int pages_ = 0;
    bool open_pages_ = false;
    // Per layer: which open page holds the positions of its last, partial
    // page, and which held the first stored page's earlier positions at the
    // last store, which attention reads.
    std::vector<int> current_open_;
    std::vector<int> attend_open_;
    // For a diagnostic Halves only: a quantised page and its dequantised
    // halves, the scratch a seal goes through.
    std::unique_ptr<DeviceBuffer> scratch_page_;
    std::unique_ptr<DeviceBuffer> scratch_halves_;
    // layers_ tables of table_stride_ entries each.
    std::unique_ptr<DeviceBuffer> tables_;
    int table_stride_ = 0;
    int resolved_pages_ = -1;
    std::uint64_t resolved_generation_ = 0;
  };

} // namespace microinfer
