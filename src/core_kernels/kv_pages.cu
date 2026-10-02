#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "microinfer/check.h"
#include "microinfer/device_ops.h"
#include "microinfer/kv_pages.h"
#include "microinfer/launch.h"
#include "microinfer/quant.h"

namespace microinfer
{
  namespace
  {

    constexpr int kBlockThreads = 256;

    __global__ void
    store_pages_kernel(const __half *__restrict__ keys,
                       const __half *__restrict__ values,
                       const unsigned long long *__restrict__ pages,
                       int page_tokens, size_t row, int start, size_t count)
    {
      for (size_t i =
               static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
           i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
      {
        const int position = start + static_cast<int>(i / row);
        __half *page =
            reinterpret_cast<__half *>(pages[position / page_tokens]);
        __half *key = page + (position % page_tokens) * row + i % row;
        key[0] = keys[i];
        key[static_cast<size_t>(page_tokens) * row] = values[i];
      }
    }

  } // namespace

  void device::store_pages(const __half *keys, const __half *values,
                           const unsigned long long *pages, int page_tokens,
                           size_t row, int start, int n)
  {
    const size_t count = static_cast<size_t>(n) * row;
    if (count == 0)
    {
      return;
    }
    store_pages_kernel<<<grid_stride_blocks(count, kBlockThreads),
                         kBlockThreads>>>(keys, values, pages, page_tokens, row,
                                          start, count);
    cuda_check(cudaGetLastError(), "store_pages kernel launch");
  }

  std::size_t KVPages::page_bytes_for(int page_tokens, std::size_t kv_width)
  {
    return 2 * static_cast<std::size_t>(page_tokens) * kv_width *
           sizeof(__half);
  }

  bool KVPages::seals_for(Tier tier, const TierMap &tier_map, bool always_seal)
  {
    if (always_seal || tier != Tier::FP16)
    {
      return true;
    }
    for (const auto &row : tier_map)
    {
      for (Tier t : row)
      {
        if (t != tier)
        {
          return true;
        }
      }
    }
    return false;
  }

  KVPages::KVPages(PagedKVCache &allocator, int layers, int page_tokens,
                   int kv_heads, int head_dim, Tier tier, Halves halves,
                   TierMap tier_map, bool always_seal)
      : allocator_(allocator), layers_(layers), page_tokens_(page_tokens),
        kv_heads_(kv_heads), head_dim_(head_dim),
        kv_width_(static_cast<std::size_t>(kv_heads) * head_dim), tier_(tier),
        halves_(halves),
        storage_tier_(tier != Tier::FP16 && halves != Halves::Both ? Tier::FP16
                                                                   : tier),
        tier_map_(std::move(tier_map)),
        current_open_(layers > 0 ? layers : 0, 0),
        attend_open_(layers > 0 ? layers : 0, 0),
        sealed_(layers > 0 ? layers : 0, 0),
        shadows_(page_bytes_for(page_tokens, kv_width_))
  {
    if (layers <= 0 || page_tokens <= 0 || kv_heads <= 0 || head_dim <= 0)
    {
      throw std::invalid_argument(
          "layers, page_tokens, kv_heads and head_dim must be positive");
    }
    if (tier == Tier::FP16 && halves != Halves::Both)
    {
      throw std::invalid_argument(
          "only a quantised tier can quantise one half of a page");
    }
    if (!tier_map_.empty() && static_cast<int>(tier_map_.size()) != layers)
    {
      throw std::invalid_argument("a tier map needs a row for every layer: " +
                                  std::to_string(tier_map_.size()) +
                                  " rows for " + std::to_string(layers) +
                                  " layers");
    }
    // A diagnostic's round trip is at the cache's one tier by its
    // definition, so a map has nothing to say to it.
    if (!tier_map_.empty() && halves != Halves::Both)
    {
      throw std::invalid_argument(
          "a diagnostic Halves rounds every page at the cache's one tier, and "
          "takes no tier map");
    }
    may_be_at_[static_cast<int>(storage_tier_)] = true;
    for (const auto &row : tier_map_)
    {
      for (Tier t : row)
      {
        may_be_at_[static_cast<int>(t)] = true;
        mixed_ = mixed_ || t != tier;
      }
    }
    seals_ = seals_for(tier, tier_map_, always_seal);
    page_bytes_ = bytes_at(storage_tier_);
    for (int t = 0; t < kTierCount; ++t)
    {
      if (may_be_at_[t])
      {
        require_page_size(static_cast<Tier>(t), "a page of positions");
      }
    }
    if (seals_)
    {
      require_page_size(Tier::FP16, "an open page");
    }
    if (halves != Halves::Both)
    {
      scratch_page_ = std::make_unique<DeviceBuffer>(bytes_at(tier));
      scratch_halves_ = std::make_unique<DeviceBuffer>(bytes_at(Tier::FP16));
    }
  }

  void KVPages::require_layer(int layer) const
  {
    if (layer < 0 || layer >= layers_)
    {
      throw std::out_of_range("layer " + std::to_string(layer) + " of " +
                              std::to_string(layers_));
    }
  }

  std::size_t KVPages::bytes_at(Tier tier) const
  {
    return tier == Tier::FP16
               ? page_bytes_for(page_tokens_, kv_width_)
               : quantised_page_layout(tier, page_tokens_, kv_heads_, head_dim_)
                     .page_bytes;
  }

  void KVPages::require_page_size(Tier tier, const char *what) const
  {
    const std::size_t needed = bytes_at(tier);
    if (allocator_.page_bytes(tier) != needed)
    {
      throw std::invalid_argument(
          std::string("the allocator's ") + tier_name(tier) + " pages hold " +
          std::to_string(allocator_.page_bytes(tier)) + " bytes; " + what +
          " of " + std::to_string(page_tokens_) + " positions at " +
          std::to_string(kv_width_) + " elements needs " +
          std::to_string(needed));
    }
  }

  KVPages::~KVPages()
  {
    // A destructor cannot throw, and a page this cache allocated is always
    // freeable; an error here would mean the allocator itself is broken.
    try
    {
      for (int page = pages_ - 1; page >= 0; --page)
      {
        for (int layer = layers_ - 1; layer >= 0; --layer)
        {
          allocator_.free({layer, page});
        }
      }
      if (open_pages_)
      {
        for (int k = static_cast<int>(kOpenPages.size()) - 1; k >= 0; --k)
        {
          for (int layer = layers_ - 1; layer >= 0; --layer)
          {
            allocator_.free({layer, kOpenPages[k]});
          }
        }
      }
    }
    catch (...)
    {
    }
  }

  void KVPages::reserve(int tokens)
  {
    // What this call takes, in order, so that a failure gives it all back
    // newest first, and the cache is as it was.
    std::vector<PageKey> taken;
    try
    {
      if (seals() && !open_pages_ && tokens > 0)
      {
        for (int page : kOpenPages)
        {
          for (int layer = 0; layer < layers_; ++layer)
          {
            allocator_.allocate({layer, page}, Tier::FP16);
            taken.push_back({layer, page});
          }
        }
      }
      // In a cache that seals, a page is taken only once all its positions
      // are coming: until then they are an open page's.
      const int needed = seals() ? tokens / page_tokens_
                                 : (tokens + page_tokens_ - 1) / page_tokens_;
      for (int page = pages_; page < needed; ++page)
      {
        for (int layer = 0; layer < layers_; ++layer)
        {
          allocator_.allocate({layer, page}, birth_tier(layer, page));
          taken.push_back({layer, page});
        }
      }
      if (seals() && tokens > 0)
      {
        open_pages_ = true;
      }
      pages_ = needed > pages_ ? needed : pages_;
    }
    catch (...)
    {
      while (!taken.empty())
      {
        allocator_.free(taken.back());
        taken.pop_back();
      }
      throw;
    }
  }

  Tier KVPages::birth_tier(int layer, int page) const
  {
    if (layer >= 0 && layer < static_cast<int>(tier_map_.size()) && page >= 0 &&
        page < static_cast<int>(tier_map_[layer].size()))
    {
      return tier_map_[layer][page];
    }
    return storage_tier_;
  }

  Tier KVPages::page_tier(PageKey key) const
  {
    return allocator_.locate(key).tier;
  }

  KVPages::ResolvedPage KVPages::resolve_page(PageKey key) const
  {
    const Tier tier = allocator_.locate(key).tier;
    return {tier, allocator_.address(key, allocator_.page_bytes(tier))};
  }

  KVPages::LayerTable KVPages::resolve(int layer)
  {
    require_layer(layer);
    // At a quantised tier a cache can hold no page of positions yet, while
    // its first page fills in an open page; then there is no table.
    if (pages_ == 0)
    {
      return {nullptr, nullptr};
    }
    if (resolved_pages_ != pages_ ||
        resolved_generation_ != allocator_.generation_)
    {
      if (pages_ > table_stride_)
      {
        // Room to grow before the next reallocation. Freeing the old buffer
        // waits for launches that may still read it, as cudaFree does.
        table_stride_ = pages_ > 2 * table_stride_ ? pages_ : 2 * table_stride_;
        tables_ = std::make_unique<DeviceBuffer>(
            static_cast<size_t>(layers_) * table_stride_ *
            (sizeof(unsigned long long) + sizeof(std::uint8_t)));
      }
      // The addresses of every layer's pages, then each page's tier, in the
      // same order: one buffer, one upload.
      const size_t entries = static_cast<size_t>(layers_) * table_stride_;
      std::vector<unsigned long long> host(
          entries + (entries + sizeof(unsigned long long) - 1) /
                        sizeof(unsigned long long),
          0);
      auto *tiers = reinterpret_cast<std::uint8_t *>(host.data() + entries);
      for (int l = 0; l < layers_; ++l)
      {
        for (int page = 0; page < pages_; ++page)
        {
          const ResolvedPage resolved = resolve_page({l, page});
          const size_t at = static_cast<size_t>(l) * table_stride_ + page;
          host[at] = resolved.address;
          tiers[at] = static_cast<std::uint8_t>(resolved.tier);
        }
      }
      // On the legacy default stream, so ordered after every launch already
      // enqueued, which may still be reading the tables this overwrites.
      // Moving to other streams would have to order this explicitly.
      cuda_check(cudaMemcpy(tables_->raw(), host.data(),
                            entries * (sizeof(unsigned long long) +
                                       sizeof(std::uint8_t)),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy page tables host-to-device");
      resolved_pages_ = pages_;
      resolved_generation_ = allocator_.generation_;
    }
    // table_stride_ as it is now, after any growth above.
    const size_t entries = static_cast<size_t>(layers_) * table_stride_;
    const size_t at = static_cast<size_t>(layer) * table_stride_;
    return {tables_->as<const unsigned long long>() + at,
            tables_->as<const std::uint8_t>() +
                entries * sizeof(unsigned long long) + at};
  }

  __half *KVPages::open_page(int layer, int which)
  {
    return reinterpret_cast<__half *>(
        resolve_page({layer, kOpenPages[which]}).address);
  }

  void KVPages::store(int layer, const __half *keys, const __half *values,
                      int start, int n)
  {
    if (start < 0 || n < 0 || start + n > capacity_tokens())
    {
      throw std::out_of_range(
          "positions [" + std::to_string(start) + ", " +
          std::to_string(start + n) + ") are not all on pages; " +
          std::to_string(capacity_tokens()) + " positions are reserved");
    }
    require_layer(layer);
    if (n == 0)
    {
      return;
    }
    if (seals())
    {
      store_sealed(layer, keys, values, start, n);
      return;
    }
    device::store_pages(keys, values, resolve(layer).pages, page_tokens_,
                        kv_width_, start, n);
  }

  void KVPages::seal(int layer, int span, const __half *keys,
                     const __half *values)
  {
    // Resolved here, between allocator operations, and used at once: nothing
    // below allocates or frees. The page is sealed at the tier the page
    // table records for it.
    const ResolvedPage resolved = resolve_page({layer, span});
    auto *page = reinterpret_cast<std::uint8_t *>(resolved.address);
    sealed_[layer] = std::max(sealed_[layer], span + 1);
    if (halves_ == Halves::Both && resolved.tier == Tier::FP16)
    {
      // A page born at FP16 in a cache that seals: its rows as they came,
      // as the FP16 layout holds them.
      copy_rows(reinterpret_cast<__half *>(page), keys, values);
      return;
    }
    if (halves_ == Halves::Both)
    {
      device::quantise_page(keys, values, page, resolved.tier, page_tokens_,
                            page_tokens_, kv_heads_, head_dim_);
      return;
    }
    // The diagnostic: the tier's round trip for one half, the other as it
    // came, both into an FP16 page. The round trip's tier is the cache's,
    // tier_, by definition of the diagnostic; the page it writes is FP16.
    const std::size_t half = static_cast<std::size_t>(page_tokens_) * kv_width_;
    auto *round_trip = scratch_halves_->as<__half>();
    device::quantise_page(keys, values, scratch_page_->as<std::uint8_t>(),
                          tier_, page_tokens_, page_tokens_, kv_heads_,
                          head_dim_);
    device::dequantise_page(scratch_page_->as<const std::uint8_t>(), round_trip,
                            round_trip + half, tier_, page_tokens_, kv_heads_,
                            head_dim_);
    const __half *k = halves_ == Halves::Keys ? round_trip : keys;
    const __half *v = halves_ == Halves::Values ? round_trip + half : values;
    copy_rows(reinterpret_cast<__half *>(page), k, v);
  }

  void KVPages::copy_rows(__half *page, const __half *keys,
                          const __half *values)
  {
    const std::size_t half = static_cast<std::size_t>(page_tokens_) * kv_width_;
    const std::size_t bytes = half * sizeof(__half);
    cuda_check(cudaMemcpy(page, keys, bytes, cudaMemcpyDeviceToDevice),
               "cudaMemcpy sealed keys");
    cuda_check(cudaMemcpy(page + half, values, bytes, cudaMemcpyDeviceToDevice),
               "cudaMemcpy sealed values");
  }

  void KVPages::store_sealed(int layer, const __half *keys,
                             const __half *values, int start, int n)
  {
    const std::size_t half = static_cast<std::size_t>(page_tokens_) * kv_width_;
    const int first = start / page_tokens_;
    // The open page holding the first page's earlier positions, if it has
    // any. Attention will read them there, so a later page of this store
    // must go into the other open page.
    const int current = current_open_[layer];
    attend_open_[layer] = current;

    const int end = start + n;
    for (int at = start; at < end;)
    {
      const int span = at / page_tokens_;
      const int span_start = span * page_tokens_;
      const int take = std::min(end, span_start + page_tokens_) - at;
      const std::size_t from = static_cast<std::size_t>(at - start) * kv_width_;
      if (take == page_tokens_)
      {
        seal(layer, span, keys + from, values + from);
      }
      else
      {
        const int which = span == first ? current : 1 - current;
        __half *open = open_page(layer, which);
        const std::size_t row = static_cast<std::size_t>(at - span_start);
        const std::size_t bytes = take * kv_width_ * sizeof(__half);
        cuda_check(cudaMemcpy(open + row * kv_width_, keys + from, bytes,
                              cudaMemcpyDeviceToDevice),
                   "cudaMemcpy keys into an open page");
        cuda_check(cudaMemcpy(open + half + row * kv_width_, values + from,
                              bytes, cudaMemcpyDeviceToDevice),
                   "cudaMemcpy values into an open page");
        if (at + take == span_start + page_tokens_)
        {
          seal(layer, span, open, open + half);
        }
        current_open_[layer] = which;
      }
      at += take;
    }
  }

  void KVPages::downgrade(int layer, int page, Tier target)
  {
    if (!seals())
    {
      throw std::invalid_argument(
          "only a cache that seals can downgrade a page: in one that does not, "
          "every page is FP16 and its last is still being written (ADR-0011)");
    }
    if (halves_ != Halves::Both)
    {
      throw std::invalid_argument("a diagnostic Halves stores every page at "
                                  "FP16 by its definition; none downgrades");
    }
    require_layer(layer);
    const std::string name =
        "page " + std::to_string(page) + " of layer " + std::to_string(layer);
    if (page < 0 || page >= pages_)
    {
      throw std::invalid_argument(name +
                                  " is not a page of positions this cache "
                                  "holds; only one of those downgrades");
    }
    if (page >= sealed_[layer])
    {
      throw std::invalid_argument(
          name + " is not sealed: its positions are reserved, not all stored");
    }
    if (target == Tier::FP16)
    {
      throw std::invalid_argument("a downgrade is to a quantised tier");
    }
    const PageKey key{layer, page};
    const Tier current = page_tier(key);
    if (!is_lower(target, current))
    {
      throw std::invalid_argument(name + " is at " + tier_name(current) +
                                  "; a downgrade is to a lower "
                                  "tier, and " +
                                  tier_name(target) + " is not");
    }
    // Below FP16 the page's own bytes are codes, and a downgrade quantises
    // from FP16 only (#95): from the page's shadow, which a page born at a
    // quantised tier never had.
    const bool from_shadow = current != Tier::FP16;
    if (from_shadow && !shadows_.contains(key))
    {
      throw std::invalid_argument(
          name + " was born at " + tier_name(current) +
          " and has no FP16 shadow to quantise from; only a page sealed at "
          "FP16 downgrades");
    }
    require_page_size(target, "a page of positions");

    // 1. The page at its target tier, under a key of its own while the
    // page it replaces is still read; and, from a shadow, an FP16 page to
    // upload it to, from the allocator, so that the cache's memory is all
    // the allocator's (ADR-0007), freed again before this returns. If
    // either fails, nothing has changed.
    const PageKey staged{layer, kStagingPage};
    const PageKey upload{layer, kShadowUploadPage};
    allocator_.allocate(staged, target);
    if (from_shadow)
    {
      try
      {
        allocator_.allocate(upload, Tier::FP16);
      }
      catch (...)
      {
        allocator_.free(staged);
        throw;
      }
    }
    try
    {
      // 2. Quantised from FP16, exactly as a page born at the target is
      // from its rows: the page itself while it is FP16, its shadow after
      // (#95), so that a page at tier T is always quantise_page(FP16, T).
      // Before its first downgrade the page's FP16 bytes go to its shadow
      // (#94), and the copy is complete before the page is freed below.
      // Pages are resolved here, between allocator operations, and used at
      // once.
      const __half *rows = nullptr;
      if (!from_shadow)
      {
        rows = reinterpret_cast<const __half *>(resolve_page(key).address);
        if (!shadows_.contains(key))
        {
          shadows_.take(key, rows);
        }
      }
      else
      {
        rows = reinterpret_cast<const __half *>(resolve_page(upload).address);
        cuda_check(cudaMemcpy(const_cast<__half *>(rows), shadows_.shadow(key),
                              bytes_at(Tier::FP16), cudaMemcpyHostToDevice),
                   "cudaMemcpy a page's FP16 shadow host-to-device");
      }
      device::quantise_page(
          rows, rows + static_cast<std::size_t>(page_tokens_) * kv_width_,
          reinterpret_cast<std::uint8_t *>(resolve_page(staged).address),
          target, page_tokens_, page_tokens_, kv_heads_, head_dim_);
      // The upload page is FP16's tail, so its free moves nothing, and it
      // synchronises, so the quantisation has read it before it goes.
      if (from_shadow)
      {
        allocator_.free(upload);
      }
    }
    catch (...)
    {
      if (from_shadow && allocator_.contains(upload))
      {
        allocator_.free(upload);
      }
      allocator_.free(staged);
      throw;
    }
    // 3 and 4. The page table names the new page, and the old page is
    // freed, its tier's tail moving into the slot (ADR-0007). The free's
    // copy is ordered after the quantisation on the legacy default stream,
    // and it synchronises before any granule is unmapped.
    allocator_.replace(key, staged);
    may_be_at_[static_cast<int>(target)] = true;
  }

  void KVPages::attention(int layer, const __half *q, const __half *k_bias,
                          const RopeTable *rope, __half *out, int seq_q,
                          int seq_k, int heads, int kv_heads, int head_dim,
                          const __half *keys, const __half *values)
  {
    if (static_cast<std::size_t>(kv_heads) * head_dim != kv_width_)
    {
      throw std::invalid_argument(
          "kv_heads * head_dim is " + std::to_string(kv_heads * head_dim) +
          "; this cache holds " + std::to_string(kv_width_) + " per token");
    }
    if (seq_k < 0 || seq_k > capacity_tokens())
    {
      throw std::out_of_range(
          "seq_k " + std::to_string(seq_k) + " exceeds the " +
          std::to_string(capacity_tokens()) + " positions on pages");
    }
    if (seq_q <= 0)
    {
      return;
    }
    if (!seals())
    {
      device::attention_paged(q, resolve(layer).pages, page_tokens_, k_bias,
                              rope, out, seq_q, seq_k, heads, kv_heads,
                              head_dim);
      return;
    }
    if (keys == nullptr || values == nullptr)
    {
      throw std::invalid_argument(
          "attention at a quantised tier reads each query's own page from "
          "the rows the queries stored; pass them");
    }
    if (!open_pages_)
    {
      throw std::logic_error("attention before any position was reserved");
    }
    // Each page is read at the tier the page table records for it (#91).
    const LayerTable table = resolve(layer);
    device::attention_paged_causal(q, table.pages, table.tiers, may_be_at_,
                                   open_page(layer, attend_open_[layer]), keys,
                                   values, page_tokens_, k_bias, rope, out,
                                   seq_q, seq_k, heads, kv_heads, head_dim);
  }

} // namespace microinfer
