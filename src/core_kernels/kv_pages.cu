#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
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

  namespace
  {
    const char *tier_name(Tier tier)
    {
      switch (tier)
      {
      case Tier::FP16:
        return "FP16";
      case Tier::INT8:
        return "INT8";
      case Tier::INT4:
        return "INT4";
      case Tier::INT2:
        return "INT2";
      }
      return "an unknown tier";
    }
  } // namespace

  KVPages::KVPages(PagedKVCache &allocator, int layers, int page_tokens,
                   int kv_heads, int head_dim, Tier tier, Halves halves)
      : allocator_(allocator), layers_(layers), page_tokens_(page_tokens),
        kv_heads_(kv_heads), head_dim_(head_dim),
        kv_width_(static_cast<std::size_t>(kv_heads) * head_dim), tier_(tier),
        halves_(halves),
        storage_tier_(tier != Tier::FP16 && halves != Halves::Both ? Tier::FP16
                                                                   : tier),
        current_open_(layers > 0 ? layers : 0, 0),
        attend_open_(layers > 0 ? layers : 0, 0)
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
    const std::size_t fp16_bytes = page_bytes_for(page_tokens, kv_width_);
    const QuantisedPageLayout *layout = nullptr;
    QuantisedPageLayout quantised_layout{};
    if (quantised())
    {
      quantised_layout =
          quantised_page_layout(tier, page_tokens, kv_heads, head_dim);
      layout = &quantised_layout;
    }
    page_bytes_ = storage_tier_ == Tier::FP16 ? fp16_bytes : layout->page_bytes;
    const auto check = [&](Tier t, std::size_t needed, const char *what)
    {
      if (allocator.page_bytes(t) != needed)
      {
        throw std::invalid_argument(
            std::string("the allocator's ") + tier_name(t) + " pages hold " +
            std::to_string(allocator.page_bytes(t)) + " bytes; " + what +
            " of " + std::to_string(page_tokens) + " positions at " +
            std::to_string(kv_width_) + " elements needs " +
            std::to_string(needed));
      }
    };
    check(storage_tier_, page_bytes_, "a page of positions");
    if (quantised())
    {
      check(Tier::FP16, fp16_bytes, "an open page");
    }
    if (quantised() && halves != Halves::Both)
    {
      scratch_page_ = std::make_unique<DeviceBuffer>(layout->page_bytes);
      scratch_halves_ = std::make_unique<DeviceBuffer>(fp16_bytes);
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
      if (quantised() && !open_pages_ && tokens > 0)
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
      // At a quantised tier a page is taken only once all its positions are
      // coming: until then they are an open page's.
      const int needed = quantised()
                             ? tokens / page_tokens_
                             : (tokens + page_tokens_ - 1) / page_tokens_;
      for (int page = pages_; page < needed; ++page)
      {
        for (int layer = 0; layer < layers_; ++layer)
        {
          allocator_.allocate({layer, page}, storage_tier_);
          taken.push_back({layer, page});
        }
      }
      if (quantised() && tokens > 0)
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

  const unsigned long long *KVPages::resolve(int layer)
  {
    if (layer < 0 || layer >= layers_)
    {
      throw std::out_of_range("layer " + std::to_string(layer) + " of " +
                              std::to_string(layers_));
    }
    // At a quantised tier a cache can hold no page of positions yet, while
    // its first page fills in an open page; then there is no table.
    if (pages_ == 0)
    {
      return nullptr;
    }
    if (resolved_pages_ != pages_ ||
        resolved_generation_ != allocator_.generation_)
    {
      if (pages_ > table_stride_)
      {
        // Room to grow before the next reallocation. Freeing the old buffer
        // waits for launches that may still read it, as cudaFree does.
        table_stride_ = pages_ > 2 * table_stride_ ? pages_ : 2 * table_stride_;
        tables_ = std::make_unique<DeviceBuffer>(static_cast<size_t>(layers_) *
                                                 table_stride_ *
                                                 sizeof(unsigned long long));
      }
      std::vector<unsigned long long> host(
          static_cast<size_t>(layers_) * table_stride_, 0);
      for (int l = 0; l < layers_; ++l)
      {
        for (int page = 0; page < pages_; ++page)
        {
          host[static_cast<size_t>(l) * table_stride_ + page] =
              allocator_.address({l, page}, page_bytes_);
        }
      }
      // On the legacy default stream, so ordered after every launch already
      // enqueued, which may still be reading the tables this overwrites.
      // Moving to other streams would have to order this explicitly.
      cuda_check(cudaMemcpy(tables_->raw(), host.data(),
                            host.size() * sizeof(unsigned long long),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy page tables host-to-device");
      resolved_pages_ = pages_;
      resolved_generation_ = allocator_.generation_;
    }
    return tables_->as<const unsigned long long>() +
           static_cast<size_t>(layer) * table_stride_;
  }

  __half *KVPages::open_page(int layer, int which)
  {
    return reinterpret_cast<__half *>(allocator_.address(
        {layer, kOpenPages[which]}, page_bytes_for(page_tokens_, kv_width_)));
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
    if (layer < 0 || layer >= layers_)
    {
      throw std::out_of_range("layer " + std::to_string(layer) + " of " +
                              std::to_string(layers_));
    }
    if (n == 0)
    {
      return;
    }
    if (quantised())
    {
      store_quantised(layer, keys, values, start, n);
      return;
    }
    device::store_pages(keys, values, resolve(layer), page_tokens_, kv_width_,
                        start, n);
  }

  void KVPages::seal(int layer, int span, const __half *keys,
                     const __half *values)
  {
    // Resolved here, between allocator operations, and used at once: nothing
    // below allocates or frees.
    auto *page = reinterpret_cast<std::uint8_t *>(
        allocator_.address({layer, span}, page_bytes_));
    if (halves_ == Halves::Both)
    {
      device::quantise_page(keys, values, page, tier_, page_tokens_,
                            page_tokens_, kv_heads_, head_dim_);
      return;
    }
    // The diagnostic: the tier's round trip for one half, the other as it
    // came, both into an FP16 page.
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
    auto *out = reinterpret_cast<__half *>(page);
    const std::size_t bytes = half * sizeof(__half);
    cuda_check(cudaMemcpy(out, k, bytes, cudaMemcpyDeviceToDevice),
               "cudaMemcpy sealed keys");
    cuda_check(cudaMemcpy(out + half, v, bytes, cudaMemcpyDeviceToDevice),
               "cudaMemcpy sealed values");
  }

  void KVPages::store_quantised(int layer, const __half *keys,
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
    if (!quantised())
    {
      device::attention_paged(q, resolve(layer), page_tokens_, k_bias, rope,
                              out, seq_q, seq_k, heads, kv_heads, head_dim);
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
    device::attention_paged_causal(
        q, resolve(layer), open_page(layer, attend_open_[layer]), keys, values,
        page_tokens_, tier_, storage_tier_ == Tier::FP16, k_bias, rope, out,
        seq_q, seq_k, heads, kv_heads, head_dim);
  }

} // namespace microinfer
