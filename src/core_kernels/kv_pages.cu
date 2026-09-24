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

  KVPages::KVPages(PagedKVCache &allocator, int layers, int page_tokens,
                   int kv_heads, int head_dim, Tier tier)
      : allocator_(allocator), layers_(layers), page_tokens_(page_tokens),
        kv_heads_(kv_heads), head_dim_(head_dim),
        kv_width_(static_cast<std::size_t>(kv_heads) * head_dim), tier_(tier)
  {
    if (layers <= 0 || page_tokens <= 0 || kv_heads <= 0 || head_dim <= 0)
    {
      throw std::invalid_argument(
          "layers, page_tokens, kv_heads and head_dim must be positive");
    }
    const std::size_t fp16_bytes = page_bytes_for(page_tokens, kv_width_);
    page_bytes_ = quantised() ? quantised_page_layout(tier, page_tokens,
                                                      kv_heads, head_dim)
                                    .page_bytes
                              : fp16_bytes;
    const auto check = [&](Tier t, std::size_t needed, const char *what)
    {
      if (allocator.page_bytes(t) != needed)
      {
        throw std::invalid_argument(
            "the allocator's pages at tier " +
            std::to_string(static_cast<int>(t)) + " hold " +
            std::to_string(allocator.page_bytes(t)) + " bytes; " + what +
            " of " + std::to_string(page_tokens) + " positions at " +
            std::to_string(kv_width_) + " elements need " +
            std::to_string(needed));
      }
    };
    check(tier, page_bytes_, "a page at this cache's tier");
    if (quantised())
    {
      check(Tier::FP16, fp16_bytes, "an open page");
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
        for (int layer = layers_ - 1; layer >= 0; --layer)
        {
          allocator_.free({layer, kOpenPage});
        }
      }
    }
    catch (...)
    {
    }
  }

  void KVPages::allocate_open_pages()
  {
    int allocated = 0;
    try
    {
      for (; allocated < layers_; ++allocated)
      {
        allocator_.allocate({allocated, kOpenPage}, Tier::FP16);
      }
    }
    catch (...)
    {
      while (allocated > 0)
      {
        allocator_.free({--allocated, kOpenPage});
      }
      throw;
    }
    open_pages_ = true;
  }

  void KVPages::reserve(int tokens)
  {
    if (quantised() && !open_pages_ && tokens > 0)
    {
      allocate_open_pages();
    }
    // At a quantised tier a page is taken only once all its positions are
    // coming: until then they are the open page's.
    const int needed = quantised() ? tokens / page_tokens_
                                   : (tokens + page_tokens_ - 1) / page_tokens_;
    for (; pages_ < needed; ++pages_)
    {
      int allocated = 0;
      try
      {
        for (; allocated < layers_; ++allocated)
        {
          allocator_.allocate({allocated, pages_}, tier_);
        }
      }
      catch (...)
      {
        // Give back this position range's pages, newest first, so the cache
        // still holds whole ranges, and the next reserve can try again.
        while (allocated > 0)
        {
          allocator_.free({--allocated, pages_});
        }
        throw;
      }
    }
  }

  const unsigned long long *KVPages::resolve(int layer)
  {
    if (layer < 0 || layer >= layers_)
    {
      throw std::out_of_range("layer " + std::to_string(layer) + " of " +
                              std::to_string(layers_));
    }
    if (resolved_pages_ != pages_ ||
        resolved_generation_ != allocator_.generation_)
    {
      // At a quantised tier the open page's entry follows the tier's pages.
      const int entries = pages_ + (open_pages_ ? 1 : 0);
      if (entries > table_stride_)
      {
        // Room to grow before the next reallocation. Freeing the old buffer
        // waits for launches that may still read it, as cudaFree does.
        table_stride_ =
            entries > 2 * table_stride_ ? entries : 2 * table_stride_;
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
        if (open_pages_)
        {
          host[static_cast<size_t>(l) * table_stride_ + pages_] =
              allocator_.address({l, kOpenPage},
                                 page_bytes_for(page_tokens_, kv_width_));
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

  void KVPages::store_quantised(int layer, const __half *keys,
                                const __half *values, int start, int n)
  {
    const std::size_t fp16_bytes = page_bytes_for(page_tokens_, kv_width_);
    // Addresses are resolved here, between allocator operations, and used at
    // once: nothing below allocates or frees.
    auto *open = reinterpret_cast<__half *>(
        allocator_.address({layer, kOpenPage}, fp16_bytes));
    const std::size_t half = static_cast<std::size_t>(page_tokens_) * kv_width_;
    const auto seal = [&](int span, const __half *k, const __half *v)
    {
      auto *page = reinterpret_cast<std::uint8_t *>(
          allocator_.address({layer, span}, page_bytes_));
      device::quantise_page(k, v, page, tier_, page_tokens_, page_tokens_,
                            kv_heads_, head_dim_);
    };

    const int end = start + n;
    for (int at = start; at < end;)
    {
      const int span = at / page_tokens_;
      const int span_start = span * page_tokens_;
      const int take = std::min(end, span_start + page_tokens_) - at;
      const std::size_t from = static_cast<std::size_t>(at - start) * kv_width_;
      if (take == page_tokens_)
      {
        seal(span, keys + from, values + from);
      }
      else
      {
        const std::size_t row = static_cast<std::size_t>(at - span_start);
        const std::size_t bytes = take * kv_width_ * sizeof(__half);
        cuda_check(cudaMemcpy(open + row * kv_width_, keys + from, bytes,
                              cudaMemcpyDeviceToDevice),
                   "cudaMemcpy keys into the open page");
        cuda_check(cudaMemcpy(open + half + row * kv_width_, values + from,
                              bytes, cudaMemcpyDeviceToDevice),
                   "cudaMemcpy values into the open page");
        if (at + take == span_start + page_tokens_)
        {
          seal(span, open, open + half);
        }
      }
      at += take;
    }
  }

  void KVPages::attention(int layer, const __half *q, const __half *k_bias,
                          const RopeTable *rope, __half *out, int seq_q,
                          int seq_k, int heads, int kv_heads, int head_dim)
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
    if (quantised())
    {
      device::attention_paged_quantised(q, resolve(layer), pages_, page_tokens_,
                                        tier_, k_bias, rope, out, seq_q, seq_k,
                                        heads, kv_heads, head_dim);
      return;
    }
    device::attention_paged(q, resolve(layer), page_tokens_, k_bias, rope, out,
                            seq_q, seq_k, heads, kv_heads, head_dim);
  }

} // namespace microinfer
