#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "microinfer/check.h"
#include "microinfer/device_ops.h"
#include "microinfer/kv_pages.h"
#include "microinfer/launch.h"

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

  KVPages::KVPages(PagedKVCache &allocator, int layers, int page_tokens,
                   std::size_t row, Tier tier)
      : allocator_(allocator), layers_(layers), page_tokens_(page_tokens),
        row_(row), tier_(tier),
        page_bytes_(2 * static_cast<std::size_t>(page_tokens) * row *
                    sizeof(__half))
  {
    if (layers <= 0 || page_tokens <= 0 || row == 0)
    {
      throw std::invalid_argument(
          "layers, page_tokens and row must be positive");
    }
    if (allocator.page_bytes(tier) != page_bytes_)
    {
      throw std::invalid_argument("the allocator's pages at this tier hold " +
                                  std::to_string(allocator.page_bytes(tier)) +
                                  " bytes; " + std::to_string(page_tokens) +
                                  " positions of keys and values at " +
                                  std::to_string(row) + " fp16 each need " +
                                  std::to_string(page_bytes_));
    }
  }

  KVPages::~KVPages()
  {
    // A destructor cannot throw, and a page this cache allocated is always
    // freeable; an error here would mean the allocator itself is broken.
    try
    {
      for (int layer = 0; layer < layers_; ++layer)
      {
        for (int page = pages_ - 1; page >= 0; --page)
        {
          allocator_.free({layer, page});
        }
      }
    }
    catch (...)
    {
    }
  }

  void KVPages::reserve(int tokens)
  {
    const int needed = (tokens + page_tokens_ - 1) / page_tokens_;
    for (; pages_ < needed; ++pages_)
    {
      for (int layer = 0; layer < layers_; ++layer)
      {
        allocator_.allocate({layer, pages_}, tier_);
      }
    }
  }

  const unsigned long long *KVPages::resolve(int layer, int count)
  {
    if (layer < 0 || layer >= layers_)
    {
      throw std::out_of_range("layer " + std::to_string(layer) + " of " +
                              std::to_string(layers_));
    }
    if (count > table_capacity_)
    {
      table_ =
          std::make_unique<DeviceBuffer>(count * sizeof(unsigned long long));
      table_capacity_ = count;
    }
    std::vector<unsigned long long> host(count);
    for (int page = 0; page < count; ++page)
    {
      host[page] = allocator_.address({layer, page}, page_bytes_);
    }
    // Ordered after every launch already enqueued, which may still be reading
    // the table this overwrites.
    cuda_check(cudaMemcpy(table_->raw(), host.data(),
                          count * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy page table host-to-device");
    return table_->as<const unsigned long long>();
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
    const int last_page = (start + n - 1) / page_tokens_;
    device::store_pages(keys, values, resolve(layer, last_page + 1),
                        page_tokens_, row_, start, n);
  }

  void KVPages::attention(int layer, const __half *q, const __half *k_bias,
                          __half *out, int seq_q, int seq_k, int heads,
                          int kv_heads, int head_dim, double theta)
  {
    if (static_cast<std::size_t>(kv_heads) * head_dim != row_)
    {
      throw std::invalid_argument(
          "kv_heads * head_dim is " + std::to_string(kv_heads * head_dim) +
          "; this cache's rows are " + std::to_string(row_));
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
    const int pages = (seq_k + page_tokens_ - 1) / page_tokens_;
    device::attention_paged(q, resolve(layer, pages), page_tokens_, k_bias, out,
                            seq_q, seq_k, heads, kv_heads, head_dim, theta);
  }

} // namespace microinfer
