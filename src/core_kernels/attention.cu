#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "microinfer/check.h"
#include "microinfer/device_buffer.h"
#include "microinfer/device_ops.h"
#include "microinfer/kernels.h"
#include "microinfer/quant.h"
#include "microinfer/rope_table.h"
#include "microinfer/staging.h"

namespace microinfer
{
  namespace
  {

    constexpr int kBlockThreads = 128;

    // Causal scaled dot-product attention with online softmax, one thread
    // block per (tile of kAttentionTileQ queries, head). The block walks the
    // keys in tiles of kAttentionTileK, keeping for each query row a running
    // maximum m, a running denominator l and an unnormalised output o. When a
    // tile raises the maximum, what has been accumulated so far is rescaled by
    // exp(m_old - m_new) before the tile's own contribution is added. The full
    // (seq_q, seq_k) score matrix never exists; a tile's (kAttentionTileQ,
    // kAttentionTileK) slice of it does, in shared memory, and is overwritten
    // by the next.
    //
    // Causal masking, and why it costs no divergence:
    //
    //   - Whole tiles above the diagonal are never visited. The loop bound is
    //     the last key the block's *last* query can see, which is uniform
    //     across the block.
    //   - Within the tiles that are visited, each score is replaced by -inf
    //     with a select, not a branch. Checked in the sm_89 SASS, it compiles
    //     to `FSEL R, R, -INF, !P0`, so every lane executes the same
    //     instructions whatever its position.
    //     A select also discards what it replaces, so a NaN in a masked key
    //     cannot leak — as multiplying by zero, or adding a large negative,
    //     would let it.
    //
    // No mask tensor exists anywhere: whether key j is visible to query i is
    // the comparison j <= i + (seq_k - seq_q), computed where it is needed.
    //
    // Layout is (tokens, heads, head_dim), row-major, for q and out, and
    // (tokens, kv_heads, head_dim) for k and v.
    //
    // Grouped-query attention is an indexing rule, nothing more: the block for
    // query head h reads KV head h / group, where group = heads / kv_heads.
    // That is HuggingFace's repeat_kv order — contiguous groups, not
    // interleaved — and the KV heads are never copied out to one per query
    // head. group == 1 is ordinary multi-head attention.
    //
    // Each query head's block loads its KV tile for itself, so a group's
    // blocks load the same tile `group` times. Sharing one load across the
    // group is an optimisation for later; it changes no arithmetic.
    //
    // Keys may be stored without their projection's bias (ADR-0009). Then
    // k_bias is non-null, and each key is completed as it is loaded:
    // k = stored + RoPE(k_bias, j), where j, the key's row, is its position.
    // The addition is in fp32, so the stored part, not the bias, sets the
    // precision of a key. A tile's keys are held as fp32 in shared memory for
    // that reason. Without k_bias, widening fp16 to fp32 is exact, and the
    // arithmetic is what it always was.
    __device__ float rotated_bias(const __half *__restrict__ bias, int position,
                                  int d, int head_dim,
                                  const float *__restrict__ rope)
    {
      // Rotate-half pairing: dimension j with j + half. cos and sin come from
      // the table (rope_table.h), formed once per position in fp64.
      const int half = head_dim / 2;
      const int j = d < half ? d : d - half;
      const float *cs = rope + 2 * (static_cast<size_t>(position) * half + j);
      const float b1 = __half2float(bias[j]);
      const float b2 = __half2float(bias[j + half]);
      return d < half ? b1 * cs[0] - b2 * cs[1] : b2 * cs[0] + b1 * cs[1];
    }

    // Element `at` of key j's row, widened to fp32, and of its value's row.
    // Three layouts, one kernel: the arithmetic is shared, so only addressing
    // and dequantisation differ between them. The paged path gives
    // bit-identical output to the contiguous one (#14), and the quantised
    // one gives what either gives over the pages dequantise_page would
    // return (#18).
    //
    // Contiguous: rows (tokens, kv_heads, head_dim), one after another.
    struct ContiguousKV
    {
      const __half *k;
      const __half *v;
      size_t row;

      __device__ float key(int j, size_t at) const
      {
        return __half2float(k[j * row + at]);
      }
      __device__ __half value(int j, size_t at) const
      {
        return v[j * row + at];
      }
      // What a query tile starting at `position` reads: the same for every
      // tile here. Only the causal layout below depends on it.
      __device__ ContiguousKV view_for(int) const { return *this; }
    };

    // Paged (ADR-0004, ADR-0007): key j is in page j / page_tokens at row
    // j % page_tokens. A page holds page_tokens rows of keys, then as many of
    // values. `pages` is the page table for this layer, resolved on the host
    // for this launch alone, so no launch reads an address an allocator
    // operation may since have moved.
    struct PagedKV
    {
      const unsigned long long *pages;
      int page_tokens;
      size_t row;

      __device__ const __half *key_row(int j) const
      {
        return reinterpret_cast<const __half *>(pages[j / page_tokens]) +
               (j % page_tokens) * row;
      }
      __device__ float key(int j, size_t at) const
      {
        return __half2float(key_row(j)[at]);
      }
      __device__ __half value(int j, size_t at) const
      {
        return key_row(j)[static_cast<size_t>(page_tokens) * row + at];
      }
      __device__ PagedKV view_for(int) const { return *this; }
    };

    // How a sealed page's rows are read: a quantised page in quant.h's
    // layout, each code dequantised exactly as dequantise_page does it, one
    // fp32 fused multiply-add rounded once to fp16 ...
    template <int Bits> struct QuantisedFormat
    {
      static constexpr int kPerByte = 8 / Bits;
      static constexpr unsigned kMask = (1u << Bits) - 1;

      QuantisedPageLayout layout;
      size_t row;
      int kv_heads;
      int head_dim;

      __device__ __half decode(const std::uint8_t *page, size_t codes, size_t e,
                               const __half *scales, const __half *zeros,
                               size_t group) const
      {
        const unsigned byte = page[codes + e / kPerByte];
        const float q =
            static_cast<float>((byte >> ((e % kPerByte) * Bits)) & kMask);
        return __float2half_rn(
            fmaf(q, __half2float(scales[group]), __half2float(zeros[group])));
      }
      __device__ float key(const std::uint8_t *page, int t, size_t at) const
      {
        return __half2float(decode(
            page, layout.key_codes, t * row + at,
            reinterpret_cast<const __half *>(page + layout.key_scales),
            reinterpret_cast<const __half *>(page + layout.key_zeros), at));
      }
      __device__ __half value(const std::uint8_t *page, int t, size_t at) const
      {
        return decode(
            page, layout.value_codes, t * row + at,
            reinterpret_cast<const __half *>(page + layout.value_scales),
            reinterpret_cast<const __half *>(page + layout.value_zeros),
            static_cast<size_t>(t) * kv_heads + at / head_dim);
      }
    };

    // ... or an FP16 page, keys then values, as PagedKV reads it: the
    // diagnostic pages of KVPages' Halves::Keys and Halves::Values, whose
    // other half was quantised and dequantised when the page was sealed.
    struct Fp16Format
    {
      size_t row;
      int page_tokens;

      __device__ float key(const std::uint8_t *page, int t, size_t at) const
      {
        return __half2float(
            reinterpret_cast<const __half *>(page)[t * row + at]);
      }
      __device__ __half value(const std::uint8_t *page, int t, size_t at) const
      {
        return reinterpret_cast<const __half *>(
            page)[(static_cast<size_t>(page_tokens) + t) * row + at];
      }
    };

    // A cache at a quantised tier, read causally (ADR-0011): a query reads
    // every page before its own as sealed, and its own page at FP16, as
    // decode does, whatever else arrived in the same launch. Of its own
    // page, the positions before `start`, the first query's, are in the open
    // page they were written to; the rest are the rows this launch's queries
    // brought, `chunk_k` and `chunk_v`. The kernel's query tiles are aligned
    // so that no tile spans two pages, and view_for binds a tile's page.
    template <typename Format> struct CausalPagedKV
    {
      const unsigned long long *sealed;
      const __half *open;
      const __half *chunk_k;
      const __half *chunk_v;
      int page_tokens;
      int start;
      size_t row;
      Format format;
      int own = 0;

      __device__ CausalPagedKV view_for(int position) const
      {
        CausalPagedKV view = *this;
        view.own = position / page_tokens;
        return view;
      }
      __device__ const std::uint8_t *page(int p) const
      {
        return reinterpret_cast<const std::uint8_t *>(sealed[p]);
      }
      __device__ float key(int j, size_t at) const
      {
        const int p = j / page_tokens;
        const int t = j % page_tokens;
        if (p < own)
        {
          return format.key(page(p), t, at);
        }
        if (j >= start)
        {
          return __half2float(
              chunk_k[static_cast<size_t>(j - start) * row + at]);
        }
        return __half2float(open[t * row + at]);
      }
      __device__ __half value(int j, size_t at) const
      {
        const int p = j / page_tokens;
        const int t = j % page_tokens;
        if (p < own)
        {
          return format.value(page(p), t, at);
        }
        if (j >= start)
        {
          return chunk_v[static_cast<size_t>(j - start) * row + at];
        }
        return open[(static_cast<size_t>(page_tokens) + t) * row + at];
      }
    };

    template <typename KV>
    __global__ void attention_kernel(const __half *__restrict__ q, KV kv,
                                     const __half *__restrict__ k_bias,
                                     __half *__restrict__ out, int seq_q,
                                     int seq_k, int heads, int kv_heads,
                                     int head_dim, float scale,
                                     const float *__restrict__ rope, int lead)
    {
      extern __shared__ float smem[];
      float *q_s = smem; // kAttentionTileQ x head_dim
      float *o_s =
          q_s + kAttentionTileQ * head_dim; // kAttentionTileQ x head_dim
      float *s_s =
          o_s + kAttentionTileQ * head_dim; // kAttentionTileQ x kAttentionTileK
      float *m_s = s_s + kAttentionTileQ * kAttentionTileK; // kAttentionTileQ
      float *l_s = m_s + kAttentionTileQ;                   // kAttentionTileQ
      float *rescale_s = l_s + kAttentionTileQ;             // kAttentionTileQ
      float *k_s = rescale_s + kAttentionTileQ; // kAttentionTileK x head_dim
      __half *v_s = reinterpret_cast<__half *>(
          k_s + kAttentionTileK * head_dim); // kAttentionTileK x head_dim

      const int head = blockIdx.y;
      // Tiles start `lead` rows before the first query, so that a layout
      // can align them to absolute positions; rows outside [0, seq_q) are
      // padding, computed on and never stored. Each query's output depends
      // on no other row, so where the tiles fall changes no bit of it.
      const int q_first = blockIdx.x * kAttentionTileQ - lead;
      const int q_last = min(q_first + kAttentionTileQ, seq_q) - 1;
      // Bottom-right alignment: query i is at absolute position i + shift.
      const int shift = seq_k - seq_q;
      const KV view = kv.view_for(q_first + shift);
      const size_t token_stride = static_cast<size_t>(heads) * head_dim;
      const size_t head_offset = static_cast<size_t>(head) * head_dim;
      const int group = heads / kv_heads;
      const size_t kv_head_offset =
          static_cast<size_t>(head / group) * head_dim;

      // The scale is folded into q once, rather than applied to every score.
      for (int i = threadIdx.x; i < kAttentionTileQ * head_dim; i += blockDim.x)
      {
        const int r = i / head_dim;
        const int d = i % head_dim;
        q_s[i] = q_first + r >= 0 && q_first + r <= q_last
                     ? __half2float(
                           q[(q_first + r) * token_stride + head_offset + d]) *
                           scale
                     : 0.0f;
        o_s[i] = 0.0f;
      }
      for (int r = threadIdx.x; r < kAttentionTileQ; r += blockDim.x)
      {
        m_s[r] = -INFINITY;
        l_s[r] = 0.0f;
      }
      __syncthreads();

      // The last key the block's last query can see. Uniform across the block,
      // so every thread runs the same number of iterations.
      const int last_key = q_last + shift;
      const int key_tiles = last_key / kAttentionTileK + 1;

      for (int tile = 0; tile < key_tiles; ++tile)
      {
        const int k_first = tile * kAttentionTileK;

        for (int i = threadIdx.x; i < kAttentionTileK * head_dim;
             i += blockDim.x)
        {
          const int c = i / head_dim;
          const int d = i % head_dim;
          const bool in_range = k_first + c < seq_k;
          const size_t at = kv_head_offset + d;
          float key = in_range ? view.key(k_first + c, at) : 0.0f;
          if (k_bias != nullptr && in_range)
          {
            key += rotated_bias(k_bias + kv_head_offset, k_first + c, d,
                                head_dim, rope);
          }
          k_s[i] = key;
          v_s[i] = in_range ? view.value(k_first + c, at) : __float2half(0.0f);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < kAttentionTileQ * kAttentionTileK;
             i += blockDim.x)
        {
          const int r = i / kAttentionTileK;
          const int c = i % kAttentionTileK;
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d)
          {
            dot += q_s[r * head_dim + d] * k_s[c * head_dim + d];
          }
          // Keys past seq_k are covered too: a valid query's position is at
          // most seq_k - 1, and a padding row's output is never stored.
          const bool visible = k_first + c <= q_first + r + shift;
          s_s[i] = visible ? dot : -INFINITY;
        }
        __syncthreads();

        // Per-row bookkeeping, one thread per query row. The first tile always
        // contains key 0, which every query sees, so m is finite from the first
        // tile onward and m_old - m_new is never -inf - -inf.
        for (int r = threadIdx.x; r < kAttentionTileQ; r += blockDim.x)
        {
          float *row = s_s + r * kAttentionTileK;
          const float m_old = m_s[r];
          float m_new = m_old;
          for (int c = 0; c < kAttentionTileK; ++c)
          {
            m_new = fmaxf(m_new, row[c]);
          }
          float sum = 0.0f;
          for (int c = 0; c < kAttentionTileK; ++c)
          {
            const float p = expf(row[c] - m_new); // exp(-inf) is exactly 0
            row[c] = p;
            sum += p;
          }
          const float rescale = expf(m_old - m_new); // 0 on the first tile
          l_s[r] = l_s[r] * rescale + sum;
          m_s[r] = m_new;
          rescale_s[r] = rescale;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < kAttentionTileQ * head_dim;
             i += blockDim.x)
        {
          const int r = i / head_dim;
          const int d = i % head_dim;
          const float *p = s_s + r * kAttentionTileK;
          float acc = o_s[i] * rescale_s[r];
          for (int c = 0; c < kAttentionTileK; ++c)
          {
            acc += p[c] * __half2float(v_s[c * head_dim + d]);
          }
          o_s[i] = acc;
        }
        // The next tile overwrites k_s, v_s and s_s.
        __syncthreads();
      }

      for (int i = threadIdx.x; i < kAttentionTileQ * head_dim; i += blockDim.x)
      {
        const int r = i / head_dim;
        const int d = i % head_dim;
        if (q_first + r >= 0 && q_first + r <= q_last)
        {
          out[(q_first + r) * token_stride + head_offset + d] =
              __float2half(o_s[i] / l_s[r]);
        }
      }
    }

    size_t shared_bytes(int head_dim)
    {
      const size_t floats =
          2 * static_cast<size_t>(kAttentionTileQ) * head_dim +
          kAttentionTileQ * kAttentionTileK + 3 * kAttentionTileQ +
          static_cast<size_t>(kAttentionTileK) * head_dim;
      const size_t halves = static_cast<size_t>(kAttentionTileK) * head_dim;
      return floats * sizeof(float) + halves * sizeof(__half);
    }

  } // namespace

  namespace
  {

    // Checked in both entry points rather than only at the binding: the kernel
    // divides by heads / kv_heads, so a count that does not group is a division
    // by zero or a silently wrong KV head, whoever the caller is.
    void check_grouping(int heads, int kv_heads)
    {
      if (heads < 0 || kv_heads < 0 || (heads == 0) != (kv_heads == 0))
      {
        throw std::invalid_argument(
            std::to_string(heads) + " query heads and " +
            std::to_string(kv_heads) +
            " KV heads: either both are zero or neither is");
      }
      if (kv_heads > 0 && heads % kv_heads != 0)
      {
        throw std::invalid_argument(
            std::to_string(heads) + " query heads and " +
            std::to_string(kv_heads) +
            " KV heads: the KV head count must divide the query head count, "
            "so that every KV head serves the same number of query heads");
      }
    }

  } // namespace

  namespace
  {

    template <typename KV>
    void launch(const __half *q, KV kv, const __half *k_bias,
                const RopeTable *rope, __half *out, int seq_q, int seq_k,
                int heads, int kv_heads, int head_dim, int lead = 0)
    {
      check_grouping(heads, kv_heads);
      if (seq_q <= 0 || heads == 0 || head_dim <= 0)
      {
        return;
      }
      // A bias is completed from the table, which must reach every key.
      if (k_bias != nullptr &&
          (rope == nullptr || rope->head_dim() != head_dim ||
           rope->positions() < seq_k))
      {
        throw std::invalid_argument(
            "a key bias needs a RoPE table for head_dim " +
            std::to_string(head_dim) + " covering " + std::to_string(seq_k) +
            " positions");
      }

      // Past 48 KiB of dynamic shared memory a kernel must opt in. head_dim
      // 128 stays under it; 256 does not, and head_dim is a parameter. The
      // opt-in is a driver call, so it is made only when a launch needs more
      // than any before it, not once per layer per step; one record per
      // layout, since each is its own kernel. Atomic because the NumPy binding
      // releases the GIL: two threads may both make the call, which is
      // harmless, but not race on the variable.
      const size_t smem = shared_bytes(head_dim);
      static std::atomic<size_t> opted_in{0};
      if (smem > opted_in.load())
      {
        cuda_check(
            cudaFuncSetAttribute(attention_kernel<KV>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem)),
            "cudaFuncSetAttribute attention shared memory");
        opted_in = smem;
      }

      const dim3 grid((lead + seq_q + kAttentionTileQ - 1) / kAttentionTileQ,
                      heads);
      const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
      attention_kernel<KV><<<grid, kBlockThreads, smem>>>(
          q, kv, k_bias, out, seq_q, seq_k, heads, kv_heads, head_dim, scale,
          k_bias != nullptr ? rope->data() : nullptr, lead);
      cuda_check(cudaGetLastError(), "attention kernel launch");
    }

  } // namespace

  void device::attention(const __half *q, const __half *k, const __half *v,
                         const __half *k_bias, const RopeTable *rope,
                         __half *out, int seq_q, int seq_k, int heads,
                         int kv_heads, int head_dim)
  {
    const size_t row = static_cast<size_t>(kv_heads) * head_dim;
    launch(q, ContiguousKV{k, v, row}, k_bias, rope, out, seq_q, seq_k, heads,
           kv_heads, head_dim);
  }

  void device::attention_paged(const __half *q, const unsigned long long *pages,
                               int page_tokens, const __half *k_bias,
                               const RopeTable *rope, __half *out, int seq_q,
                               int seq_k, int heads, int kv_heads, int head_dim)
  {
    if (page_tokens <= 0)
    {
      throw std::invalid_argument("page_tokens must be positive, got " +
                                  std::to_string(page_tokens));
    }
    const size_t row = static_cast<size_t>(kv_heads) * head_dim;
    launch(q, PagedKV{pages, page_tokens, row}, k_bias, rope, out, seq_q, seq_k,
           heads, kv_heads, head_dim);
  }

  void device::attention_paged_causal(
      const __half *q, const unsigned long long *sealed, const __half *open,
      const __half *chunk_k, const __half *chunk_v, int page_tokens, Tier tier,
      bool fp16_pages, const __half *k_bias, const RopeTable *rope, __half *out,
      int seq_q, int seq_k, int heads, int kv_heads, int head_dim)
  {
    if (page_tokens <= 0 || page_tokens % kAttentionTileQ != 0)
    {
      throw std::invalid_argument(
          "causal paged attention aligns query tiles of " +
          std::to_string(kAttentionTileQ) +
          " to pages, so page_tokens must be a positive multiple of it; got " +
          std::to_string(page_tokens));
    }
    const size_t row = static_cast<size_t>(kv_heads) * head_dim;
    const int start = seq_k - seq_q;
    // The first tile begins at the last multiple of the tile size at or
    // before `start`, so every tile lies within one page.
    const int lead = start % kAttentionTileQ;
    if (fp16_pages)
    {
      launch(q,
             CausalPagedKV<Fp16Format>{sealed, open, chunk_k, chunk_v,
                                       page_tokens, start, row,
                                       Fp16Format{row, page_tokens}},
             k_bias, rope, out, seq_q, seq_k, heads, kv_heads, head_dim, lead);
      return;
    }
    const QuantisedPageLayout layout =
        quantised_page_layout(tier, page_tokens, kv_heads, head_dim);
    with_code_width(
        tier,
        [&](auto width)
        {
          constexpr int kBits = decltype(width)::value;
          using Format = QuantisedFormat<kBits>;
          launch(q,
                 CausalPagedKV<Format>{sealed, open, chunk_k, chunk_v,
                                       page_tokens, start, row,
                                       Format{layout, row, kv_heads, head_dim}},
                 k_bias, rope, out, seq_q, seq_k, heads, kv_heads, head_dim,
                 lead);
        });
  }

  void attention(const float *q, const float *k, const float *v,
                 const float *k_bias, float *out, int seq_q, int seq_k,
                 int heads, int kv_heads, int head_dim, double theta)
  {
    check_grouping(heads, kv_heads);
    if (seq_q <= 0 || heads == 0 || head_dim <= 0)
    {
      return; // No elements exist, so there is nothing to write.
    }

    const size_t q_count = static_cast<size_t>(seq_q) * heads * head_dim;
    const size_t kv_count = static_cast<size_t>(seq_k) * kv_heads * head_dim;

    DeviceBuffer dev_q(q_count * sizeof(__half));
    DeviceBuffer dev_k(kv_count * sizeof(__half));
    DeviceBuffer dev_v(kv_count * sizeof(__half));
    DeviceBuffer dev_out(q_count * sizeof(__half));

    upload_fp16(dev_q, q, q_count, "cudaMemcpy q host-to-device");
    upload_fp16(dev_k, k, kv_count, "cudaMemcpy k host-to-device");
    upload_fp16(dev_v, v, kv_count, "cudaMemcpy v host-to-device");
    std::optional<DeviceBuffer> dev_bias;
    std::optional<RopeTable> rope;
    if (k_bias != nullptr)
    {
      rope.emplace(head_dim, theta);
      rope->cover(seq_k);
      const size_t bias_count = static_cast<size_t>(kv_heads) * head_dim;
      dev_bias.emplace(bias_count * sizeof(__half));
      upload_fp16(*dev_bias, k_bias, bias_count,
                  "cudaMemcpy k_bias host-to-device");
    }

    device::attention(dev_q.as<const __half>(), dev_k.as<const __half>(),
                      dev_v.as<const __half>(),
                      dev_bias ? dev_bias->as<const __half>() : nullptr,
                      rope ? &*rope : nullptr, dev_out.as<__half>(), seq_q,
                      seq_k, heads, kv_heads, head_dim);
    cuda_check(cudaDeviceSynchronize(), "attention kernel execution");

    download_fp16(out, dev_out, q_count, "cudaMemcpy output device-to-host");
  }

} // namespace microinfer
