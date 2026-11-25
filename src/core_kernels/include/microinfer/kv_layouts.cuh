#pragma once

// How an attention kernel reads the keys and values of a cache, whatever it
// computes with them: three layouts of one shape, and the bias that completes
// a key stored without it (ADR-0009). Device code, included by the attention
// kernels alone.

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

#include "microinfer/device_ops.h"
#include "microinfer/packed_halves.cuh"
#include "microinfer/quant.h"

namespace microinfer
{
  namespace
  {

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

    // Four channels of a key and of its value, fp16 as they are read: an even
    // channel and the one after, as one register's two halves, and the two
    // half a head away that RoPE pairs them with. What prefill loads at a
    // time (prefill_attention.cuh): a row's page and tier are found once for
    // all eight elements, and adjacent ones are read together.
    struct Quad
    {
      uint32_t key[2];
      uint32_t value[2];
    };

    // The quad of channels `at`, `at` + 1 and those `half` after them of rows
    // held at fp16.
    __device__ __forceinline__ Quad quad_of_rows(const __half *key_row,
                                                 const __half *value_row,
                                                 size_t at, int half)
    {
      return {{two_halves(key_row + at), two_halves(key_row + at + half)},
              {two_halves(value_row + at), two_halves(value_row + at + half)}};
    }

    // Element `at` of key j's row, widened to fp32, and of its value's row.
    // Three layouts, read by each attention kernel: the arithmetic is shared,
    // so only addressing and dequantisation differ between them. The paged path
    // gives bit-identical output to the contiguous one (#14), and the quantised
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
      // Key j's quad in KV head `kv_head` from channel `d` of it, an even one.
      __device__ Quad quad(int j, int kv_head, int d, int head_dim) const
      {
        return quad_of_rows(k + j * row, v + j * row,
                            static_cast<size_t>(kv_head) * head_dim + d,
                            head_dim / 2);
      }
      // What a query tile starting at `position` reads: the same for every
      // tile here. Only the causal layout below depends on it.
      __device__ ContiguousKV view_for(int) const { return *this; }
      // Every key is read here; only a causal cache's pages may be unread.
      __device__ bool reads(int) const { return true; }
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
      __device__ Quad quad(int j, int kv_head, int d, int head_dim) const
      {
        const __half *keys = key_row(j);
        return quad_of_rows(keys, keys + static_cast<size_t>(page_tokens) * row,
                            static_cast<size_t>(kv_head) * head_dim + d,
                            head_dim / 2);
      }
      __device__ PagedKV view_for(int) const { return *this; }
      __device__ bool reads(int) const { return true; }
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
      // ceil(2^32 / head_dim): a value's KV head, at / head_dim, as one
      // multiply rather than a division by a number known only at run
      // time, which cost INT4 decode a fifth of its time at 32K (#204).
      // Exact while at * head_dim < 2^32 (attention_paged_causal).
      unsigned long long head_dim_reciprocal;

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
            static_cast<size_t>(t) * kv_heads +
                ((static_cast<unsigned long long>(at) * head_dim_reciprocal) >>
                 32));
      }
      // Two adjacent codes, from an even element, dequantised as decode does.
      __device__ __forceinline__ uint32_t two(const std::uint8_t *codes,
                                              size_t e, float scale0,
                                              float zero0, float scale1,
                                              float zero1) const
      {
        unsigned pair;
        if constexpr (Bits == 8)
        {
          pair = *reinterpret_cast<const std::uint16_t *>(codes + e);
        }
        else
        {
          pair = codes[e / kPerByte] >> ((e % kPerByte) * Bits);
        }
        const __half low = __float2half_rn(
            fmaf(static_cast<float>(pair & kMask), scale0, zero0));
        const __half high = __float2half_rn(
            fmaf(static_cast<float>((pair >> Bits) & kMask), scale1, zero1));
        return pack(low, high);
      }
      __device__ Quad quad(const std::uint8_t *page, int t, int kv_head, int d,
                           int head_dim) const
      {
        Quad q;
        const __half *key_scales =
            reinterpret_cast<const __half *>(page + layout.key_scales);
        const __half *key_zeros =
            reinterpret_cast<const __half *>(page + layout.key_zeros);
        const size_t token = static_cast<size_t>(t) * kv_heads + kv_head;
        const float scale = __half2float(reinterpret_cast<const __half *>(
            page + layout.value_scales)[token]);
        const float zero = __half2float(
            reinterpret_cast<const __half *>(page + layout.value_zeros)[token]);
#pragma unroll
        for (int side = 0; side < 2; ++side)
        {
          const size_t at = static_cast<size_t>(kv_head) * head_dim + d +
                            side * (head_dim / 2);
          const __half2 scales =
              *reinterpret_cast<const __half2 *>(key_scales + at);
          const __half2 zeros =
              *reinterpret_cast<const __half2 *>(key_zeros + at);
          q.key[side] = two(page + layout.key_codes, t * row + at,
                            __low2float(scales), __low2float(zeros),
                            __high2float(scales), __high2float(zeros));
          q.value[side] = two(page + layout.value_codes, t * row + at, scale,
                              zero, scale, zero);
        }
        return q;
      }
    };

    // ... or an FP16 page, keys then values, as PagedKV reads it: a page
    // born at FP16 in a cache that seals (ADR-0011, amended by #90), or a
    // diagnostic page of KVPages' Halves::Keys and Halves::Values, whose
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
      __device__ Quad quad(const std::uint8_t *page, int t, int kv_head, int d,
                           int head_dim) const
      {
        const __half *rows = reinterpret_cast<const __half *>(page);
        return quad_of_rows(
            rows + t * row, rows + (static_cast<size_t>(page_tokens) + t) * row,
            static_cast<size_t>(kv_head) * head_dim + d, head_dim / 2);
      }
    };

    // Each sealed page read at its own tier (#91): `tiers` is the layer's
    // table of the tier the page table records for each page, beside its
    // addresses, and a page is read in that tier's format. A tier no page
    // of the cache can be at has no layout, and is never named.
    //
    // The branch costs no divergence within a warp, where a key tile lies in
    // one page. A tile is loaded row by row, and a warp's lanes load
    // consecutive elements of one row, or, by quads at head_dim 64, of two
    // adjacent ones: of one page, at one tier. Different warps may take
    // different branches, which costs nothing.
    struct TierFormats
    {
      const std::uint8_t *tiers;
      Fp16Format fp16;
      QuantisedFormat<8> int8;
      QuantisedFormat<4> int4;
      QuantisedFormat<2> int2;

      // Calls read(format) with page p's format: the one switch on a tier.
      template <typename Read>
      __device__ auto with_format(int p, Read read) const
      {
        switch (static_cast<Tier>(tiers[p]))
        {
        case Tier::INT8:
          return read(int8);
        case Tier::INT4:
          return read(int4);
        case Tier::INT2:
          return read(int2);
        case Tier::FP16:
          break;
        }
        return read(fp16);
      }
      __device__ float key(int p, const std::uint8_t *page, int t,
                           size_t at) const
      {
        return with_format(p, [&](const auto &format)
                           { return format.key(page, t, at); });
      }
      __device__ __half value(int p, const std::uint8_t *page, int t,
                              size_t at) const
      {
        return with_format(p, [&](const auto &format)
                           { return format.value(page, t, at); });
      }
      __device__ Quad quad(int p, const std::uint8_t *page, int t, int kv_head,
                           int d, int head_dim) const
      {
        return with_format(
            p, [&](const auto &format)
            { return format.quad(page, t, kv_head, d, head_dim); });
      }
    };

    // A cache that seals, read causally (ADR-0011): a query reads every page
    // before its own as sealed, each at its own tier, and its own page at
    // FP16, as decode does, whatever else arrived in the same launch. Of its
    // own page, the positions before `start`, the first query's, are in the
    // open page they were written to; the rest are the rows this launch's
    // queries brought, `chunk_k` and `chunk_v`. The kernel's query tiles are
    // aligned so that no tile spans two pages, and view_for binds a tile's
    // page. A sealed page whose tier is kUnreadPage, evicted or masked
    // (#160), is read by no query: reads() says so, and key() and value()
    // give zeros for it without touching its address, which may have been
    // freed.
    struct CausalPagedKV
    {
      const unsigned long long *sealed;
      const __half *open;
      const __half *chunk_k;
      const __half *chunk_v;
      int page_tokens;
      int start;
      size_t row;
      TierFormats formats;
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
      __device__ bool unread(int p) const
      {
        return formats.tiers[p] == device::kUnreadPage;
      }
      // A query's own page and those after it are always read.
      __device__ bool reads(int j) const
      {
        const int p = j / page_tokens;
        return p >= own || !unread(p);
      }
      __device__ float key(int j, size_t at) const
      {
        const int p = j / page_tokens;
        const int t = j % page_tokens;
        if (p < own)
        {
          return unread(p) ? 0.0f : formats.key(p, page(p), t, at);
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
          return unread(p) ? __float2half(0.0f)
                           : formats.value(p, page(p), t, at);
        }
        if (j >= start)
        {
          return chunk_v[static_cast<size_t>(j - start) * row + at];
        }
        return open[(static_cast<size_t>(page_tokens) + t) * row + at];
      }
      __device__ Quad quad(int j, int kv_head, int d, int head_dim) const
      {
        const int p = j / page_tokens;
        const int t = j - p * page_tokens;
        const size_t at = static_cast<size_t>(kv_head) * head_dim + d;
        if (p < own)
        {
          return unread(p) ? Quad{}
                           : formats.quad(p, page(p), t, kv_head, d, head_dim);
        }
        if (j >= start)
        {
          const size_t first = static_cast<size_t>(j - start) * row;
          return quad_of_rows(chunk_k + first, chunk_v + first, at,
                              head_dim / 2);
        }
        return quad_of_rows(open + t * row,
                            open + (static_cast<size_t>(page_tokens) + t) * row,
                            at, head_dim / 2);
      }
    };

  } // namespace
} // namespace microinfer
