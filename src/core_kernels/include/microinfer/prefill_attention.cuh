#pragma once

// Prefill's attention on the tensor cores (#213): what attention_kernel
// computes for a launch of more than one query, at the head_dims the models
// have. Device code, included by attention.cu alone.

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

#include "microinfer/attention_partials.cuh"
#include "microinfer/kernels.h"
#include "microinfer/kv_layouts.cuh"
#include "microinfer/tensor_cores.cuh"

namespace microinfer
{
  namespace
  {

    // Query heads of a KV head's group that a block holds at most.
    constexpr int kPrefillHeadsPerBlock = 8;

    // The warps a block has at most, a tile of one head's queries each: what
    // its registers hold, a warp's growing with head_dim.
    constexpr int prefill_warps_per_block(int head_dim)
    {
      return head_dim <= 64 ? 16 : 12;
    }

    constexpr bool prefill_head_dim(int head_dim)
    {
      return head_dim == 64 || head_dim == 128;
    }

    // What P is multiplied by before it is rounded to fp16, and o divided by
    // after: a power of two, which rounds nothing. With it, two fp16 hold a
    // weight to 1.2e-10 of the row's largest, where fp16's subnormals end;
    // and what the first leaves, at most 2^-11 of it, has a product with V
    // that fp16 sums over a tile's 32 keys while |v| stays under 8000.
    constexpr float kWeightScale = 512.0f;

    // A row in shared memory, in elements: eight more than it holds, so that
    // the eight rows a tensor core's load reads 16 bytes of at once lie in
    // eight different runs of banks, and not all in one.
    __host__ __device__ constexpr int prefill_stride(int head_dim)
    {
      return head_dim + 8;
    }

    constexpr size_t prefill_shared_bytes(int head_dim, int warps, bool bias)
    {
      return sizeof(__half) * prefill_stride(head_dim) *
                 (static_cast<size_t>(warps) * kAttentionTileQ +
                  kAttentionTileK * (bias ? 3 : 2)) +
             sizeof(float) * kAttentionTileK;
    }

    // An fp32 number as two fp16 ones, the nearest and what it leaves, which
    // together hold 22 bits of it.
    __device__ __forceinline__ void split_in_two(float x, __half &high,
                                                 __half &low)
    {
      high = __float2half_rn(x);
      low = __float2half_rn(x - __half2float(high));
    }

    // Half `e` of two in a register, widened.
    __device__ __forceinline__ float element(uint32_t two, int e)
    {
      return __half2float(
          __ushort_as_half(static_cast<unsigned short>(two >> (16 * e))));
    }

    // Causal attention with online softmax, as attention_kernel has it, and
    // as its comment describes: the same tiles of kAttentionTileQ queries,
    // the same key tiles, masking, splits and partials, and the same output
    // within ADR-0006's gate. What differs:
    //
    // - **A warp is a query head.** A block holds `heads_per_block` heads of
    //   a KV head's group, a warp each, over the same queries, so that the
    //   key and value tile is loaded, completed and dequantised once for all
    //   of them, into shared memory as fp16.
    // - **The products are the tensor cores'**, fp16 operands accumulated in
    //   fp32. The scores of a tile and its P stay in the warp's registers.
    // - **Where fp16 would not hold an operand, two do.** A completed key
    //   (ADR-0009) and a softmax weight are fp32 numbers, and ADR-0006's gate
    //   rejects either rounded to fp16. Each is multiplied as its nearest
    //   fp16 and what that leaves, two products summed in fp32. A query, a
    //   key with no bias and a value are fp16 already.
    //
    // A row's output depends on no other row, as there.
    template <typename KV, int D, bool kBias>
    __global__ void prefill_attention_kernel(
        const __half *__restrict__ q, KV kv, const __half *__restrict__ k_bias,
        __half *__restrict__ out, int seq_q, int seq_k, int heads, int kv_heads,
        float scale, const float *__restrict__ rope, int lead,
        int heads_per_block, int tiles_per_block, int split_keys, int splits,
        float *__restrict__ partials)
    {
      using namespace tensor_cores;
      constexpr int kRows = kAttentionTileQ;
      constexpr int kKeys = kAttentionTileK;
      constexpr int kScoreColumns = kKeys / 8; // C's of a warp's scores
      constexpr int kHalf = D / 2;
      constexpr int kStride = prefill_stride(D);
      static_assert(kRows == 16, "a warp's C is 16 rows");

      // The tensor cores' loads need 16 bytes' alignment.
      extern __shared__ float4 prefill_smem[];
      // warps x kRows x kStride
      __half *q_s = reinterpret_cast<__half *>(prefill_smem);
      __half *k_s = q_s + blockDim.x / 32 * kRows * kStride; // kKeys x kStride
      __half *v_s = k_s + kKeys * kStride;                   // kKeys x kStride
      // What each completed key leaves of its fp16: with a bias only.
      [[maybe_unused]] __half *k_low_s = v_s + kKeys * kStride;
      // Whether the layout reads each key of the tile: 1 or 0.
      float *reads_s = reinterpret_cast<float *>(
          kBias ? k_low_s + kKeys * kStride : k_low_s);

      const int lane = threadIdx.x % 32;
      const int warp = threadIdx.x / 32;
      const int head_first = blockIdx.y * heads_per_block;
      const int head = head_first + warp % heads_per_block;
      const int split = blockIdx.z;
      // The block's queries, and this warp's tile of them.
      const int block_first = blockIdx.x * tiles_per_block * kRows - lead;
      const int block_last =
          min(block_first + tiles_per_block * kRows, seq_q) - 1;
      const int q_first = block_first + warp / heads_per_block * kRows;
      const int warps = heads_per_block * tiles_per_block;
      const int shift = seq_k - seq_q;
      const KV view = kv.view_for(block_first + shift);
      const size_t token_stride = static_cast<size_t>(heads) * D;
      const int group = heads / kv_heads;
      const int kv_head = head_first / group;
      [[maybe_unused]] const __half *bias =
          kBias ? k_bias + static_cast<size_t>(kv_head) * D : nullptr;
      const auto stored = [&](int query)
      { return query >= 0 && query < seq_q; };

      for (int i = threadIdx.x; i < warps * kRows * kHalf; i += blockDim.x)
      {
        const int d = 2 * (i % kHalf);
        const int row = i / kHalf % kRows;
        const int w = i / kHalf / kRows;
        const int query = block_first + w / heads_per_block * kRows + row;
        *reinterpret_cast<uint32_t *>(q_s + (w * kRows + row) * kStride + d) =
            stored(query) ? two_halves(q + query * token_stride +
                                       static_cast<size_t>(
                                           head_first + w % heads_per_block) *
                                           D +
                                       d)
                          : 0u;
      }

      constexpr int kHalfBytes = sizeof(__half);
      const uint32_t q_at = address_a(q_s + warp * kRows * kStride, kStride);
      const uint32_t k_at = address_b_rows(k_s, kStride);
      const uint32_t v_at = address_b_columns(v_s, kStride);
      [[maybe_unused]] const uint32_t k_low_at =
          address_b_rows(k_low_s, kStride);

      // This thread's part of its warp's rows: two of them, kept as a pair.
      const int rows[2] = {lane / 4, lane / 4 + 8};
      const int column = 2 * (lane % 4); // of the two in a C, the first
      float o[D / 8][4];
      for (int n = 0; n < D / 8; ++n)
      {
        for (int r = 0; r < 4; ++r)
        {
          o[n][r] = 0.0f;
        }
      }
      float m[2] = {-INFINITY, -INFINITY};
      float l[2] = {0.0f, 0.0f};

      const int key_first = split * split_keys;
      const int key_end = min(key_first + split_keys, block_last + shift + 1);

      for (int k_first = key_first; k_first < key_end; k_first += kKeys)
      {
        __syncthreads();
        // A unit of loading is a quad: two adjacent channels of one key and
        // of its value, and the two RoPE pairs them with, whose rotation, of
        // the key's bias, is one.
        for (int i = threadIdx.x; i < kKeys * (D / 4); i += blockDim.x)
        {
          const int c = i / (D / 4);
          const int d = 2 * (i % (D / 4));
          const int j = k_first + c;
          const bool in_range = j < seq_k;
          const Quad quad = in_range ? view.quad(j, kv_head, d, D) : Quad{};
          *reinterpret_cast<uint32_t *>(v_s + c * kStride + d) = quad.value[0];
          *reinterpret_cast<uint32_t *>(v_s + c * kStride + d + kHalf) =
              quad.value[1];
          uint32_t high[2] = {quad.key[0], quad.key[1]};
          if constexpr (kBias)
          {
            __half part[2][2] = {};
            __half left[2][2] = {};
#pragma unroll
            for (int e = 0; in_range && e < 2; ++e)
            {
              const float2 turn = reinterpret_cast<const float2 *>(
                  rope)[static_cast<size_t>(j) * kHalf + d + e];
              const float b1 = __half2float(bias[d + e]);
              const float b2 = __half2float(bias[d + e + kHalf]);
              split_in_two(element(quad.key[0], e) +
                               (b1 * turn.x - b2 * turn.y),
                           part[0][e], left[0][e]);
              split_in_two(element(quad.key[1], e) +
                               (b2 * turn.x + b1 * turn.y),
                           part[1][e], left[1][e]);
            }
#pragma unroll
            for (int side = 0; side < 2; ++side)
            {
              high[side] = pack(part[side][0], part[side][1]);
              *reinterpret_cast<uint32_t *>(k_low_s + c * kStride + d +
                                            side * kHalf) =
                  pack(left[side][0], left[side][1]);
            }
          }
          *reinterpret_cast<uint32_t *>(k_s + c * kStride + d) = high[0];
          *reinterpret_cast<uint32_t *>(k_s + c * kStride + d + kHalf) =
              high[1];
          if (d == 0)
          {
            // A key past seq_k is masked by position already.
            reads_s[c] = !in_range || view.reads(j) ? 1.0f : 0.0f;
          }
        }
        __syncthreads();

        float s[kScoreColumns][4];
        [[maybe_unused]] uint32_t s_low[kScoreColumns][2] = {};
#pragma unroll
        for (int f = 0; f < kScoreColumns; ++f)
        {
#pragma unroll
          for (int r = 0; r < 4; ++r)
          {
            s[f][r] = 0.0f;
          }
        }
#pragma unroll
        for (int d = 0; d < D; d += 16)
        {
          uint32_t a[4];
          load_a(a, q_at + kHalfBytes * d);
#pragma unroll
          for (int f = 0; f < kScoreColumns; ++f)
          {
            uint32_t b[2];
            if constexpr (kBias)
            {
              load_b_rows(b, k_low_at + kHalfBytes * (8 * f * kStride + d));
              multiply_add_fp16(s_low[f], a, b);
            }
            load_b_rows(b, k_at + kHalfBytes * (8 * f * kStride + d));
            multiply_add(s[f], a, b);
          }
        }

        // Scores masked, and each row's maximum over the tile: over this
        // thread's columns, then over the four threads that share its rows.
        float m_tile[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int f = 0; f < kScoreColumns; ++f)
        {
#pragma unroll
          for (int r = 0; r < 4; ++r)
          {
            const int c = 8 * f + column + r % 2;
            const bool visible = k_first + c <= q_first + rows[r / 2] + shift &&
                                 reads_s[c] != 0.0f;
            if constexpr (kBias)
            {
              s[f][r] += element(s_low[f][r / 2], r % 2);
            }
            s[f][r] = visible ? s[f][r] * scale : -INFINITY;
            m_tile[r / 2] = fmaxf(m_tile[r / 2], s[f][r]);
          }
        }
        const auto over_row = [](float x, auto fold)
        {
          x = fold(x, __shfl_xor_sync(0xffffffffu, x, 1));
          return fold(x, __shfl_xor_sync(0xffffffffu, x, 2));
        };
        float rescale[2];
        float from[2];
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
          m_tile[r] =
              over_row(m_tile[r], [](float x, float y) { return fmaxf(x, y); });
          // Exponents are taken from 0 while m is -inf, as attention_kernel
          // has it: every weight is then exactly 0.
          const float m_new = fmaxf(m[r], m_tile[r]);
          from[r] = m_new == -INFINITY ? 0.0f : m_new;
          rescale[r] = expf(m[r] - from[r]);
          m[r] = m_new;
        }

        // P, as the A's of its product with V: two C's of scores each.
        float sum[2] = {0.0f, 0.0f};
        uint32_t p_high[kScoreColumns / 2][4];
        uint32_t p_low[kScoreColumns / 2][4];
#pragma unroll
        for (int f = 0; f < kScoreColumns; ++f)
        {
#pragma unroll
          for (int side = 0; side < 2; ++side)
          {
            __half high[2];
            __half low[2];
#pragma unroll
            for (int e = 0; e < 2; ++e)
            {
              const float p = expf(s[f][2 * side + e] - from[side]);
              sum[side] += p;
              split_in_two(p * kWeightScale, high[e], low[e]);
            }
            p_high[f / 2][f % 2 * 2 + side] = pack(high[0], high[1]);
            p_low[f / 2][f % 2 * 2 + side] = pack(low[0], low[1]);
          }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
          sum[r] = over_row(sum[r], [](float x, float y) { return x + y; });
          l[r] = l[r] * rescale[r] + sum[r];
        }

#pragma unroll
        for (int n = 0; n < D / 8; ++n)
        {
          float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          uint32_t acc_low[2] = {0u, 0u};
#pragma unroll
          for (int f = 0; f < kScoreColumns / 2; ++f)
          {
            uint32_t b[2];
            load_b_columns(b, v_at + kHalfBytes * (16 * f * kStride + 8 * n));
            multiply_add_fp16(acc_low, p_low[f], b);
            multiply_add(acc, p_high[f], b);
          }
#pragma unroll
          for (int r = 0; r < 4; ++r)
          {
            o[n][r] = o[n][r] * rescale[r / 2] +
                      (acc[r] + element(acc_low[r / 2], r % 2)) / kWeightScale;
          }
        }
        // The next tile overwrites k_s and v_s.
      }

      for (int r = 0; r < 4; ++r)
      {
        const int row = rows[r / 2];
        const int query = q_first + row;
        if (!stored(query))
        {
          continue;
        }
        float *slot =
            splits == 1
                ? nullptr
                : partials +
                      ((static_cast<size_t>(split) * seq_q + query) * heads +
                       head) *
                          partial_slot_floats(D);
        for (int n = 0; n < D / 8; ++n)
        {
          const int d = 8 * n + column + r % 2;
          if (splits == 1)
          {
            out[query * token_stride + static_cast<size_t>(head) * D + d] =
                __float2half(o[n][r] / l[r / 2]);
          }
          else
          {
            slot[kPartialOut + d] = o[n][r];
          }
        }
        if (splits > 1 && column == 0 && r % 2 == 0)
        {
          slot[kPartialM] = m[r / 2];
          slot[kPartialL] = l[r / 2];
        }
      }
    }

  } // namespace
} // namespace microinfer
