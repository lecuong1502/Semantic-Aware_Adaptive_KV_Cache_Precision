#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

#include "microinfer/check.h"
#include "microinfer/device_buffer.h"
#include "microinfer/kernels.h"
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
    // Layout is (tokens, heads, head_dim), row-major, for q, k, v and out.
    __global__ void
    attention_kernel(const __half *__restrict__ q, const __half *__restrict__ k,
                     const __half *__restrict__ v, __half *__restrict__ out,
                     int seq_q, int seq_k, int heads, int head_dim, float scale)
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
      __half *k_s = reinterpret_cast<__half *>(rescale_s + kAttentionTileQ);
      __half *v_s =
          k_s + kAttentionTileK * head_dim; // kAttentionTileK x head_dim

      const int head = blockIdx.y;
      const int q_first = blockIdx.x * kAttentionTileQ;
      const int rows = min(kAttentionTileQ, seq_q - q_first);
      // Bottom-right alignment: query i is at absolute position i + shift.
      const int shift = seq_k - seq_q;
      const size_t token_stride = static_cast<size_t>(heads) * head_dim;
      const size_t head_offset = static_cast<size_t>(head) * head_dim;

      // The scale is folded into q once, rather than applied to every score.
      for (int i = threadIdx.x; i < kAttentionTileQ * head_dim; i += blockDim.x)
      {
        const int r = i / head_dim;
        const int d = i % head_dim;
        q_s[i] = r < rows
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
      const int last_key = q_first + rows - 1 + shift;
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
          const size_t at = (k_first + c) * token_stride + head_offset + d;
          k_s[i] = in_range ? k[at] : __float2half(0.0f);
          v_s[i] = in_range ? v[at] : __float2half(0.0f);
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
            dot += q_s[r * head_dim + d] * __half2float(k_s[c * head_dim + d]);
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

      for (int i = threadIdx.x; i < rows * head_dim; i += blockDim.x)
      {
        const int r = i / head_dim;
        const int d = i % head_dim;
        out[(q_first + r) * token_stride + head_offset + d] =
            __float2half(o_s[i] / l_s[r]);
      }
    }

    size_t shared_bytes(int head_dim)
    {
      const size_t floats =
          2 * static_cast<size_t>(kAttentionTileQ) * head_dim +
          kAttentionTileQ * kAttentionTileK + 3 * kAttentionTileQ;
      const size_t halves = 2 * static_cast<size_t>(kAttentionTileK) * head_dim;
      return floats * sizeof(float) + halves * sizeof(__half);
    }

  } // namespace

  void attention(const float *q, const float *k, const float *v, float *out,
                 int seq_q, int seq_k, int heads, int head_dim)
  {
    if (seq_q <= 0 || heads <= 0 || head_dim <= 0)
    {
      return; // No elements exist, so there is nothing to write.
    }

    const size_t q_count = static_cast<size_t>(seq_q) * heads * head_dim;
    const size_t kv_count = static_cast<size_t>(seq_k) * heads * head_dim;

    DeviceBuffer dev_q(q_count * sizeof(__half));
    DeviceBuffer dev_k(kv_count * sizeof(__half));
    DeviceBuffer dev_v(kv_count * sizeof(__half));
    DeviceBuffer dev_out(q_count * sizeof(__half));

    upload_fp16(dev_q, q, q_count, "cudaMemcpy q host-to-device");
    upload_fp16(dev_k, k, kv_count, "cudaMemcpy k host-to-device");
    upload_fp16(dev_v, v, kv_count, "cudaMemcpy v host-to-device");

    // Past 48 KiB of dynamic shared memory a kernel must opt in. head_dim 128
    // stays under it; 256 does not, and head_dim is a parameter.
    const size_t smem = shared_bytes(head_dim);
    cuda_check(cudaFuncSetAttribute(attention_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(smem)),
               "cudaFuncSetAttribute attention shared memory");

    const dim3 grid((seq_q + kAttentionTileQ - 1) / kAttentionTileQ, heads);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    attention_kernel<<<grid, kBlockThreads, smem>>>(
        dev_q.as<const __half>(), dev_k.as<const __half>(),
        dev_v.as<const __half>(), dev_out.as<__half>(), seq_q, seq_k, heads,
        head_dim, scale);
    cuda_check(cudaGetLastError(), "attention kernel launch");
    cuda_check(cudaDeviceSynchronize(), "attention kernel execution");

    download_fp16(out, dev_out, q_count, "cudaMemcpy output device-to-host");
  }

} // namespace microinfer
