#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <algorithm>

#include "microinfer/attention_partials.cuh"
#include "microinfer/check.h"
#include "microinfer/device_buffer.h"
#include "microinfer/device_ops.h"
#include "microinfer/kernels.h"
#include "microinfer/kv_layouts.cuh"
#include "microinfer/prefill_attention.cuh"
#include "microinfer/quant.h"
#include "microinfer/rope_table.h"
#include "microinfer/staging.h"

namespace microinfer
{
  namespace
  {

    // Threads a block. The kernel waits on memory more than it computes, so
    // more of them in flight is what speeds it: at 32K positions a decode
    // launch took 1.6 ms with 128 and 0.66 ms with 512 (#204). Where a
    // thread works changes no arithmetic.
    constexpr int kBlockThreads = 512;

    // Causal scaled dot-product attention with online softmax, one thread
    // block per tile of rows: kAttentionTileQ queries of one head, or, for a
    // decode step's one query, every head of a GQA group (see
    // attention_kernel). The block walks the
    // keys in tiles of kAttentionTileK, keeping for each query row a running
    // maximum m, a running denominator l and an unnormalised output o. When a
    // tile raises the maximum, what has been accumulated so far is rescaled by
    // exp(m_old - m_new) before the tile's own contribution is added. The full
    // (seq_q, seq_k) score matrix never exists; a tile's (rows,
    // kAttentionTileK) slice of it does, in shared memory, and is overwritten
    // by the next.
    //
    // A launch with too few blocks to fill the device, as a decode step's is
    // at long contexts, also splits each query's keys into runs, in the
    // manner of flash-decoding (#204): a block walks one run and leaves its
    // m, l and o as a partial, and combine_kernel folds the runs, each
    // weighed by exp(m_run - m). It sums in another order than one block
    // would, so a split launch's output differs from an unsplit one's in its
    // last bits; ADR-0006 holds it to float64 within the same gate.
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
    // the comparison j <= i + (seq_k - seq_q), computed where it is needed,
    // and whether the layout reads key j at all, reads(j), true but for a
    // causal cache's unread pages (#160). That is asked once a key, as its
    // tile is loaded, and joins the same select: an unread key's score is
    // -inf as a later one's is, and its bytes are never touched.
    //
    // Layout is (tokens, heads, head_dim), row-major, for q and out, and
    // (tokens, kv_heads, head_dim) for k and v.
    //
    // A decode step may also ask for its per-page attention mass (#98): each
    // page's share of each query head's softmax, for the scorer. It is the
    // same kernel with PageMass compiled in: each of the one query's rows'
    // threads sums each page's exp(s - m) from the scores the bookkeeping
    // below already computes, into shared memory, and once m and l are final
    // every thread of the block makes pages' shares from those sums; in a
    // split launch, shares of the run's own softmax, which combine_kernel
    // brings to the whole one's. Without it the kernel is compiled as it
    // always was, and no bit of `out` differs with it or without.
    //
    // Grouped-query attention is an indexing rule, nothing more: a row for
    // query head h reads KV head h / group, where group = heads / kv_heads.
    // That is HuggingFace's repeat_kv order — contiguous groups, not
    // interleaved — and the KV heads are never copied out to one per query
    // head. group == 1 is ordinary multi-head attention.
    //
    // A decode block holds the group's every head, and loads and dequantises
    // each key tile once for all of them (#204). A launch of more queries, a
    // prefill's, has a block of one head's queries, which loads its KV tile
    // for itself. Neither changes a row's arithmetic. At the models' head_dims
    // such a launch is not this kernel's but prefill_attention_kernel's, on
    // the tensor cores (#213), unless set_prefill_on_tensor_cores said not.
    //
    // Keys may be stored without their projection's bias (ADR-0009). Then
    // k_bias is non-null, and each key is completed as it is loaded:
    // k = stored + RoPE(k_bias, j), where j, the key's row, is its position.
    // The addition is in fp32, so the stored part, not the bias, sets the
    // precision of a key. A tile's keys are held as fp32 in shared memory for
    // that reason. Without k_bias, widening fp16 to fp32 is exact, and the
    // arithmetic is what it always was.

    // A decode step's per-page attention mass (#98), kept by the thread of
    // each query head's row as the key tiles go by. Pages come in order, so
    // one is open at a time: its sum of exp(s - m) and that m, rescaled as m
    // rises. A finished page is left in shared memory, its sum and its m,
    // indexed from `first`, the first page of the block's keys; once the
    // block's m and l are final, its threads make pages' shares from them,
    // sum * exp(m_page - m) / l. A row's thread is the one every other waits
    // for, so it does as little as it can: a division only on a new page,
    // no log, nothing global (#100).
    struct PageMass
    {
      float *partial_sum;
      float *partial_max;
      int page_tokens;
      int first;
      int open = -1;
      int end = 0; // the position after the open page
      float sum = 0.0f;
      float max = -INFINITY;

      __device__ void add(int position, float p, float m)
      {
        if (position >= end)
        {
          close();
          open = position / page_tokens;
          end = (open + 1) * page_tokens;
          sum = 0.0f;
          max = m;
        }
        else if (max < m)
        {
          sum *= expf(max - m);
          max = m;
        }
        sum += p;
      }
      __device__ void close()
      {
        if (open >= 0)
        {
          partial_sum[open - first] = sum;
          partial_max[open - first] = max;
        }
      }
    };

    // Shared memory strides, in elements. A key row and a score row are one
    // element longer than they hold, so that the 32 lanes of a warp, which
    // read 32 different rows at the same column, read 32 different banks
    // rather than one 32 times over. Where an element sits changes no
    // arithmetic.
    __host__ __device__ inline int key_stride(int head_dim)
    {
      return head_dim + 1;
    }
    constexpr int kScoreStride = kAttentionTileK + 1;

    // A tile's rows are (query, head) pairs: queries_per_block queries, each
    // with heads_per_block consecutive query heads of one KV head, row r
    // being query r / heads_per_block and head r % heads_per_block of them.
    // Prefill, off the tensor cores, has 16 queries of one head a row, as the
    // kernel always had; decode has its one query and every head of a GQA
    // group, which then share each key tile the block loads and dequantises
    // (#204). A row's arithmetic is its own whichever rows share its tile.
    //
    // A launch may also split each query's keys into `splits` runs of
    // `split_keys` (blockIdx.z), each block making a partial m, l and
    // unnormalised output over its run alone, for combine_kernel to fold. A
    // launch of one split writes its output itself, as the kernel always did.
    template <typename KV, bool kMass>
    __global__ void attention_kernel(
        const __half *__restrict__ q, KV kv, const __half *__restrict__ k_bias,
        __half *__restrict__ out, int seq_q, int seq_k, int heads, int kv_heads,
        int head_dim, float scale, const float *__restrict__ rope, int lead,
        int heads_per_block, int queries_per_block, int split_keys, int splits,
        float *__restrict__ partials, float *__restrict__ mass,
        int mass_page_tokens, int mass_local_pages)
    {
      const int rows = heads_per_block * queries_per_block;
      const int ks = key_stride(head_dim);
      extern __shared__ float smem[];
      float *q_s = smem;                      // rows x head_dim
      float *o_s = q_s + rows * head_dim;     // rows x head_dim
      float *s_s = o_s + rows * head_dim;     // rows x kScoreStride
      float *m_s = s_s + rows * kScoreStride; // rows
      float *l_s = m_s + rows;                // rows
      float *rescale_s = l_s + rows;          // rows
      // Whether the layout reads each key of the tile: 1 or 0.
      float *reads_s = rescale_s + rows;      // kAttentionTileK
      float *k_s = reads_s + kAttentionTileK; // kAttentionTileK x ks
      __half *v_s = reinterpret_cast<__half *>(
          k_s + kAttentionTileK * ks); // kAttentionTileK x head_dim
      // With the masses only: two floats per head and page, after the rest.
      [[maybe_unused]] float *mass_s =
          reinterpret_cast<float *>(v_s + kAttentionTileK * head_dim);

      const int head_first = blockIdx.y * heads_per_block;
      const int split = blockIdx.z;
      // Tiles start `lead` rows before the first query, so that a layout
      // can align them to absolute positions; rows outside [0, seq_q) are
      // padding, computed on and never stored. Each query's output depends
      // on no other row, so where the tiles fall changes no bit of it.
      const int q_first = blockIdx.x * queries_per_block - lead;
      const int q_last = min(q_first + queries_per_block, seq_q) - 1;
      // Bottom-right alignment: query i is at absolute position i + shift.
      const int shift = seq_k - seq_q;
      const KV view = kv.view_for(q_first + shift);
      const size_t token_stride = static_cast<size_t>(heads) * head_dim;
      const int group = heads / kv_heads;
      const size_t kv_head_offset =
          static_cast<size_t>(head_first / group) * head_dim;
      const auto query_of = [&](int r)
      { return q_first + r / heads_per_block; };
      const auto head_of = [&](int r)
      { return head_first + r % heads_per_block; };
      const auto stored = [&](int r)
      { return query_of(r) >= 0 && query_of(r) <= q_last; };

      // The scale is folded into q once, rather than applied to every score.
      for (int i = threadIdx.x; i < rows * head_dim; i += blockDim.x)
      {
        const int r = i / head_dim;
        const int d = i % head_dim;
        q_s[i] = stored(r)
                     ? __half2float(
                           q[query_of(r) * token_stride +
                             static_cast<size_t>(head_of(r)) * head_dim + d]) *
                           scale
                     : 0.0f;
        o_s[i] = 0.0f;
      }
      for (int r = threadIdx.x; r < rows; r += blockDim.x)
      {
        m_s[r] = -INFINITY;
        l_s[r] = 0.0f;
      }

      // This block's keys: its split's run, up to the last key the block's
      // last query can see. Uniform across the block, so every thread runs
      // the same number of iterations.
      const int key_first = split * split_keys;
      const int key_end = min(key_first + split_keys, q_last + shift + 1);

      // The per-page attention mass, compiled only into the kernel that
      // writes it, so that the one that does not is the kernel it was. Each
      // of the one query's rows keeps its own, in its own thread.
      [[maybe_unused]] PageMass page_mass{};
      [[maybe_unused]] const int mass_row = -q_first * heads_per_block;
      if constexpr (kMass)
      {
        const int r = threadIdx.x - mass_row;
        if (r >= 0 && r < heads_per_block)
        {
          page_mass =
              PageMass{mass_s + r * mass_local_pages,
                       mass_s + (heads_per_block + r) * mass_local_pages,
                       mass_page_tokens, key_first / mass_page_tokens};
        }
      }
      __syncthreads();

      for (int k_first = key_first; k_first < key_end;
           k_first += kAttentionTileK)
      {
        // Two adjacent channels of a row an iteration, so that what they
        // share, the page's tier and address and, below 8 bits, the byte
        // that holds both codes, is read once. head_dim is even.
        for (int i = 2 * threadIdx.x; i < kAttentionTileK * head_dim;
             i += 2 * blockDim.x)
        {
          const int c = i / head_dim;
          const bool in_range = k_first + c < seq_k;
#pragma unroll
          for (int d = i % head_dim; d < i % head_dim + 2; ++d)
          {
            const size_t at = kv_head_offset + d;
            float key = in_range ? view.key(k_first + c, at) : 0.0f;
            if (k_bias != nullptr && in_range)
            {
              key += rotated_bias(k_bias + kv_head_offset, k_first + c, d,
                                  head_dim, rope);
            }
            k_s[c * ks + d] = key;
            v_s[c * head_dim + d] =
                in_range ? view.value(k_first + c, at) : __float2half(0.0f);
          }
          if (i % head_dim == 0)
          {
            // A key past seq_k is masked by position already.
            reads_s[c] = !in_range || view.reads(k_first + c) ? 1.0f : 0.0f;
          }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < rows * kAttentionTileK; i += blockDim.x)
        {
          const int r = i / kAttentionTileK;
          const int c = i % kAttentionTileK;
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d)
          {
            dot += q_s[r * head_dim + d] * k_s[c * ks + d];
          }
          // Keys past seq_k are covered too: a valid query's position is at
          // most seq_k - 1, and a padding row's output is never stored.
          const bool visible =
              k_first + c <= query_of(r) + shift && reads_s[c] != 0.0f;
          s_s[r * kScoreStride + c] = visible ? dot : -INFINITY;
        }
        __syncthreads();

        // Per-row bookkeeping, one thread per row. A row's every score in a
        // tile can be -inf: a tile of unread keys (#160), or a padding
        // row's. m is then -inf still, and exponents are taken from 0
        // instead, so that every p and the rescale are exactly 0, never
        // exp(-inf - -inf). Once m is finite it is used itself, so a row
        // that reads key 0 computes exactly what it always did.
        for (int r = threadIdx.x; r < rows; r += blockDim.x)
        {
          float *row = s_s + r * kScoreStride;
          const float m_old = m_s[r];
          float m_new = m_old;
          for (int c = 0; c < kAttentionTileK; ++c)
          {
            m_new = fmaxf(m_new, row[c]);
          }
          const float m_from = m_new == -INFINITY ? 0.0f : m_new;
          float sum = 0.0f;
          for (int c = 0; c < kAttentionTileK; ++c)
          {
            const float p = expf(row[c] - m_from); // exp(-inf) is exactly 0
            row[c] = p;
            sum += p;
          }
          const float rescale = expf(m_old - m_from); // 0 while m_old is -inf
          l_s[r] = l_s[r] * rescale + sum;
          m_s[r] = m_new;
          rescale_s[r] = rescale;
          if constexpr (kMass)
          {
            if (query_of(r) == 0)
            {
              // When P is a multiple of the tile's size, as the default 32
              // is, every tile lies within one page and adds its sum at
              // once: the keys past seq_k in it are masked, and add
              // nothing. Otherwise a tile may span pages, and adds key by
              // key.
              if (page_mass.page_tokens % kAttentionTileK == 0)
              {
                page_mass.add(k_first, sum, m_new);
              }
              else
              {
                for (int c = 0; c < kAttentionTileK && k_first + c < seq_k; ++c)
                {
                  page_mass.add(k_first + c, row[c], m_new);
                }
              }
            }
          }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < rows * head_dim; i += blockDim.x)
        {
          const int r = i / head_dim;
          const int d = i % head_dim;
          const float *p = s_s + r * kScoreStride;
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

      // A slot of the partials: (split, query, head), each partial_slot_floats.
      const auto slot = [&](int r)
      {
        return partials +
               ((static_cast<size_t>(split) * seq_q + query_of(r)) * heads +
                head_of(r)) *
                   partial_slot_floats(head_dim);
      };
      for (int i = threadIdx.x; i < rows * head_dim; i += blockDim.x)
      {
        const int r = i / head_dim;
        const int d = i % head_dim;
        if (!stored(r))
        {
          continue;
        }
        if (splits == 1)
        {
          out[query_of(r) * token_stride +
              static_cast<size_t>(head_of(r)) * head_dim + d] =
              __float2half(o_s[i] / l_s[r]);
        }
        else
        {
          slot(r)[kPartialOut + d] = o_s[i];
        }
      }
      if (splits > 1)
      {
        for (int r = threadIdx.x; r < rows; r += blockDim.x)
        {
          if (stored(r))
          {
            slot(r)[kPartialM] = m_s[r];
            slot(r)[kPartialL] = l_s[r];
          }
        }
      }
      if constexpr (kMass)
      {
        // The one query's rows, one a head, and the threads that kept them.
        // With one query there is one block of rows (lead is less than a
        // tile), and the bookkeeping gives thread r row r, there being a
        // thread for every row.
        static_assert(kBlockThreads >= kAttentionTileQ,
                      "a thread keeps each row of a tile");
        const int r = threadIdx.x - mass_row;
        if (r >= 0 && r < heads_per_block)
        {
          page_mass.close();
        }
        __syncthreads();
        const int pages = (seq_k + mass_page_tokens - 1) / mass_page_tokens;
        const int first = key_first / mass_page_tokens;
        const int local =
            (min(key_first + split_keys, seq_k) - 1) / mass_page_tokens -
            first + 1;
        for (int i = threadIdx.x; i < heads_per_block * local; i += blockDim.x)
        {
          const int h = i / local;
          const int page = first + i % local;
          const float m = m_s[mass_row + h];
          const float l = l_s[mass_row + h];
          const float sum = mass_s[h * mass_local_pages + i % local];
          const float max =
              mass_s[(heads_per_block + h) * mass_local_pages + i % local];
          float *head_mass = mass + static_cast<size_t>(head_first + h) * pages;
          if (splits == 1)
          {
            head_mass[page] = sum * expf(max - m) / l;
            continue;
          }
          // A split's share, relative to its own m; combine_kernel brings
          // it to the whole softmax's. A split whose every key is unread
          // has none. A page begun in the split before is carried, and
          // that split holds the page's first share.
          const float share = m == -INFINITY ? 0.0f : sum * expf(max - m);
          if (page * mass_page_tokens < key_first)
          {
            slot(mass_row + h)[kPartialCarry] = share;
          }
          else
          {
            head_mass[page] = share;
          }
        }
      }
    }

    // Folds a split launch's partials into each row's output (#204): with M
    // the largest of the splits' m, a split weighs w = exp(m - M), and the
    // output is sum(w o) / sum(w l). With the masses, each page's share
    // relative to its split's m is weighed by its split's w, and by the next
    // split's for the part a page carries into it, then divided by the same
    // sum(w l). Splits are folded in order, one block a (query, head) row.
    __global__ void combine_kernel(const float *__restrict__ partials,
                                   __half *__restrict__ out, int seq_q,
                                   int heads, int head_dim, int splits,
                                   int split_keys, float *__restrict__ mass,
                                   int page_tokens, int seq_k)
    {
      extern __shared__ float weight_s[]; // splits, then the sum of w l
      const int query = blockIdx.x / heads;
      const int head = blockIdx.x % heads;
      const size_t slot_floats = partial_slot_floats(head_dim);
      const auto slot = [&](int s)
      {
        return partials +
               ((static_cast<size_t>(s) * seq_q + query) * heads + head) *
                   slot_floats;
      };
      if (threadIdx.x == 0)
      {
        float m = -INFINITY;
        for (int s = 0; s < splits; ++s)
        {
          m = fmaxf(m, slot(s)[kPartialM]);
        }
        float l = 0.0f;
        for (int s = 0; s < splits; ++s)
        {
          // A split none of whose keys this row reads weighs exactly 0.
          const float w = expf(slot(s)[kPartialM] - m);
          weight_s[s] = w;
          l += slot(s)[kPartialL] * w;
        }
        weight_s[splits] = l;
      }
      __syncthreads();
      const float l = weight_s[splits];
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
      {
        float acc = 0.0f;
        for (int s = 0; s < splits; ++s)
        {
          acc += slot(s)[kPartialOut + d] * weight_s[s];
        }
        out[(static_cast<size_t>(query) * heads + head) * head_dim + d] =
            __float2half(acc / l);
      }
      if (mass == nullptr)
      {
        return;
      }
      const int pages = (seq_k + page_tokens - 1) / page_tokens;
      float *head_mass = mass + static_cast<size_t>(head) * pages;
      for (int page = threadIdx.x; page < pages; page += blockDim.x)
      {
        const int s = page * page_tokens / split_keys;
        float share = head_mass[page] * weight_s[s];
        if (s + 1 < splits && (page + 1) * page_tokens > (s + 1) * split_keys)
        {
          share += slot(s + 1)[kPartialCarry] * weight_s[s + 1];
        }
        head_mass[page] = share / l;
      }
    }

    size_t shared_bytes(int head_dim, int rows)
    {
      const size_t floats =
          2 * static_cast<size_t>(rows) * head_dim + rows * kScoreStride +
          3 * rows + kAttentionTileK +
          static_cast<size_t>(kAttentionTileK) * key_stride(head_dim);
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

    // Where a decode step's per-page attention mass goes, and the pages it
    // is summed over. MassOutput{}, value-initialised, is none.
    struct MassOutput
    {
      float *mass;
      int page_tokens;
    };

    // How a launch lays out its rows and its keys (attention_kernel): not
    // to be confused with a requantisation plan.
    struct LaunchShape
    {
      int heads_per_block;
      int queries_per_block;
      int lead;
      int splits;
      int split_keys;
      // Whether prefill_attention_kernel computes it.
      bool on_tensor_cores;
    };

    // A launch splits its keys until it has about this many blocks for each
    // multiprocessor, and never into runs shorter than kMinSplitKeys: what
    // a split saves is a block's walk over its keys, and what it costs is a
    // partial written and folded once.
    constexpr int kSplitBlocksPerMultiprocessor = 8;
    constexpr int kMinSplitKeys = 256;

    int ceil_div(int a, int b) { return (a + b - 1) / b; }
    int round_up(int a, int b) { return ceil_div(a, b) * b; }

    int split_blocks()
    {
      static const int blocks = []
      {
        int device = 0;
        int count = 0;
        cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        cuda_check(cudaDeviceGetAttribute(
                       &count, cudaDevAttrMultiProcessorCount, device),
                   "cudaDeviceGetAttribute multiprocessor count");
        return kSplitBlocksPerMultiprocessor * count;
      }();
      return blocks;
    }

    // The rows of partials a caller's buffer holds, whatever the launch: a
    // tile for each block a launch is split to, and, for launches of more
    // than one query, eight, so that a prefill chunk of 512 queries of a
    // dozen heads may take three splits, without which 8 of 20
    // multiprocessors have nothing to do for the second half of it (#213). At
    // head_dim 128 there, 10.7 MB, and a decode step's 1.3.
    long partial_rows(bool one_query)
    {
      return static_cast<long>(split_blocks()) * kAttentionTileQ *
             (one_query ? 1 : kPrefillHeadsPerBlock);
    }

    // Whether prefill may run on the tensor cores: it may, unless
    // device::set_prefill_on_tensor_cores said not.
    std::atomic<bool> tensor_cores_allowed{true};

    // The most consecutive heads of a group of `group` that divide it and
    // are at most `most`.
    int heads_of_group(int group, int most)
    {
      int h = std::min(group, most);
      while (group % h != 0)
      {
        --h;
      }
      return h;
    }

    // A decode step's one query takes every head of its KV head's group in
    // one block, or as many of them as divide the group and fit a tile. A
    // launch of more queries takes a tile of them, of one head; on the tensor
    // cores a block has the group's heads too, and two tiles of queries where
    // its warps and, aligned, a page hold them. Keys are split only where
    // partials are given to hold the splits, and only where the launch has
    // too few blocks to fill the device: on the tensor cores, counting blocks
    // of two tiles from the first query, whether or not a page holds both and
    // wherever its tiles begin, so that the layout decides nothing of it. A
    // run of keys is a whole number of key tiles, and at least a page, so
    // that a page spans at most two runs.
    LaunchShape shape_for(int seq_q, int seq_k, int heads, int kv_heads,
                          int head_dim, bool aligned, int page_tokens,
                          bool may_split)
    {
      LaunchShape p{
          1, kAttentionTileQ, 0, 1, round_up(seq_k, kAttentionTileK), false};
      int queries_counted = kAttentionTileQ;
      const int group = heads / kv_heads;
      if (seq_q == 1)
      {
        p.queries_per_block = queries_counted = 1;
        p.heads_per_block = heads_of_group(group, kAttentionTileQ);
      }
      else if (tensor_cores_allowed &&
               with_prefill_head_dim(head_dim, [](auto) {}))
      {
        p.on_tensor_cores = true;
        p.heads_per_block = heads_of_group(group, kPrefillHeadsPerBlock);
        constexpr int kTwoTiles = 2 * kAttentionTileQ;
        if (seq_q > kAttentionTileQ &&
            2 * p.heads_per_block <= prefill_warps_per_block(head_dim))
        {
          queries_counted = kTwoTiles;
          if (!aligned || page_tokens % kTwoTiles == 0)
          {
            p.queries_per_block = kTwoTiles;
          }
        }
      }
      p.lead = aligned ? (seq_k - seq_q) % p.queries_per_block : 0;
      if (!may_split)
      {
        return p;
      }
      const long blocks =
          static_cast<long>(ceil_div((p.on_tensor_cores ? 0 : p.lead) + seq_q,
                                     queries_counted)) *
          (heads / p.heads_per_block);
      // No more splits than the partials hold rows of.
      const int splits = static_cast<int>(std::min<long>(
          {split_blocks() / blocks, seq_k / kMinSplitKeys,
           partial_rows(seq_q == 1) / (static_cast<long>(seq_q) * heads)}));
      if (splits <= 1)
      {
        return p;
      }
      p.split_keys =
          std::max(round_up(ceil_div(seq_k, splits), kAttentionTileK),
                   round_up(page_tokens, kAttentionTileK));
      p.splits = ceil_div(seq_k, p.split_keys);
      if (p.splits == 1)
      {
        p.split_keys = round_up(seq_k, kAttentionTileK);
      }
      return p;
    }

    template <typename KV>
    void launch(const __half *q, KV kv, const __half *k_bias,
                const RopeTable *rope, __half *out, int seq_q, int seq_k,
                int heads, int kv_heads, int head_dim, bool aligned,
                int page_tokens, MassOutput mass, float *partials)
    {
      check_grouping(heads, kv_heads);
      if (mass.mass != nullptr && seq_q != 1)
      {
        throw std::invalid_argument(
            "the per-page attention mass is a decode step's, of one query; "
            "got " +
            std::to_string(seq_q) + " queries");
      }
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

      // A block loads two adjacent channels of a row at a time.
      if (head_dim % 2 != 0)
      {
        throw std::invalid_argument("attention loads channels in pairs, so "
                                    "head_dim must be even; got " +
                                    std::to_string(head_dim));
      }
      const LaunchShape p =
          shape_for(seq_q, seq_k, heads, kv_heads, head_dim, aligned,
                    page_tokens, partials != nullptr);
      // The masses take two floats per head and page of the block's keys
      // in shared memory besides: all of them with one split.
      const int mass_local_pages =
          mass.mass != nullptr ? std::min(ceil_div(seq_k, mass.page_tokens),
                                          p.split_keys / mass.page_tokens + 2)
                               : 0;
      // Past 48 KiB of dynamic shared memory a kernel must opt in. head_dim
      // 128 stays under it; 256 does not, and head_dim is a parameter. The
      // opt-in is a driver call, so it is made only when a launch needs more
      // than any before it, not once per layer per step; one record per
      // layout, since each is its own kernel. Atomic because the NumPy binding
      // releases the GIL: two threads may both make the call, which is
      // harmless, but not race on the variable.
      const size_t smem =
          p.on_tensor_cores
              ? prefill_shared_bytes(head_dim,
                                     p.heads_per_block * p.queries_per_block /
                                         kAttentionTileQ,
                                     k_bias != nullptr)
              : shared_bytes(head_dim,
                             p.heads_per_block * p.queries_per_block) +
                    2 * static_cast<size_t>(p.heads_per_block) *
                        mass_local_pages * sizeof(float);
      const dim3 grid(ceil_div(p.lead + seq_q, p.queries_per_block),
                      heads / p.heads_per_block, p.splits);
      const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
      const auto allow = [&](auto kernel, std::atomic<size_t> &opted_in)
      {
        if (smem > opted_in.load())
        {
          cuda_check(cudaFuncSetAttribute(
                         kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem)),
                     "cudaFuncSetAttribute attention shared memory");
          opted_in = smem;
        }
      };
      const float *table = k_bias != nullptr ? rope->data() : nullptr;
      const auto run = [&](auto kernel, std::atomic<size_t> &opted_in)
      {
        allow(kernel, opted_in);
        kernel<<<grid, kBlockThreads, smem>>>(
            q, kv, k_bias, out, seq_q, seq_k, heads, kv_heads, head_dim, scale,
            table, p.lead, p.heads_per_block, p.queries_per_block, p.split_keys,
            p.splits, partials, mass.mass, mass.page_tokens, mass_local_pages);
      };
      static std::atomic<size_t> opted_in{0};
      static std::atomic<size_t> opted_in_with_mass{0};
      const auto run_prefill = [&](auto kernel, std::atomic<size_t> &opted_in)
      {
        allow(kernel, opted_in);
        const int tiles = p.queries_per_block / kAttentionTileQ;
        kernel<<<grid, 32 * p.heads_per_block * tiles, smem>>>(
            q, kv, k_bias, out, seq_q, seq_k, heads, kv_heads, scale, table,
            p.lead, p.heads_per_block, tiles, p.split_keys, p.splits, partials);
      };
      const auto run_prefill_at = [&](auto dim)
      {
        constexpr int kD = decltype(dim)::value;
        static std::atomic<size_t> opted_in_with_bias{0};
        static std::atomic<size_t> opted_in_without{0};
        if (k_bias != nullptr)
        {
          run_prefill(prefill_attention_kernel<KV, kD, true>,
                      opted_in_with_bias);
        }
        else
        {
          run_prefill(prefill_attention_kernel<KV, kD, false>,
                      opted_in_without);
        }
      };
      if (p.on_tensor_cores)
      {
        with_prefill_head_dim(head_dim, run_prefill_at);
      }
      else if (mass.mass != nullptr)
      {
        run(attention_kernel<KV, true>, opted_in_with_mass);
      }
      else
      {
        run(attention_kernel<KV, false>, opted_in);
      }
      cuda_check(cudaGetLastError(), "attention kernel launch");
      if (p.splits > 1)
      {
        combine_kernel<<<seq_q * heads, kBlockThreads,
                         (p.splits + 1) * sizeof(float)>>>(
            partials, out, seq_q, heads, head_dim, p.splits, p.split_keys,
            mass.mass, mass.page_tokens, seq_k);
        cuda_check(cudaGetLastError(), "attention combine kernel launch");
      }
    }

  } // namespace

  size_t device::attention_partials_floats(int head_dim, bool one_query)
  {
    return static_cast<size_t>(partial_rows(one_query)) *
           partial_slot_floats(head_dim);
  }

  bool device::set_prefill_on_tensor_cores(bool allowed)
  {
    return tensor_cores_allowed.exchange(allowed);
  }

  std::pair<int, int> device::attention_split(int seq_q, int seq_k, int heads,
                                              int kv_heads, int head_dim,
                                              int page_tokens)
  {
    check_grouping(heads, kv_heads);
    const LaunchShape p = shape_for(seq_q, seq_k, heads, kv_heads, head_dim,
                                    page_tokens > 0, page_tokens, true);
    return {p.splits, p.split_keys};
  }

  void device::attention(const __half *q, const __half *k, const __half *v,
                         const __half *k_bias, const RopeTable *rope,
                         __half *out, int seq_q, int seq_k, int heads,
                         int kv_heads, int head_dim, float *partials)
  {
    const size_t row = static_cast<size_t>(kv_heads) * head_dim;
    launch(q, ContiguousKV{k, v, row}, k_bias, rope, out, seq_q, seq_k, heads,
           kv_heads, head_dim, false, 0, MassOutput{}, partials);
  }

  void device::attention_paged(const __half *q, const unsigned long long *pages,
                               int page_tokens, const __half *k_bias,
                               const RopeTable *rope, __half *out, int seq_q,
                               int seq_k, int heads, int kv_heads, int head_dim,
                               float *mass, float *partials)
  {
    if (page_tokens <= 0)
    {
      throw std::invalid_argument("page_tokens must be positive, got " +
                                  std::to_string(page_tokens));
    }
    const size_t row = static_cast<size_t>(kv_heads) * head_dim;
    launch(q, PagedKV{pages, page_tokens, row}, k_bias, rope, out, seq_q, seq_k,
           heads, kv_heads, head_dim, false, page_tokens,
           MassOutput{mass, page_tokens}, partials);
  }

  void device::attention_paged_causal(
      const __half *q, const unsigned long long *sealed,
      const std::uint8_t *tiers, const std::array<bool, kTierCount> &may_be_at,
      const __half *open, const __half *chunk_k, const __half *chunk_v,
      int page_tokens, const __half *k_bias, const RopeTable *rope, __half *out,
      int seq_q, int seq_k, int heads, int kv_heads, int head_dim, float *mass,
      float *partials)
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
    // A layout for each quantised tier a page can be at, and none for the
    // others, whose shape the layout may not hold.
    const auto layout = [&](Tier tier)
    {
      return may_be_at[static_cast<int>(tier)]
                 ? quantised_page_layout(tier, page_tokens, kv_heads, head_dim)
                 : QuantisedPageLayout{};
    };
    // floor(at * ceil(2^32 / d) / 2^32) is floor(at / d) while at * d < 2^32:
    // the reciprocal's excess, under 1 / 2^32 a unit of `at`, then stays
    // below the 1 / d by which at / d falls short of the next integer.
    // `at` is below row = kv_heads * head_dim.
    if (static_cast<unsigned long long>(row) * head_dim >= (1ull << 32))
    {
      throw std::invalid_argument(
          "a row of " + std::to_string(row) + " elements at head_dim " +
          std::to_string(head_dim) + " is too wide for the KV head's division");
    }
    const unsigned long long reciprocal =
        ((1ull << 32) + head_dim - 1) /
        static_cast<unsigned long long>(head_dim);
    const TierFormats formats{
        tiers, Fp16Format{row, page_tokens},
        QuantisedFormat<8>{layout(Tier::INT8), row, kv_heads, reciprocal},
        QuantisedFormat<4>{layout(Tier::INT4), row, kv_heads, reciprocal},
        QuantisedFormat<2>{layout(Tier::INT2), row, kv_heads, reciprocal}};
    // The first tile begins at the last multiple of its queries at or before
    // `start`, so every tile lies within one page: `aligned`.
    launch(q,
           CausalPagedKV{sealed, open, chunk_k, chunk_v, page_tokens, start,
                         row, formats},
           k_bias, rope, out, seq_q, seq_k, heads, kv_heads, head_dim, true,
           page_tokens, MassOutput{mass, page_tokens}, partials);
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
    // This surface splits a long launch's keys as the engine does, into
    // partials of its own.
    DeviceBuffer dev_partials(
        device::attention_partials_floats(head_dim, seq_q == 1) *
        sizeof(float));
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
                      seq_k, heads, kv_heads, head_dim,
                      dev_partials.as<float>());
    cuda_check(cudaDeviceSynchronize(), "attention kernel execution");

    download_fp16(out, dev_out, q_count, "cudaMemcpy output device-to-host");
  }

} // namespace microinfer
