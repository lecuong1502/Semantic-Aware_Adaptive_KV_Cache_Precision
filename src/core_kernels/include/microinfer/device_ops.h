#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

#include "microinfer/paged_kv_cache.h"
#include "microinfer/rope_table.h"

namespace microinfer
{

  // The kernels, on device memory: what the engine's forward pass runs.
  //
  // kernels.h is the same arithmetic behind a host-memory surface, for tests
  // (Seam B): each of its functions uploads, calls the launcher declared here,
  // and downloads. So the per-kernel tests exercise exactly the code the engine
  // runs, and nothing here needs a numerics test of its own beyond showing it
  // is that code.
  //
  // Every launcher enqueues on the default stream and returns without
  // synchronising. A forward pass is a long chain of these and a sync after
  // each would serialise the host against every launch. Launch errors are
  // checked at once; execution errors surface at the next synchronising call,
  // which the engine makes when it reads a result back.
  namespace device
  {

    void rmsnorm(const __half *x, const __half *weight, __half *out, int rows,
                 int hidden, float eps);

    // The same RMSNorm reading an fp32 input: the residual stream (ADR-0010).
    void rmsnorm_f32(const float *x, const __half *weight, __half *out,
                     int rows, int hidden, float eps);

    // positions is on the device: one int32 per token.
    void rope(const __half *x, const int32_t *positions, __half *out, int seq,
              int heads, int head_dim, double theta);

    void swiglu(const __half *gate, const __half *up, __half *out,
                size_t count);

    // bias may be null.
    void linear(const __half *x, const __half *weight, const __half *bias,
                __half *out, int rows, int in_features, int out_features);

    // The same projection with an fp32 output: for the LM head, whose logits
    // feed a softmax that the gate compares against an fp32 reference.
    // Rounding 151,936 logits to fp16 would cost up to 2^-11 relative on each
    // before a single comparison was made.
    void linear_fp32_out(const __half *x, const __half *weight, float *out,
                         int rows, int in_features, int out_features);

    // out += x W^T, with out in fp32: a projection added into the fp32
    // residual stream inside the GEMM, never rounded to fp16 (ADR-0010).
    void linear_accumulate(const __half *x, const __half *weight, float *out,
                           int rows, int in_features, int out_features);

    // k_bias, when non-null, is the key projection's bias, (kv_heads,
    // head_dim), which k was stored without (ADR-0009): each key is completed
    // as k + RoPE(k_bias, j) at its position j, with cos and sin read from
    // `rope`, which must cover seq_k positions. Null k_bias means k is whole,
    // and rope is then not read.
    void attention(const __half *q, const __half *k, const __half *v,
                   const __half *k_bias, const RopeTable *rope, __half *out,
                   int seq_q, int seq_k, int heads, int kv_heads, int head_dim);

    // The same attention, reading keys and values through a page table
    // (#14). `pages` is on the device: one page's address per page_tokens
    // positions, for this layer, resolved for this launch alone. A page
    // holds page_tokens rows of keys, (kv_heads, head_dim) each, then as many
    // rows of values. Only addressing differs from attention() above, so the
    // two give bit-identical output on the same keys and values.
    void attention_paged(const __half *q, const unsigned long long *pages,
                         int page_tokens, const __half *k_bias,
                         const RopeTable *rope, __half *out, int seq_q,
                         int seq_k, int heads, int kv_heads, int head_dim);

    // The same attention over pages at a quantised tier (#18). The first
    // `sealed` entries of `pages` are pages at `tier`, in quant.h's layout;
    // the entry after them is the layer's open page, FP16 and laid out as
    // attention_paged's. Each code is dequantised as dequantise_page does it,
    // so the output is attention_paged's over the pages dequantise_page
    // would give, to the bit.
    void attention_paged_quantised(const __half *q,
                                   const unsigned long long *pages, int sealed,
                                   int page_tokens, Tier tier,
                                   const __half *k_bias, const RopeTable *rope,
                                   __half *out, int seq_q, int seq_k, int heads,
                                   int kv_heads, int head_dim);

    // Writes n tokens' key and value rows into their pages, token t at
    // position start + t. `pages` as for attention_paged.
    void store_pages(const __half *keys, const __half *values,
                     const unsigned long long *pages, int page_tokens,
                     size_t row, int start, int n);

    // out[i] = table[ids[i]], a row of `hidden` each. ids is on the device.
    void embed(const int32_t *ids, const __half *table, __half *out, int count,
               int hidden, int vocab);
    // The same gather, widened to fp32: the start of the residual stream.
    void embed_f32(const int32_t *ids, const __half *table, float *out,
                   int count, int hidden, int vocab);

    // out = a + b, in fp32, rounded once. out may alias a or b: the residual
    // stream is updated in place.
    void add(const __half *a, const __half *b, __half *out, size_t count);

    // out[r] = the index of the largest of cols values in row r, the lowest
    // index among ties, as argmax conventionally breaks them.
    void argmax_rows(const float *x, int rows, int cols, int32_t *out);

  } // namespace device

} // namespace microinfer
