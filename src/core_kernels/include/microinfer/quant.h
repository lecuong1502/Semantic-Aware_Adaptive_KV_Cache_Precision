#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "microinfer/paged_kv_cache.h"

namespace microinfer
{

  // A quantised page (ADR-0005, #16): one layer's keys and values for P
  // positions at INT8, INT4 or INT2. Every quantised tier shares this layout;
  // only the width of a code differs.
  //
  // Keys are quantised per (head, channel), across the page's P positions:
  // KIVI found key outliers along channels. Values are quantised per (head,
  // token), across head_dim channels. Both are asymmetric: a group's
  // zero-point is its minimum, which a code of 0 stands for, and its scale is
  // its range over the 2^bits - 1 steps above that. A value comes back as
  // code * scale + zero-point. A group whose range is zero has scale 0, every
  // code 0, and comes back exactly.
  //
  // The regions, back to back from byte 0, W = kv_heads * head_dim:
  //
  //   key codes      P rows of W codes     row t is token t, element h*D + d
  //   value codes    P rows of W codes     the same
  //   key scales     W fp16                element h*D + d
  //   key zeros      W fp16                element h*D + d
  //   value scales   P * kv_heads fp16     element t*kv_heads + h
  //   value zeros    P * kv_heads fp16     element t*kv_heads + h
  //
  // The rows are the FP16 page's rows (kv_pages.h), codes in place of
  // halves. Below 8 bits, 8/bits consecutive codes share a byte, the first in
  // the lowest bits. head_dim * bits must be a multiple of 8, so a value
  // group never shares a byte with the next.
  //
  // The metadata is 4 * (W + P * kv_heads) bytes at every tier: 1280 on
  // Qwen2.5-1.5B (2 KV heads of 128). A compression ratio that leaves it out
  // is false (ADR-0005), so the layout reports the effective bits it gives.
  struct QuantisedPageLayout
  {
    int bits;
    std::size_t key_codes;
    std::size_t value_codes;
    std::size_t key_scales;
    std::size_t key_zeros;
    std::size_t value_scales;
    std::size_t value_zeros;
    std::size_t metadata_bytes;
    std::size_t page_bytes;
    // Bits per cached element with the metadata counted: 8 * page_bytes over
    // the 2 * P * W elements a page holds.
    double effective_bits;
  };

  // The one place the quantised layout's arithmetic is written down.
  // Throws std::invalid_argument for FP16, which is not quantised, and for a
  // shape the layout cannot hold.
  QuantisedPageLayout quantised_page_layout(Tier tier, int page_tokens,
                                            int kv_heads, int head_dim);

  // Asked to quantise a page that is still being filled. A key channel's
  // scale spans all P positions of the page, so until the last is written
  // there is none to compute: the open page stays at FP16 (ADR-0005).
  class OpenPage : public std::invalid_argument
  {
  public:
    OpenPage(int filled, int page_tokens);
  };

  namespace device
  {

    // Quantises one page: keys and values are each (page_tokens, kv_heads,
    // head_dim) fp16, the FP16 page's two halves, and `page` receives
    // quantised_page_layout(...).page_bytes. `filled` is how many of the
    // page's positions have been written; fewer than page_tokens throws
    // OpenPage before anything is launched. Enqueues on the default stream,
    // as the other device launchers do (device_ops.h).
    void quantise_page(const __half *keys, const __half *values,
                       std::uint8_t *page, Tier tier, int page_tokens,
                       int filled, int kv_heads, int head_dim);

    // The inverse: an FP16 page's keys and values, each (page_tokens,
    // kv_heads, head_dim), computed as code * scale + zero-point in one fp32
    // fused multiply-add and rounded once to fp16.
    void dequantise_page(const std::uint8_t *page, __half *keys, __half *values,
                         Tier tier, int page_tokens, int kv_heads,
                         int head_dim);

  } // namespace device

  // The same, on host memory, for Seam B: fp32 keys and values are rounded
  // to fp16 on the way up, as the cache holds them, and come back as fp32.
  // `page` is host memory of the layout's page_bytes.
  void quantise_page(const float *keys, const float *values, std::uint8_t *page,
                     Tier tier, int page_tokens, int filled, int kv_heads,
                     int head_dim);
  void dequantise_page(const std::uint8_t *page, float *keys, float *values,
                       Tier tier, int page_tokens, int kv_heads, int head_dim);

} // namespace microinfer
