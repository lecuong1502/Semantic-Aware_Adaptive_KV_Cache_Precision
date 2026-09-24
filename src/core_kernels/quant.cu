#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "microinfer/check.h"
#include "microinfer/device_buffer.h"
#include "microinfer/launch.h"
#include "microinfer/quant.h"
#include "microinfer/staging.h"

namespace microinfer
{
  namespace
  {

    constexpr int kBlockThreads = 128;
    constexpr int kWarp = 32;
    // The values kernel gives each warp one group and lets it return whole.
    static_assert(kBlockThreads % kWarp == 0, "blocks must be whole warps");

    int bits_of(Tier tier)
    {
      switch (tier)
      {
      case Tier::INT8:
        return 8;
      case Tier::INT4:
        return 4;
      case Tier::INT2:
        return 2;
      case Tier::FP16:
        break;
      }
      throw std::invalid_argument(
          "FP16 is not a quantised tier: its page is kv_pages.h's, with no "
          "codes and no metadata");
    }

    void require_positive(int n, const char *what)
    {
      if (n <= 0)
      {
        throw std::invalid_argument(
            std::string(what) + " must be positive, got " + std::to_string(n));
      }
    }

    // The group's scale: its range over the steps above its minimum, formed in
    // fp64 and rounded *up* to the fp16 the page stores, so that the last
    // level always reaches the group's maximum and no code is ever clamped.
    // Rounded to nearest, a scale below fp16's smallest normal, where the
    // spacing is coarse, could fall short by an eighth, and the clamp would
    // cost the top of the range several steps. Codes are then formed against
    // the stored scale, so that dequantising with it is consistent with how
    // they were chosen.
    template <int Bits> __device__ __half scale_of(float lo, float hi)
    {
      constexpr int kLevels = (1 << Bits) - 1;
      const double exact = (static_cast<double>(hi) - lo) / kLevels;
      const __half nearest = __double2half(exact);
      if (static_cast<double>(__half2float(nearest)) >= exact)
      {
        return nearest;
      }
      // Non-negative, so the next fp16 up is the next bit pattern.
      return __ushort_as_half(
          static_cast<unsigned short>(__half_as_ushort(nearest) + 1));
    }

    // Rounds at most twice, at x - zero and at the quotient; either can move
    // a code only where its exact value sits at a rounding tie.
    template <int Bits>
    __device__ unsigned code_of(float x, float zero, float scale)
    {
      constexpr float kLevels = (1 << Bits) - 1;
      if (scale == 0.0f)
      {
        return 0;
      }
      const float q = rintf((x - zero) / scale);
      return static_cast<unsigned>(fminf(fmaxf(q, 0.0f), kLevels));
    }

    // Keys, per (head, channel) across the page's positions. One thread owns
    // the 8/Bits consecutive channels that share a byte of every row, so it
    // writes whole bytes; the zero-point, a minimum of fp16 values, is
    // itself fp16 and stored exactly.
    template <int Bits>
    __global__ void quantise_keys_kernel(const __half *__restrict__ keys,
                                         std::uint8_t *__restrict__ codes,
                                         __half *__restrict__ scales,
                                         __half *__restrict__ zeros,
                                         int page_tokens, int width)
    {
      constexpr int kPerByte = 8 / Bits;
      const int column = blockIdx.x * blockDim.x + threadIdx.x;
      const int columns = width / kPerByte;
      if (column >= columns)
      {
        return;
      }
      const int first = column * kPerByte;
      const auto row = [width](int t)
      { return static_cast<size_t>(t) * width; };

      float lo[kPerByte], hi[kPerByte], scale[kPerByte];
      for (int c = 0; c < kPerByte; ++c)
      {
        lo[c] = INFINITY;
        hi[c] = -INFINITY;
      }
      for (int t = 0; t < page_tokens; ++t)
      {
        for (int c = 0; c < kPerByte; ++c)
        {
          const float x = __half2float(keys[row(t) + first + c]);
          lo[c] = fminf(lo[c], x);
          hi[c] = fmaxf(hi[c], x);
        }
      }
      for (int c = 0; c < kPerByte; ++c)
      {
        const __half s = scale_of<Bits>(lo[c], hi[c]);
        scales[first + c] = s;
        zeros[first + c] = __float2half(lo[c]);
        scale[c] = __half2float(s);
      }
      for (int t = 0; t < page_tokens; ++t)
      {
        unsigned byte = 0;
        for (int c = 0; c < kPerByte; ++c)
        {
          const float x = __half2float(keys[row(t) + first + c]);
          byte |= code_of<Bits>(x, lo[c], scale[c]) << (c * Bits);
        }
        codes[static_cast<size_t>(t) * columns + column] =
            static_cast<std::uint8_t>(byte);
      }
    }

    // Values, per (head, token) across head_dim channels. One warp per group:
    // the lanes reduce its minimum and maximum, then write its bytes.
    template <int Bits>
    __global__ void quantise_values_kernel(const __half *__restrict__ values,
                                           std::uint8_t *__restrict__ codes,
                                           __half *__restrict__ scales,
                                           __half *__restrict__ zeros,
                                           int groups, int head_dim)
    {
      constexpr int kPerByte = 8 / Bits;
      const int group = (blockIdx.x * blockDim.x + threadIdx.x) / kWarp;
      const int lane = threadIdx.x % kWarp;
      if (group >= groups)
      {
        return;
      }
      const __half *x = values + static_cast<size_t>(group) * head_dim;

      float lo = INFINITY, hi = -INFINITY;
      for (int i = lane; i < head_dim; i += kWarp)
      {
        const float v = __half2float(x[i]);
        lo = fminf(lo, v);
        hi = fmaxf(hi, v);
      }
      for (int offset = kWarp / 2; offset > 0; offset /= 2)
      {
        lo = fminf(lo, __shfl_xor_sync(0xffffffffu, lo, offset));
        hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, offset));
      }
      const __half s = scale_of<Bits>(lo, hi);
      if (lane == 0)
      {
        scales[group] = s;
        zeros[group] = __float2half(lo);
      }
      const float scale = __half2float(s);

      const int bytes = head_dim / kPerByte;
      std::uint8_t *out = codes + static_cast<size_t>(group) * bytes;
      for (int b = lane; b < bytes; b += kWarp)
      {
        unsigned byte = 0;
        for (int c = 0; c < kPerByte; ++c)
        {
          const float v = __half2float(x[b * kPerByte + c]);
          byte |= code_of<Bits>(v, lo, scale) << (c * Bits);
        }
        out[b] = static_cast<std::uint8_t>(byte);
      }
    }

    // One half of the page, keys or values, one thread per byte of codes.
    // Element e of the half is row e / width. Its group is its channel,
    // e % width, for keys; for values it is its (token, head), e / head_dim,
    // which every code in a byte shares because head_dim * Bits is a
    // multiple of 8.
    template <int Bits, bool PerChannel>
    __global__ void dequantise_kernel(const std::uint8_t *__restrict__ codes,
                                      const __half *__restrict__ scales,
                                      const __half *__restrict__ zeros,
                                      __half *__restrict__ out, size_t bytes,
                                      int width, int head_dim)
    {
      constexpr int kPerByte = 8 / Bits;
      constexpr unsigned kMask = (1u << Bits) - 1;
      for (size_t i =
               static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
           i < bytes; i += static_cast<size_t>(gridDim.x) * blockDim.x)
      {
        const unsigned byte = codes[i];
        for (int c = 0; c < kPerByte; ++c)
        {
          const size_t e = i * kPerByte + c;
          const size_t group = PerChannel ? e % width : e / head_dim;
          const float q = static_cast<float>((byte >> (c * Bits)) & kMask);
          out[e] = __float2half_rn(
              fmaf(q, __half2float(scales[group]), __half2float(zeros[group])));
        }
      }
    }

    template <int Bits>
    void launch_quantise(const __half *keys, const __half *values,
                         std::uint8_t *page, const QuantisedPageLayout &layout,
                         int page_tokens, int kv_heads, int head_dim)
    {
      const int width = kv_heads * head_dim;
      const int columns = width / (8 / Bits);
      quantise_keys_kernel<Bits>
          <<<(columns + kBlockThreads - 1) / kBlockThreads, kBlockThreads>>>(
              keys, page + layout.key_codes,
              reinterpret_cast<__half *>(page + layout.key_scales),
              reinterpret_cast<__half *>(page + layout.key_zeros), page_tokens,
              width);
      cuda_check(cudaGetLastError(), "quantise keys kernel launch");

      const int groups = page_tokens * kv_heads;
      const int warps_per_block = kBlockThreads / kWarp;
      quantise_values_kernel<Bits>
          <<<(groups + warps_per_block - 1) / warps_per_block, kBlockThreads>>>(
              values, page + layout.value_codes,
              reinterpret_cast<__half *>(page + layout.value_scales),
              reinterpret_cast<__half *>(page + layout.value_zeros), groups,
              head_dim);
      cuda_check(cudaGetLastError(), "quantise values kernel launch");
    }

    template <int Bits>
    void launch_dequantise(const std::uint8_t *page, __half *keys,
                           __half *values, const QuantisedPageLayout &layout,
                           int kv_heads, int head_dim)
    {
      const size_t bytes = layout.value_codes - layout.key_codes;
      const int width = kv_heads * head_dim;
      const int grid = grid_stride_blocks(bytes, kBlockThreads);
      const auto meta = [page](std::size_t offset)
      { return reinterpret_cast<const __half *>(page + offset); };
      dequantise_kernel<Bits, true><<<grid, kBlockThreads>>>(
          page + layout.key_codes, meta(layout.key_scales),
          meta(layout.key_zeros), keys, bytes, width, head_dim);
      cuda_check(cudaGetLastError(), "dequantise keys kernel launch");
      dequantise_kernel<Bits, false><<<grid, kBlockThreads>>>(
          page + layout.value_codes, meta(layout.value_scales),
          meta(layout.value_zeros), values, bytes, width, head_dim);
      cuda_check(cudaGetLastError(), "dequantise values kernel launch");
    }

    // The one place a tier chooses its kernels: `launch` is called with the
    // code width as a compile-time constant. The kernels and the layout are
    // the same for every quantised tier; only the width differs.
    template <typename Launch> void with_code_width(Tier tier, Launch &&launch)
    {
      switch (tier)
      {
      case Tier::INT8:
        launch(std::integral_constant<int, 8>{});
        return;
      case Tier::INT4:
        launch(std::integral_constant<int, 4>{});
        return;
      case Tier::INT2:
        launch(std::integral_constant<int, 2>{});
        return;
      case Tier::FP16:
        bits_of(tier); // throws: FP16 is not quantised
      }
    }

    void check_filled(int filled, int page_tokens)
    {
      if (filled < page_tokens)
      {
        throw OpenPage(filled, page_tokens);
      }
      if (filled > page_tokens)
      {
        throw std::invalid_argument(std::to_string(filled) +
                                    " positions do not fit a page of " +
                                    std::to_string(page_tokens));
      }
    }

  } // namespace

  QuantisedPageLayout quantised_page_layout(Tier tier, int page_tokens,
                                            int kv_heads, int head_dim)
  {
    const int bits = bits_of(tier);
    require_positive(page_tokens, "page_tokens");
    require_positive(kv_heads, "kv_heads");
    require_positive(head_dim, "head_dim");
    if (head_dim * bits % 8 != 0)
    {
      throw std::invalid_argument(
          "head_dim " + std::to_string(head_dim) + " at " +
          std::to_string(bits) +
          " bits leaves a value group sharing a byte with the next");
    }
    const std::size_t width = static_cast<std::size_t>(kv_heads) * head_dim;
    const std::size_t codes =
        static_cast<std::size_t>(page_tokens) * width * bits / 8;
    const std::size_t key_meta = width * sizeof(__half);
    const std::size_t value_meta =
        static_cast<std::size_t>(page_tokens) * kv_heads * sizeof(__half);

    QuantisedPageLayout layout{};
    layout.bits = bits;
    layout.key_codes = 0;
    layout.value_codes = codes;
    layout.key_scales = 2 * codes;
    layout.key_zeros = layout.key_scales + key_meta;
    layout.value_scales = layout.key_zeros + key_meta;
    layout.value_zeros = layout.value_scales + value_meta;
    layout.page_bytes = layout.value_zeros + value_meta;
    layout.metadata_bytes = layout.page_bytes - 2 * codes;
    layout.effective_bits = 8.0 * static_cast<double>(layout.page_bytes) /
                            static_cast<double>(2 * page_tokens * width);
    return layout;
  }

  OpenPage::OpenPage(int filled, int page_tokens)
      : std::invalid_argument(
            "the page holds " + std::to_string(filled) + " of " +
            std::to_string(page_tokens) +
            " positions and is still being filled: a key channel's scale "
            "spans the whole page, so the open page stays at FP16 (ADR-0005)")
  {
  }

  void device::quantise_page(const __half *keys, const __half *values,
                             std::uint8_t *page, Tier tier, int page_tokens,
                             int filled, int kv_heads, int head_dim)
  {
    const QuantisedPageLayout layout =
        quantised_page_layout(tier, page_tokens, kv_heads, head_dim);
    check_filled(filled, page_tokens);
    with_code_width(tier,
                    [&](auto bits)
                    {
                      launch_quantise<decltype(bits)::value>(
                          keys, values, page, layout, page_tokens, kv_heads,
                          head_dim);
                    });
  }

  void device::dequantise_page(const std::uint8_t *page, __half *keys,
                               __half *values, Tier tier, int page_tokens,
                               int kv_heads, int head_dim)
  {
    const QuantisedPageLayout layout =
        quantised_page_layout(tier, page_tokens, kv_heads, head_dim);
    with_code_width(tier,
                    [&](auto bits)
                    {
                      launch_dequantise<decltype(bits)::value>(
                          page, keys, values, layout, kv_heads, head_dim);
                    });
  }

  void quantise_page(const float *keys, const float *values, std::uint8_t *page,
                     Tier tier, int page_tokens, int filled, int kv_heads,
                     int head_dim)
  {
    const QuantisedPageLayout layout =
        quantised_page_layout(tier, page_tokens, kv_heads, head_dim);
    check_filled(filled, page_tokens); // before anything is uploaded

    const size_t count = static_cast<size_t>(page_tokens) * kv_heads * head_dim;
    DeviceBuffer dev_keys(count * sizeof(__half));
    DeviceBuffer dev_values(count * sizeof(__half));
    DeviceBuffer dev_page(layout.page_bytes);
    upload_fp16(dev_keys, keys, count, "cudaMemcpy keys host-to-device");
    upload_fp16(dev_values, values, count, "cudaMemcpy values host-to-device");

    device::quantise_page(dev_keys.as<const __half>(),
                          dev_values.as<const __half>(),
                          dev_page.as<std::uint8_t>(), tier, page_tokens,
                          filled, kv_heads, head_dim);
    cuda_check(cudaDeviceSynchronize(), "quantise kernel execution");
    cuda_check(cudaMemcpy(page, dev_page.raw(), layout.page_bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy page device-to-host");
  }

  void dequantise_page(const std::uint8_t *page, float *keys, float *values,
                       Tier tier, int page_tokens, int kv_heads, int head_dim)
  {
    const QuantisedPageLayout layout =
        quantised_page_layout(tier, page_tokens, kv_heads, head_dim);

    const size_t count = static_cast<size_t>(page_tokens) * kv_heads * head_dim;
    DeviceBuffer dev_page(layout.page_bytes);
    DeviceBuffer dev_keys(count * sizeof(__half));
    DeviceBuffer dev_values(count * sizeof(__half));
    cuda_check(cudaMemcpy(dev_page.raw(), page, layout.page_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy page host-to-device");

    device::dequantise_page(dev_page.as<const std::uint8_t>(),
                            dev_keys.as<__half>(), dev_values.as<__half>(),
                            tier, page_tokens, kv_heads, head_dim);
    cuda_check(cudaDeviceSynchronize(), "dequantise kernel execution");
    download_fp16(keys, dev_keys, count, "cudaMemcpy keys device-to-host");
    download_fp16(values, dev_values, count,
                  "cudaMemcpy values device-to-host");
  }

} // namespace microinfer
