#pragma once

// Two fp16 numbers in one 32-bit register, as the tensor cores read them and
// as adjacent channels are loaded. Device code.

#include <cuda_fp16.h>

#include <cstdint>

namespace microinfer
{
  namespace
  {

    __device__ __forceinline__ uint32_t pack(__half low, __half high)
    {
      return (static_cast<uint32_t>(__half_as_ushort(high)) << 16) |
             __half_as_ushort(low);
    }

    // The first (0) or second (1) of two, widened to fp32.
    __device__ __forceinline__ float element(uint32_t two, int e)
    {
      return __half2float(
          __ushort_as_half(static_cast<unsigned short>(two >> (16 * e))));
    }

    // The two that begin at `at`, whose address is a multiple of 4 bytes.
    __device__ __forceinline__ uint32_t two_halves(const __half *at)
    {
      return *reinterpret_cast<const uint32_t *>(at);
    }

  } // namespace
} // namespace microinfer
