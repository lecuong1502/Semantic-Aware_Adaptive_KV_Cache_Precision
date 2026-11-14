#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "microinfer/check.h"
#include "microinfer/device_ops.h"

namespace microinfer
{
  namespace
  {

    constexpr int kBlockThreads = 256;
    // A grid's second dimension holds at most this many copies a launch.
    constexpr int kMaxGridY = 65535;
    // Blocks a copy, at most: a page is tens of KiB, and many pages copy at
    // once, so a few blocks each keep the device busy.
    constexpr std::size_t kBlocksPerCopy = 8;

    // Copy blockIdx.y + first: bytes from from[i] to to[i], in units of
    // Unit, the blocks of x striding over them. Copies are independent:
    // no copy may read what another writes.
    template <typename Unit>
    __global__ void copy_pages_kernel(const void *const *__restrict__ from,
                                      void *const *__restrict__ to,
                                      std::size_t units, int first)
    {
      const auto *src = static_cast<const Unit *>(from[first + blockIdx.y]);
      auto *dst = static_cast<Unit *>(to[first + blockIdx.y]);
      for (std::size_t i =
               static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
           i < units; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
      {
        dst[i] = src[i];
      }
    }

  } // namespace

  void device::copy_pages(const void *const *from, void *const *to,
                          std::size_t bytes, int count, bool aligned16)
  {
    if (count <= 0 || bytes == 0)
    {
      return;
    }
    const auto launch = [&](auto unit)
    {
      using Unit = decltype(unit);
      const std::size_t units = bytes / sizeof(Unit);
      const int blocks = static_cast<int>(std::min<std::size_t>(
          (units + kBlockThreads - 1) / kBlockThreads, kBlocksPerCopy));
      for (int first = 0; first < count; first += kMaxGridY)
      {
        const int n = std::min(kMaxGridY, count - first);
        copy_pages_kernel<Unit>
            <<<dim3(blocks, n), kBlockThreads>>>(from, to, units, first);
        cuda_check(cudaGetLastError(), "copy pages kernel launch");
      }
    };
    if (aligned16 && bytes % sizeof(uint4) == 0)
    {
      launch(uint4{});
    }
    else
    {
      launch(std::uint8_t{});
    }
  }

} // namespace microinfer
