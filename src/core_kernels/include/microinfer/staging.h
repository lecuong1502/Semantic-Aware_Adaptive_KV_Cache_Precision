#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "microinfer/check.h"
#include "microinfer/device_buffer.h"

namespace microinfer
{

  // Seam B's host side: every kernel entry point takes fp32 NumPy and works in
  // fp16 on the device. The conversion happens on the host so the device never
  // holds an fp32 copy.
  //
  // The downcast is a real rounding, not a formality. A caller passing
  // arbitrary fp32 pays for it on top of the kernel's own output rounding; the
  // kernel tests measure both.
  inline void upload_fp16(DeviceBuffer &dst, const float *host, size_t count,
                          const char *what)
  {
    std::vector<__half> staged(count);
    for (size_t i = 0; i < count; ++i)
    {
      staged[i] = __float2half(host[i]);
    }
    cuda_check(cudaMemcpy(dst.raw(), staged.data(), count * sizeof(__half),
                          cudaMemcpyHostToDevice),
               what);
  }

  inline void download_fp16(float *host, const DeviceBuffer &src, size_t count,
                            const char *what)
  {
    std::vector<__half> staged(count);
    cuda_check(cudaMemcpy(staged.data(), src.raw(), count * sizeof(__half),
                          cudaMemcpyDeviceToHost),
               what);
    for (size_t i = 0; i < count; ++i)
    {
      host[i] = __half2float(staged[i]);
    }
  }

} // namespace microinfer
