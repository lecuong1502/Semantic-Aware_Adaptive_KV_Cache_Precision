#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "microinfer/check.h"
#include "microinfer/device_ops.h"
#include "microinfer/launch.h"

namespace microinfer
{
  namespace
  {

    constexpr int kBlockThreads = 256;

    // Not __restrict__: the residual stream is updated in place, so out is a
    // or b. Each element is read before it is written, by the same thread.
    __global__ void add_kernel(const __half *a, const __half *b, __half *out,
                               size_t count)
    {
      for (size_t i =
               static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
           i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
      {
        out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
      }
    }

  } // namespace

  void device::add(const __half *a, const __half *b, __half *out, size_t count)
  {
    if (count == 0)
    {
      return;
    }
    add_kernel<<<grid_stride_blocks(count, kBlockThreads), kBlockThreads>>>(
        a, b, out, count);
    cuda_check(cudaGetLastError(), "add kernel launch");
  }

} // namespace microinfer
