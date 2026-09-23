#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "microinfer/check.h"
#include "microinfer/device_buffer.h"
#include "microinfer/device_ops.h"
#include "microinfer/kernels.h"
#include "microinfer/launch.h"
#include "microinfer/staging.h"

namespace microinfer
{
  namespace
  {

    constexpr int kBlockThreads = 256;

    // Elementwise, so the layout is irrelevant and a flat grid-stride loop
    // serves every shape.
    //
    // Written as g / (1 + exp(-g)) rather than g * sigmoid(g) computed some
    // other way. For g < -88, exp(-g) overflows fp32 to +inf and the quotient
    // is a correctly signed zero; no form here divides inf by inf. expf rather
    // than __expf: the intrinsic's error is several ulps of fp32 at large
    // arguments, and there is no speed to buy in a kernel this memory-bound.
    __global__ void swiglu_kernel(const __half *__restrict__ gate,
                                  const __half *__restrict__ up,
                                  __half *__restrict__ out, size_t count)
    {
      for (size_t i =
               static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
           i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
      {
        const float g = __half2float(gate[i]);
        const float silu = g / (1.0f + expf(-g));
        out[i] = __float2half(silu * __half2float(up[i]));
      }
    }

  } // namespace

  void device::swiglu(const __half *gate, const __half *up, __half *out,
                      size_t count)
  {
    if (count == 0)
    {
      return;
    }
    const int grid = grid_stride_blocks(count, kBlockThreads);
    swiglu_kernel<<<grid, kBlockThreads>>>(gate, up, out, count);
    cuda_check(cudaGetLastError(), "swiglu kernel launch");
  }

  void swiglu(const float *gate, const float *up, float *out, size_t count)
  {
    if (count == 0)
    {
      return; // No elements exist, so there is nothing to write.
    }

    const size_t bytes = count * sizeof(__half);
    DeviceBuffer dev_gate(bytes);
    DeviceBuffer dev_up(bytes);
    DeviceBuffer dev_out(bytes);

    upload_fp16(dev_gate, gate, count, "cudaMemcpy gate host-to-device");
    upload_fp16(dev_up, up, count, "cudaMemcpy up host-to-device");

    device::swiglu(dev_gate.as<const __half>(), dev_up.as<const __half>(),
                   dev_out.as<__half>(), count);
    cuda_check(cudaDeviceSynchronize(), "swiglu kernel execution");

    download_fp16(out, dev_out, count, "cudaMemcpy output device-to-host");
  }

} // namespace microinfer
