#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "microinfer/check.h"
#include "microinfer/device_buffer.h"
#include "microinfer/kernels.h"
#include "microinfer/staging.h"

namespace microinfer
{
  namespace
  {

    constexpr int kBlockThreads = 128;

    // One thread block per (token, head); one thread per rotation pair.
    // Qwen2's "rotate half" pairs dimension j with j + head_dim/2, not with
    // j + 1:
    //
    //   out[j]        = x[j] * cos - x[j + half] * sin
    //   out[j + half] = x[j + half] * cos + x[j] * sin
    //
    // The angle is formed and reduced in fp64. In fp32, pos * inv_freq at the
    // fastest frequency is off by ~2e-3 rad at position 32767 — four fp16 ulps
    // of the output, before the kernel has done anything else. fp64 is slow on
    // consumer parts, but this is one sincos per pair per token, which is
    // noise next to the projections around it.
    __global__ void rope_kernel(const __half *__restrict__ x,
                                const int32_t *__restrict__ positions,
                                __half *__restrict__ out, int heads,
                                int head_dim, double theta)
    {
      const int token = blockIdx.x / heads;
      const size_t base = static_cast<size_t>(blockIdx.x) * head_dim;
      const int half = head_dim / 2;
      const double pos = static_cast<double>(positions[token]);

      for (int j = threadIdx.x; j < half; j += blockDim.x)
      {
        const double inv_freq =
            pow(theta, -2.0 * static_cast<double>(j) / head_dim);
        double s, c;
        sincos(pos * inv_freq, &s, &c);
        const float sf = static_cast<float>(s);
        const float cf = static_cast<float>(c);

        const float x1 = __half2float(x[base + j]);
        const float x2 = __half2float(x[base + j + half]);
        out[base + j] = __float2half(x1 * cf - x2 * sf);
        out[base + j + half] = __float2half(x2 * cf + x1 * sf);
      }
    }

  } // namespace

  void rope(const float *x, const int32_t *positions, float *out, int seq,
            int heads, int head_dim, double theta)
  {
    if (seq <= 0 || heads <= 0 || head_dim <= 0)
    {
      return; // No elements exist, so there is nothing to write.
    }

    const size_t count = static_cast<size_t>(seq) * heads * head_dim;
    const size_t pos_bytes = static_cast<size_t>(seq) * sizeof(int32_t);

    DeviceBuffer dev_x(count * sizeof(__half));
    DeviceBuffer dev_pos(pos_bytes);
    DeviceBuffer dev_out(count * sizeof(__half));

    upload_fp16(dev_x, x, count, "cudaMemcpy x host-to-device");
    cuda_check(
        cudaMemcpy(dev_pos.raw(), positions, pos_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy positions host-to-device");

    rope_kernel<<<seq * heads, kBlockThreads>>>(
        dev_x.as<const __half>(), dev_pos.as<const int32_t>(),
        dev_out.as<__half>(), heads, head_dim, theta);
    cuda_check(cudaGetLastError(), "rope kernel launch");
    cuda_check(cudaDeviceSynchronize(), "rope kernel execution");

    download_fp16(out, dev_out, count, "cudaMemcpy output device-to-host");
  }

} // namespace microinfer
