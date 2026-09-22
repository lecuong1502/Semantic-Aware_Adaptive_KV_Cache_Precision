#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

#include "microinfer/check.h"
#include "microinfer/device_buffer.h"
#include "microinfer/kernels.h"

namespace microinfer {
namespace {

constexpr int kBlockThreads = 256;

// The largest warp count any block can have, since CUDA caps a block at 1024
// threads and a warp is 32 lanes. Sizing the shared allocation by this bound
// rather than by kBlockThreads/32 keeps the host from having to agree with the
// device about warpSize: the kernel reads the real warpSize at runtime, and
// this array is large enough whatever it turns out to be.
constexpr int kMaxWarpsPerBlock = 32;

// One block per row. Accumulation is fp32 throughout; only the store is fp16,
// so the error is a single rounding at the end rather than a drift accumulated
// across `hidden` additions.
__global__ void rmsnorm_kernel(const __half* __restrict__ x,
                               const __half* __restrict__ weight,
                               __half* __restrict__ out, int hidden,
                               float eps) {
  __shared__ float partials[kMaxWarpsPerBlock];
  __shared__ float scale;

  const size_t row = static_cast<size_t>(blockIdx.x) * hidden;

  float acc = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = __half2float(x[row + i]);
    acc += v * v;
  }

  // Reduce within each warp by register shuffle, which needs neither shared
  // memory nor __syncthreads. Only the cross-warp step below pays for those.
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  }

  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  const int warps = (blockDim.x + warpSize - 1) / warpSize;

  if (lane == 0) {
    partials[warp] = acc;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int i = 0; i < warps; ++i) {
      total += partials[i];
    }
    scale = rsqrtf(total / static_cast<float>(hidden) + eps);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = __half2float(x[row + i]) * scale * __half2float(weight[i]);
    out[row + i] = __float2half(v);
  }
}

}  // namespace

void rmsnorm(const float* x, const float* weight, float* out, int rows,
             int hidden, float eps) {
  if (rows <= 0 || hidden <= 0) {
    return;  // No elements exist, so there is nothing to write.
  }

  const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(hidden);
  const size_t weight_bytes = static_cast<size_t>(hidden) * sizeof(__half);

  // The caller's arrays are fp32 but the kernel's domain is fp16, which is what
  // the engine stores. This downcast is therefore a real rounding, not a
  // formality: a caller passing arbitrary fp32 pays for it on top of the output
  // rounding. See tests/test_rmsnorm.py, which measures both.
  std::vector<__half> host_x(count);
  std::vector<__half> host_w(hidden);
  for (size_t i = 0; i < count; ++i) {
    host_x[i] = __float2half(x[i]);
  }
  for (int i = 0; i < hidden; ++i) {
    host_w[i] = __float2half(weight[i]);
  }

  DeviceBuffer dev_x(count * sizeof(__half));
  DeviceBuffer dev_w(weight_bytes);
  DeviceBuffer dev_out(count * sizeof(__half));

  cuda_check(cudaMemcpy(dev_x.raw(), host_x.data(), count * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy x host-to-device");
  cuda_check(cudaMemcpy(dev_w.raw(), host_w.data(), weight_bytes,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy weight host-to-device");

  rmsnorm_kernel<<<rows, kBlockThreads>>>(dev_x.as<const __half>(),
                                          dev_w.as<const __half>(),
                                          dev_out.as<__half>(), hidden, eps);
  cuda_check(cudaGetLastError(), "rmsnorm kernel launch");
  // Without this, an execution fault surfaces under the next cudaMemcpy and is
  // reported against the wrong operation.
  cuda_check(cudaDeviceSynchronize(), "rmsnorm kernel execution");

  std::vector<__half> host_out(count);
  cuda_check(cudaMemcpy(host_out.data(), dev_out.raw(), count * sizeof(__half),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy output device-to-host");

  for (size_t i = 0; i < count; ++i) {
    out[i] = __half2float(host_out[i]);
  }
}

}  // namespace microinfer
