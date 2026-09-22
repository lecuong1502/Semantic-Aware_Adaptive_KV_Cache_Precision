#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

#include "microinfer/check.h"
#include "microinfer/kernels.h"

namespace microinfer {
namespace {

constexpr int kBlockThreads = 256;

// One block per row. Accumulation is fp32 throughout; only the store is fp16,
// so the dominant error is a single rounding at the end rather than a drift
// accumulated across `hidden` additions.
__global__ void rmsnorm_kernel(const __half* __restrict__ x,
                               const __half* __restrict__ weight,
                               __half* __restrict__ out, int hidden,
                               float eps) {
  extern __shared__ float partials[];

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
    // Every partial has been consumed, so partials[0] is free to reuse as the
    // broadcast slot for the scale.
    partials[0] = rsqrtf(total / static_cast<float>(hidden) + eps);
  }
  __syncthreads();

  const float scale = partials[0];
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = __half2float(x[row + i]) * scale * __half2float(weight[i]);
    out[row + i] = __float2half(v);
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) {
    cuda_check(cudaMalloc(&ptr_, bytes), "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  template <typename T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }

 private:
  void* ptr_ = nullptr;
};

}  // namespace

void rmsnorm(const float* x, const float* weight, float* out, int rows,
             int hidden, float eps) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }

  const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(hidden);

  std::vector<__half> host_x(count);
  std::vector<__half> host_w(hidden);
  for (size_t i = 0; i < count; ++i) {
    host_x[i] = __float2half(x[i]);
  }
  for (int i = 0; i < hidden; ++i) {
    host_w[i] = __float2half(weight[i]);
  }

  DeviceBuffer dev_x(count * sizeof(__half));
  DeviceBuffer dev_w(static_cast<size_t>(hidden) * sizeof(__half));
  DeviceBuffer dev_out(count * sizeof(__half));

  cuda_check(cudaMemcpy(dev_x.as<void>(), host_x.data(), count * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy x host-to-device");
  cuda_check(cudaMemcpy(dev_w.as<void>(), host_w.data(),
                        static_cast<size_t>(hidden) * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy weight host-to-device");

  const int warps = (kBlockThreads + 31) / 32;
  const size_t shared = static_cast<size_t>(warps) * sizeof(float);
  rmsnorm_kernel<<<rows, kBlockThreads, shared>>>(
      dev_x.as<const __half>(), dev_w.as<const __half>(),
      dev_out.as<__half>(), hidden, eps);
  cuda_check(cudaGetLastError(), "rmsnorm kernel launch");

  std::vector<__half> host_out(count);
  cuda_check(cudaMemcpy(host_out.data(), dev_out.as<void>(),
                        count * sizeof(__half), cudaMemcpyDeviceToHost),
             "cudaMemcpy output device-to-host");

  for (size_t i = 0; i < count; ++i) {
    out[i] = __half2float(host_out[i]);
  }
}

}  // namespace microinfer
