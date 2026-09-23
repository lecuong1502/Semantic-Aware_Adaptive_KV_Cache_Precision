#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <optional>

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

    // One handle for the life of the process, created on first use.
    //
    // Deliberately never destroyed. cublasCreate costs tens of milliseconds and
    // a per-call handle would dominate every projection; and a static
    // destructor calling cublasDestroy runs after the CUDA runtime may already
    // have torn the context down, which is a crash at interpreter exit in
    // exchange for freeing memory the driver reclaims anyway.
    //
    // The handle holds a small workspace on the device from first use onward.
    // That is a fixed cost paid once, not memory that comes and goes, so it
    // does not disturb a free-memory reading taken on either side of an
    // allocation (ADR-0007).
    cublasHandle_t handle()
    {
      static cublasHandle_t h = []
      {
        cublasHandle_t created = nullptr;
        cublas_check(cublasCreate(&created), "cublasCreate");
        // With an fp16 output and fp32 compute, cuBLAS is otherwise free to
        // reduce split-K partial sums in the *output* precision. That is a
        // silent fp16 accumulation inside a call the caller asked to
        // accumulate in fp32, and it is exactly the error ADR-0006's gate is
        // there to catch — better not to make it possible.
        cublas_check(
            cublasSetMathMode(
                created, static_cast<cublasMath_t>(
                             CUBLAS_DEFAULT_MATH |
                             CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)),
            "cublasSetMathMode");
        return created;
      }();
      return h;
    }

    // Seeds every output row with the bias, so the GEMM can add onto it with
    // beta = 1. The sum is then formed in fp32 inside cuBLAS and rounded to
    // fp16 once, rather than rounded after the GEMM and again after a separate
    // bias add.
    __global__ void broadcast_rows_kernel(const __half *__restrict__ row,
                                          __half *__restrict__ out, int cols,
                                          size_t count)
    {
      for (size_t i =
               static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
           i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
      {
        out[i] = row[i % cols];
      }
    }

  } // namespace

  namespace
  {

    // HuggingFace stores a projection as W (out_features, in_features) and
    // computes y = x W^T, all row-major. cuBLAS is column-major, where a
    // row-major matrix reads as its own transpose. So the call computes
    //
    //   y^T (out, rows) = op(W) (out, in) * x^T (in, rows)
    //
    // where W, read column-major, is (in, out) and needs OP_T, and x, read
    // column-major, is already x^T and needs OP_N. The result y^T in
    // column-major is y in row-major, which is what the caller gets back.
    void gemm(const __half *x, const __half *weight, void *out,
              cudaDataType_t out_type, float beta, int rows, int in_features,
              int out_features)
    {
      const float alpha = 1.0f;
      cublas_check(cublasGemmEx(handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                                out_features, rows, in_features, &alpha, weight,
                                CUDA_R_16F, in_features, x, CUDA_R_16F,
                                in_features, &beta, out, out_type, out_features,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                   "cublasGemmEx");
    }

  } // namespace

  void device::linear(const __half *x, const __half *weight, const __half *bias,
                      __half *out, int rows, int in_features, int out_features)
  {
    if (rows <= 0 || out_features <= 0)
    {
      return;
    }
    float beta = 0.0f;
    if (bias != nullptr)
    {
      const size_t out_count = static_cast<size_t>(rows) * out_features;
      const int grid = grid_stride_blocks(out_count, kBlockThreads);
      broadcast_rows_kernel<<<grid, kBlockThreads>>>(bias, out, out_features,
                                                     out_count);
      cuda_check(cudaGetLastError(), "bias broadcast launch");
      beta = 1.0f;
    }
    gemm(x, weight, out, CUDA_R_16F, beta, rows, in_features, out_features);
  }

  void device::linear_fp32_out(const __half *x, const __half *weight,
                               float *out, int rows, int in_features,
                               int out_features)
  {
    if (rows <= 0 || out_features <= 0)
    {
      return;
    }
    gemm(x, weight, out, CUDA_R_32F, 0.0f, rows, in_features, out_features);
  }

  void linear(const float *x, const float *weight, const float *bias,
              float *out, int rows, int in_features, int out_features)
  {
    if (rows <= 0 || out_features <= 0)
    {
      return; // No elements exist, so there is nothing to write.
    }

    const size_t x_count = static_cast<size_t>(rows) * in_features;
    const size_t w_count = static_cast<size_t>(out_features) * in_features;
    const size_t out_count = static_cast<size_t>(rows) * out_features;

    DeviceBuffer dev_x(x_count * sizeof(__half));
    DeviceBuffer dev_w(w_count * sizeof(__half));
    DeviceBuffer dev_out(out_count * sizeof(__half));
    std::optional<DeviceBuffer> dev_bias;

    upload_fp16(dev_x, x, x_count, "cudaMemcpy x host-to-device");
    upload_fp16(dev_w, weight, w_count, "cudaMemcpy weight host-to-device");
    if (bias != nullptr)
    {
      dev_bias.emplace(static_cast<size_t>(out_features) * sizeof(__half));
      upload_fp16(*dev_bias, bias, out_features,
                  "cudaMemcpy bias host-to-device");
    }

    // With a bias, the output is seeded with it and the GEMM adds onto it
    // with beta = 1, so the sum is formed in fp32 inside cuBLAS and rounded to
    // fp16 once, rather than rounded after the GEMM and again after an add.
    device::linear(dev_x.as<const __half>(), dev_w.as<const __half>(),
                   dev_bias ? dev_bias->as<const __half>() : nullptr,
                   dev_out.as<__half>(), rows, in_features, out_features);
    // Synchronised before dev_bias and the rest go out of scope and are freed.
    cuda_check(cudaDeviceSynchronize(), "cublasGemmEx execution");

    download_fp16(out, dev_out, out_count, "cudaMemcpy output device-to-host");
  }

} // namespace microinfer
