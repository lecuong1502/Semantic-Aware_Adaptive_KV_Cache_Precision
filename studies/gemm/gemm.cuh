#pragma once

// A hand-written GEMM in two stages, against cuBLAS: the CUDA learning
// objective of the research notes' §5.1, kept off the inference path by
// ADR-0001. Nothing in src/ may include this file;
// tests/test_studies_isolation.py asserts it.
//
// Every implementation computes what the engine's projection does:
// y = x W^T, with x (rows, in_features) and W (out_features, in_features) as
// the checkpoint stores it, fp16 operands, fp32 accumulation, and one fp16
// rounding on store. Only the schedule differs, which is what makes the timings
// comparable.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "microinfer/check.h"

namespace gemm_study
{

  using microinfer::cublas_check;
  using microinfer::cuda_check;

  // One output per thread for both kernels, and a square tile for the tiled
  // one: a 32 x 32 thread block, which is also one warp per row of outputs.
  constexpr int kTile = 32;

  // Stage 1. One thread per output, each walking the whole reduction straight
  // from global memory.
  //
  // threadIdx.x runs along out_features, so a warp's 32 threads read 32
  // different rows of W at the same k: addresses in_features * 2 bytes apart,
  // one memory transaction each. x[m, k] is the same address for the whole
  // warp, a broadcast. Nothing is reused across threads except through L1/L2.
  __global__ void naive_kernel(const __half *__restrict__ x,
                               const __half *__restrict__ w,
                               __half *__restrict__ y, int rows,
                               int in_features, int out_features)
  {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= rows || n >= out_features)
    {
      return;
    }
    const __half *x_row = x + static_cast<size_t>(m) * in_features;
    const __half *w_row = w + static_cast<size_t>(n) * in_features;
    float acc = 0.0f;
    for (int k = 0; k < in_features; ++k)
    {
      acc += __half2float(x_row[k]) * __half2float(w_row[k]);
    }
    y[static_cast<size_t>(m) * out_features + n] = __float2half(acc);
  }

  // Stage 2. The block stages a kTile x kTile tile of x and one of W in shared
  // memory per step along the reduction, then each thread reads its row of the
  // x tile and its column of the W tile from there.
  //
  // Both loads are coalesced: threadIdx.x runs along k, the contiguous axis of
  // both operands. Each value loaded from global memory is used kTile times
  // from shared memory instead of once, which is the whole point. The W tile
  // is stored transposed-padded (kTile + 1 columns) so that reading it down a
  // column in the inner loop does not put a warp's 32 threads on one bank.
  __global__ void tiled_kernel(const __half *__restrict__ x,
                               const __half *__restrict__ w,
                               __half *__restrict__ y, int rows,
                               int in_features, int out_features)
  {
    __shared__ float x_s[kTile][kTile];
    __shared__ float w_s[kTile][kTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * kTile + ty;
    const int n = blockIdx.x * kTile + tx;
    // The row of W this thread loads: the tile's rows are output features,
    // loaded by ty the same way the x tile's rows are.
    const int w_load_row = blockIdx.x * kTile + ty;

    float acc = 0.0f;
    for (int k0 = 0; k0 < in_features; k0 += kTile)
    {
      const int k = k0 + tx;
      x_s[ty][tx] =
          (m < rows && k < in_features)
              ? __half2float(x[static_cast<size_t>(m) * in_features + k])
              : 0.0f;
      w_s[tx][ty] =
          (w_load_row < out_features && k < in_features)
              ? __half2float(
                    w[static_cast<size_t>(w_load_row) * in_features + k])
              : 0.0f;
      __syncthreads();

      for (int kk = 0; kk < kTile; ++kk)
      {
        acc += x_s[ty][kk] * w_s[kk][tx];
      }
      __syncthreads();
    }

    if (m < rows && n < out_features)
    {
      y[static_cast<size_t>(m) * out_features + n] = __float2half(acc);
    }
  }

  inline dim3 grid_for(int rows, int out_features)
  {
    return dim3((out_features + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  }

  inline void naive(const __half *x, const __half *w, __half *y, int rows,
                    int in_features, int out_features)
  {
    naive_kernel<<<grid_for(rows, out_features), dim3(kTile, kTile)>>>(
        x, w, y, rows, in_features, out_features);
    cuda_check(cudaGetLastError(), "naive GEMM launch");
  }

  inline void tiled(const __half *x, const __half *w, __half *y, int rows,
                    int in_features, int out_features)
  {
    tiled_kernel<<<grid_for(rows, out_features), dim3(kTile, kTile)>>>(
        x, w, y, rows, in_features, out_features);
    cuda_check(cudaGetLastError(), "tiled GEMM launch");
  }

  // The baseline, configured as the engine's projection is
  // (src/core_kernels/linear.cu): fp16 in and out, fp32 compute, and reduced-
  // precision split-K reduction disallowed. Duplicated rather than shared
  // because the engine's handle is private to it, and reaching into it from
  // here would be a dependency pointing the wrong way.
  // test_gemm_study.py asserts the two agree bit for bit.
  inline cublasHandle_t cublas_handle()
  {
    static cublasHandle_t h = []
    {
      cublasHandle_t created = nullptr;
      cublas_check(cublasCreate(&created), "cublasCreate");
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

  inline void cublas(const __half *x, const __half *w, __half *y, int rows,
                     int in_features, int out_features)
  {
    // Row-major y = x W^T is column-major y^T = W x^T: W, stored row-major
    // (out, in), is already column-major (in, out) and is transposed; x is
    // column-major (in, rows) as it stands.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(cublasGemmEx(cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                              out_features, rows, in_features, &alpha, w,
                              CUDA_R_16F, in_features, x, CUDA_R_16F,
                              in_features, &beta, y, CUDA_R_16F, out_features,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx");
  }

} // namespace gemm_study
