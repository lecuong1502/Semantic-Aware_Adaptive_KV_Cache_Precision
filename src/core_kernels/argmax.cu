#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

#include "microinfer/check.h"
#include "microinfer/device_ops.h"

namespace microinfer
{
  namespace
  {

    constexpr int kBlockThreads = 256;

    // The larger of two candidates, the lower index on a tie. Every reduction
    // step goes through this, so the tie rule holds whatever order the
    // candidates meet in. NaN never wins: a comparison with it is false.
    __device__ void keep_better(float &best, int &at, float value, int index)
    {
      if (value > best || (value == best && index < at))
      {
        best = value;
        at = index;
      }
    }

    // One block per row: a strided scan per thread, then a shuffle reduction
    // within each warp and a shared one across warps.
    __global__ void argmax_kernel(const float *__restrict__ x, int cols,
                                  int32_t *__restrict__ out)
    {
      __shared__ float best_s[32];
      __shared__ int at_s[32];

      const float *row = x + static_cast<size_t>(blockIdx.x) * cols;
      float best = -FLT_MAX;
      int at = cols; // beyond any real index, so any real value replaces it
      for (int c = threadIdx.x; c < cols; c += blockDim.x)
      {
        keep_better(best, at, row[c], c);
      }

      for (int offset = warpSize / 2; offset > 0; offset >>= 1)
      {
        keep_better(best, at, __shfl_down_sync(0xffffffffu, best, offset),
                    __shfl_down_sync(0xffffffffu, at, offset));
      }
      const int lane = threadIdx.x & (warpSize - 1);
      const int warp = threadIdx.x / warpSize;
      if (lane == 0)
      {
        best_s[warp] = best;
        at_s[warp] = at;
      }
      __syncthreads();

      if (threadIdx.x == 0)
      {
        const int warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 1; w < warps; ++w)
        {
          keep_better(best_s[0], at_s[0], best_s[w], at_s[w]);
        }
        out[blockIdx.x] = at_s[0];
      }
    }

  } // namespace

  void device::argmax_rows(const float *x, int rows, int cols, int32_t *out)
  {
    if (rows <= 0 || cols <= 0)
    {
      return;
    }
    argmax_kernel<<<rows, kBlockThreads>>>(x, cols, out);
    cuda_check(cudaGetLastError(), "argmax kernel launch");
  }

} // namespace microinfer
