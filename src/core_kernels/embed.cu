#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "microinfer/check.h"
#include "microinfer/device_ops.h"
#include "microinfer/launch.h"

namespace microinfer
{
  namespace
  {

    constexpr int kBlockThreads = 256;

    // A gather, so no arithmetic and no rounding: the output is the table's
    // fp16 row, bit for bit. An id outside the vocabulary is rejected by the
    // caller before launch; the guard here only keeps a bad id from reading
    // past the table, and leaves that row zero.
    __device__ inline void put(__half *out, __half v) { *out = v; }
    // Widening fp16 to fp32 is exact, so the fp32 residual stream starts from
    // the table's own values (ADR-0010).
    __device__ inline void put(float *out, __half v) { *out = __half2float(v); }

    template <typename Out>
    __global__ void embed_kernel(const int32_t *__restrict__ ids,
                                 const __half *__restrict__ table,
                                 Out *__restrict__ out, int hidden, int vocab,
                                 size_t count)
    {
      for (size_t i =
               static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
           i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
      {
        const size_t token = i / hidden;
        const int32_t id = ids[token];
        put(&out[i], (id >= 0 && id < vocab)
                         ? table[static_cast<size_t>(id) * hidden + i % hidden]
                         : __float2half(0.0f));
      }
    }

  } // namespace

  namespace
  {

    template <typename Out>
    void launch(const int32_t *ids, const __half *table, Out *out, int count,
                int hidden, int vocab)
    {
      const size_t total = static_cast<size_t>(count) * hidden;
      if (total == 0)
      {
        return;
      }
      embed_kernel<<<grid_stride_blocks(total, kBlockThreads), kBlockThreads>>>(
          ids, table, out, hidden, vocab, total);
      cuda_check(cudaGetLastError(), "embed kernel launch");
    }

  } // namespace

  void device::embed(const int32_t *ids, const __half *table, __half *out,
                     int count, int hidden, int vocab)
  {
    launch(ids, table, out, count, hidden, vocab);
  }

  void device::embed_f32(const int32_t *ids, const __half *table, float *out,
                         int count, int hidden, int vocab)
  {
    launch(ids, table, out, count, hidden, vocab);
  }

} // namespace microinfer
