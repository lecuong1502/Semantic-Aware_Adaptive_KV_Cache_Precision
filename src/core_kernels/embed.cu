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
    __global__ void embed_kernel(const int32_t *__restrict__ ids,
                                 const __half *__restrict__ table,
                                 __half *__restrict__ out, int hidden,
                                 int vocab, size_t count)
    {
      for (size_t i =
               static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
           i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
      {
        const size_t token = i / hidden;
        const int32_t id = ids[token];
        out[i] = (id >= 0 && id < vocab)
                     ? table[static_cast<size_t>(id) * hidden + i % hidden]
                     : __float2half(0.0f);
      }
    }

  } // namespace

  void device::embed(const int32_t *ids, const __half *table, __half *out,
                     int count, int hidden, int vocab)
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

} // namespace microinfer
