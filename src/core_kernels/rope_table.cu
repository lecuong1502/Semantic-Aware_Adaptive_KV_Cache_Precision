#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "microinfer/check.h"
#include "microinfer/launch.h"
#include "microinfer/rope_angle.cuh"
#include "microinfer/rope_table.h"

namespace microinfer
{
  namespace
  {

    constexpr int kBlockThreads = 256;

    __global__ void rope_table_kernel(float *__restrict__ table, int half,
                                      int head_dim, double theta, size_t count)
    {
      for (size_t i =
               static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
           i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
      {
        double s, c;
        rope_sincos(static_cast<double>(i / half), static_cast<int>(i % half),
                    head_dim, theta, &s, &c);
        table[2 * i] = static_cast<float>(c);
        table[2 * i + 1] = static_cast<float>(s);
      }
    }

  } // namespace

  RopeTable::RopeTable(int head_dim, double theta)
      : head_dim_(head_dim), theta_(theta)
  {
    if (head_dim <= 0 || head_dim % 2 != 0)
    {
      throw std::invalid_argument("head_dim must be positive and even, got " +
                                  std::to_string(head_dim));
    }
    if (!(theta > 0.0))
    {
      throw std::invalid_argument("theta must be positive, got " +
                                  std::to_string(theta));
    }
  }

  void RopeTable::cover(int count)
  {
    if (count <= positions_)
    {
      return;
    }
    const int grown = count > 2 * positions_ ? count : 2 * positions_;
    const int half = head_dim_ / 2;
    const size_t entries = static_cast<size_t>(grown) * half;
    auto table = std::make_unique<DeviceBuffer>(2 * entries * sizeof(float));
    rope_table_kernel<<<grid_stride_blocks(entries, kBlockThreads),
                        kBlockThreads>>>(table->as<float>(), half, head_dim_,
                                         theta_, entries);
    cuda_check(cudaGetLastError(), "rope table kernel launch");
    // The old table may still be read by a launch already enqueued; freeing
    // it waits for the device, as cudaFree does.
    table_ = std::move(table);
    positions_ = grown;
  }

  const float *RopeTable::data() const
  {
    return table_ ? table_->as<const float>() : nullptr;
  }

  std::size_t RopeTable::nbytes() const
  {
    return static_cast<size_t>(positions_) * head_dim_ * sizeof(float);
  }

} // namespace microinfer
