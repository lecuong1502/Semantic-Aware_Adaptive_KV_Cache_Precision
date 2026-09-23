#pragma once

#include <cstddef>

namespace microinfer
{

  // A cap on the grid of an elementwise, grid-stride kernel. Enough thread
  // blocks to fill any device this project targets several times over; the
  // grid-stride loop covers whatever is left, so the grid never has to scale
  // with the element count.
  constexpr int kMaxGridStrideBlocks = 4096;

  inline int grid_stride_blocks(size_t count, int block_threads)
  {
    const size_t wanted = (count + block_threads - 1) / block_threads;
    return static_cast<int>(
        wanted < kMaxGridStrideBlocks ? wanted : kMaxGridStrideBlocks);
  }

} // namespace microinfer
