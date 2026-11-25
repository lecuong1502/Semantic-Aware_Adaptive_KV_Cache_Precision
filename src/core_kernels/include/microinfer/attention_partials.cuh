#pragma once

// A split launch's partials (#204), which every attention kernel that splits
// its keys leaves and combine_kernel folds. Device code.

#include <cstddef>

namespace microinfer
{
  namespace
  {

    // What one slot of a split launch's partials holds (see `launch`): the
    // row's m, its l, the mass its first page carries over from the split
    // before (decode only), and its unnormalised output.
    constexpr int kPartialM = 0;
    constexpr int kPartialL = 1;
    constexpr int kPartialCarry = 2;
    constexpr int kPartialOut = 3;

    __host__ __device__ inline size_t partial_slot_floats(int head_dim)
    {
      return static_cast<size_t>(head_dim) + kPartialOut;
    }

  } // namespace
} // namespace microinfer
