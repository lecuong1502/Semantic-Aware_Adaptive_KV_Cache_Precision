#pragma once

#include "microinfer/rope_scaling.h"

// The RoPE rotation angle, shared by the kernels that rotate: rope, which
// rotates queries and keys, and attention's table, which rotates the key bias
// it adds back (ADR-0009). Two copies of this would drift, and a drifted angle
// rotates the bias by something other than the key it completes.
//
// Rotate-half pairing, as Qwen2 and Llama use it: dimension j rotates with
// j + half, by position * inv_freq(j). inv_freq is theta^(-2j / head_dim),
// scaled as `scaling` says (rope_scaling.h). The angle is formed in fp64. An
// fp32 angle is 18 ulp out at position 2049 and 60 ulp at 32767
// (tests/test_rope.py).

namespace microinfer
{

  __device__ inline double rope_inv_freq(int j, int head_dim, double theta,
                                         const RopeScaling &scaling)
  {
    const double inv_freq =
        pow(theta, -2.0 * static_cast<double>(j) / head_dim);
    if (!scaling.scaled)
    {
      return inv_freq;
    }
    const double wavelength = 2.0 * 3.14159265358979323846 / inv_freq;
    const double old = scaling.original_max_position_embeddings;
    if (wavelength < old / scaling.high_freq_factor)
    {
      return inv_freq; // high frequency: kept
    }
    if (wavelength > old / scaling.low_freq_factor)
    {
      return inv_freq / scaling.factor; // low frequency: scaled
    }
    const double smooth = (old / wavelength - scaling.low_freq_factor) /
                          (scaling.high_freq_factor - scaling.low_freq_factor);
    return (1.0 - smooth) * inv_freq / scaling.factor + smooth * inv_freq;
  }

  __device__ inline void rope_sincos(double position, int j, int head_dim,
                                     double theta, const RopeScaling &scaling,
                                     double *s, double *c)
  {
    sincos(position * rope_inv_freq(j, head_dim, theta, scaling), s, c);
  }

} // namespace microinfer
