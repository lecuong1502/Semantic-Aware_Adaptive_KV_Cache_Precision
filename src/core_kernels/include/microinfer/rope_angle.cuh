#pragma once

// The RoPE rotation angle, shared by the kernels that rotate: rope, which
// rotates queries and keys, and attention, which rotates the key bias it adds
// back (ADR-0009). Two copies of this would drift, and a drifted angle rotates
// the bias by something other than the key it completes.
//
// Rotate-half pairing, as Qwen2 uses it: dimension j rotates with j + half, by
// position * theta^(-2j / head_dim). The angle is formed in fp64. An fp32 angle
// is 18 ulp out at position 2049 and 60 ulp at 32767 (tests/test_rope.py).

namespace microinfer
{

  __device__ inline void rope_sincos(double position, int j, int head_dim,
                                     double theta, double *s, double *c)
  {
    const double inv_freq =
        pow(theta, -2.0 * static_cast<double>(j) / head_dim);
    sincos(position * inv_freq, s, c);
  }

} // namespace microinfer
