#pragma once

#include <cstddef>
#include <memory>

#include "microinfer/device_buffer.h"

namespace microinfer
{

  // cos and sin of every RoPE angle, per position, in fp32 on the device
  // (ADR-0009, note from #14).
  //
  // Attention adds each key's bias back rotated to the key's position. Forming
  // that rotation's angle and its cos and sin in fp64 for every element of
  // every key tile, as the first version did, made attention 3.3 times slower
  // in prefill and 4 times slower in decode. The angle depends only on the
  // position and the frequency: not on the layer, the head or the key. So it
  // is formed once per position here, still in fp64 (rope_angle.cuh), and
  // every layer's attention reads it.
  //
  // Entry (p, j) is {cos, sin} of p * theta^(-2j / head_dim), for
  // j < head_dim / 2. The table grows with the sequence, doubling so that a
  // decode step does not rebuild it every time.
  class RopeTable
  {
  public:
    RopeTable(int head_dim, double theta);

    // Makes positions [0, count) present, growing the table if needed.
    void cover(int count);

    const float *data() const;
    int positions() const { return positions_; }
    int head_dim() const { return head_dim_; }
    double theta() const { return theta_; }
    std::size_t nbytes() const;

  private:
    int head_dim_;
    double theta_;
    int positions_ = 0;
    std::unique_ptr<DeviceBuffer> table_;
  };

} // namespace microinfer
