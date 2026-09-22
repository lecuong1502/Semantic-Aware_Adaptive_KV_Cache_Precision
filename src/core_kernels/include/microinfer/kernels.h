#pragma once

#include <string>

namespace microinfer {

// Driver API (libcuda), deliberately not the runtime API. Proving this linkage
// is the point of the toolchain tracer bullet; ADR-0007's allocator depends on
// it. Returns the driver version reported by cuDriverGetVersion.
int cuda_driver_version();

std::string device_name();

// RMSNorm over the last axis: out = x / sqrt(mean(x^2) + eps) * weight.
//
// Host arrays are fp32; the kernel stores fp16 and accumulates in fp32, which
// is the arrangement the engine uses throughout. Host-to-device and
// device-to-host transfers happen inside, so the test-facing surface is plain
// NumPy (Seam B, issue #1).
//
// `hidden` and `rows` are parameters. Nothing is hardcoded to any model:
// Qwen2.5-0.5B has hidden 896 and Qwen2.5-1.5B has 1536 (ADR-0003).
void rmsnorm(const float* x, const float* weight, float* out, int rows,
             int hidden, float eps);

}  // namespace microinfer
