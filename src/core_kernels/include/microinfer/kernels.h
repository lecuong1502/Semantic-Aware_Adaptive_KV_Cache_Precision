#pragma once

#include <string>

namespace microinfer
{

  // Driver API (libcuda), deliberately not the runtime API. Proving this
  // linkage is the point of the toolchain tracer bullet; ADR-0007's allocator
  // depends on it. Returns the driver version reported by cuDriverGetVersion.
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
  void rmsnorm(const float *x, const float *weight, float *out, int rows,
               int hidden, float eps);

  // Device memory as the *driver* sees it, from cudaMemGetInfo. This is the
  // same quantity NVML reports and the one RQ2 turns on, so the engine reads it
  // rather than tracking its own allocations and trusting the two to agree.
  struct MemoryInfo
  {
    size_t free_bytes;
    size_t total_bytes;
  };

  MemoryInfo device_memory_info();

  // An owned block of fp16 on the device. Weights arrive as fp32 from the host
  // and are stored fp16, which is what every kernel reads.
  class DeviceTensor
  {
  public:
    DeviceTensor(const float *host, size_t count);
    ~DeviceTensor();
    DeviceTensor(const DeviceTensor &) = delete;
    DeviceTensor &operator=(const DeviceTensor &) = delete;

    size_t numel() const { return count_; }
    // Out of line so it can say sizeof(__half) rather than a literal 2. This
    // is the number Footprint.weights reports; it should not be a guess.
    size_t nbytes() const;
    void download(float *out) const;
    const void *data() const { return ptr_; }

  private:
    void *ptr_ = nullptr;
    size_t count_ = 0;
  };

} // namespace microinfer
