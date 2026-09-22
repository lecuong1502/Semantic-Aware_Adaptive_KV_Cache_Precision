#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "microinfer/check.h"

namespace microinfer {

// RAII device allocation. Lives in a header because every kernel needs it and
// the alternative is each one growing its own copy.
//
// This is cudaMalloc, deliberately: it is for scratch and for staging, not for
// the KV cache. The cache uses the virtual memory management API so that freed
// pages return to the driver where NVML can see them (ADR-0007), which this
// class cannot do and must not be extended to pretend to do.
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) {
    cuda_check(cudaMalloc(&ptr_, bytes), "cudaMalloc");
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  template <typename T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }

  void* raw() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

}  // namespace microinfer
