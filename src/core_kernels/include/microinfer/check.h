#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace microinfer
{

  inline void cuda_check(cudaError_t status, const char *what)
  {
    if (status != cudaSuccess)
    {
      throw std::runtime_error(std::string(what) +
                               " failed: " + cudaGetErrorString(status));
    }
  }

  // The driver API reports CUresult, not cudaError_t. Separate on purpose: the
  // two error domains are not interchangeable and conflating them hides faults.
  inline void driver_check(CUresult status, const char *what)
  {
    if (status != CUDA_SUCCESS)
    {
      const char *name = nullptr;
      cuGetErrorName(status, &name);
      const char *desc = nullptr;
      cuGetErrorString(status, &desc);
      throw std::runtime_error(std::string(what) +
                               " failed: " + (name ? name : "unknown") + " (" +
                               (desc ? desc : "no description") + ")");
    }
  }

} // namespace microinfer
