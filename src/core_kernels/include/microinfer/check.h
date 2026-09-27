#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace microinfer
{

  // The device had no memory for an allocation (#52). Contention is what
  // this project studies, and running out is its outcome, not a fault: it
  // has a type of its own, so that a caller can record it and go on, and
  // every other failure still reads as one. The message names the
  // allocation that failed.
  class OutOfMemory : public std::runtime_error
  {
  public:
    using std::runtime_error::runtime_error;
  };

  inline void cuda_check(cudaError_t status, const char *what)
  {
    if (status == cudaErrorMemoryAllocation)
    {
      cudaGetLastError(); // clear it: running out leaves the context usable
      throw OutOfMemory(std::string(what) +
                        " failed: " + cudaGetErrorString(status));
    }
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
      const std::string message = std::string(what) +
                                  " failed: " + (name ? name : "unknown") +
                                  " (" + (desc ? desc : "no description") + ")";
      if (status == CUDA_ERROR_OUT_OF_MEMORY)
      {
        throw OutOfMemory(message);
      }
      throw std::runtime_error(message);
    }
  }

  // cuBLAS has a third error domain of its own, cublasStatus_t. Same reasoning
  // as driver_check: reported under its own name, never cast into another.
  inline void cublas_check(cublasStatus_t status, const char *what)
  {
    if (status == CUBLAS_STATUS_ALLOC_FAILED)
    {
      throw OutOfMemory(std::string(what) +
                        " failed: " + cublasGetStatusString(status));
    }
    if (status != CUBLAS_STATUS_SUCCESS)
    {
      throw std::runtime_error(std::string(what) +
                               " failed: " + cublasGetStatusString(status));
    }
  }

} // namespace microinfer
