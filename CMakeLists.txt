cmake_minimum_required(VERSION 3.24)

# Detected from the GPU present at build time rather than hardcoded. Override
# for a different target, e.g. -DCMAKE_CUDA_ARCHITECTURES=89 for Ada.
#
# This must come before project(). Enabling the CUDA language initialises
# CMAKE_CUDA_ARCHITECTURES to nvcc's own default, sm_52, so a check placed
# after project() always finds it defined and never applies `native`. That
# happened: every build until this one was Maxwell code that the driver
# JIT-compiled from PTX on first load.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(microinfer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# For the editor, not the build: clangd and the VS Code C/C++ extension read the
# exact nvcc include paths from this, so CUDA headers resolve in the IDE.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(_microinfer
  src/core_kernels/argmax.cu
  src/core_kernels/attention.cu
  src/core_kernels/bindings.cpp
  src/core_kernels/bindings_device.cpp
  src/core_kernels/device.cu
  src/core_kernels/device_tensor.cu
  src/core_kernels/embed.cu
  src/core_kernels/kv_pages.cu
  src/core_kernels/linear.cu
  src/core_kernels/paged_kv_cache.cu
  src/core_kernels/residual.cu
  src/core_kernels/rmsnorm.cu
  src/core_kernels/rope.cu
  src/core_kernels/swiglu.cu
)

target_include_directories(_microinfer PRIVATE src/core_kernels/include)

# P, token positions per page (ADR-0004). A build-time parameter, and the only
# place it is set: Milestone 3 ablates it over 16, 32 and 64 by rebuilding with
# -DMICROINFER_PAGE_TOKENS=<P>.
set(MICROINFER_PAGE_TOKENS 32 CACHE STRING "Token positions per KV cache page")
target_compile_definitions(_microinfer PRIVATE
  MICROINFER_PAGE_TOKENS=${MICROINFER_PAGE_TOKENS})

# CUDA::cuda_driver is libcuda, the driver API. It is NOT the same as
# CUDA::cudart, and ADR-0007's allocator needs it: cuMemAddressReserve,
# cuMemCreate, cuMemMap and cuMemRelease live here. A missing link surfaces at
# link time, not compile time, so it is proven now rather than discovered later.
#
# CUDA::cublas is the one library the forward pass borrows (ADR-0001): dense
# projections only. Attention, the KV cache and every other kernel are ours.
target_link_libraries(_microinfer PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cublas)

install(TARGETS _microinfer DESTINATION microinfer)
