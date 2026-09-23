# The GEMM study (#11), built on its own and never by the engine's CMakeLists
# (ADR-0001). It reuses the engine's headers for error checking and staging;
# the dependency runs from the study to the engine and never back.
#
#   cmake -S studies/gemm -B build/studies-gemm \
#     -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
#   cmake --build build/studies-gemm
#
# tests/test_gemm_study.py runs exactly this on first use.

cmake_minimum_required(VERSION 3.24)

# As in the engine's CMakeLists: must precede project(), or CUDA's default
# architecture shadows `native`.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(gemm_study LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# ncu attributes stalls to source lines only when line information is present.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(ENGINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/core_kernels/include)

pybind11_add_module(_gemm_study bindings.cu)
target_include_directories(_gemm_study PRIVATE ${ENGINE_INCLUDE})
target_link_libraries(_gemm_study PRIVATE CUDA::cudart CUDA::cublas)

add_executable(gemm_bench bench.cu)
target_include_directories(gemm_bench PRIVATE ${ENGINE_INCLUDE})
target_link_libraries(gemm_bench PRIVATE CUDA::cudart CUDA::cublas CUDA::nvtx3)
