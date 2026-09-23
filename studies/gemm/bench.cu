// The benchmark driver: every implementation on every shape given, on the same
// device buffers, one after another.
//
//   gemm_bench [--repeat N] ROWS,IN,OUT [ROWS,IN,OUT ...]
//
// It prints one CSV line per (shape, implementation) with the median of N
// timed launches. The timed launches, and only they, sit between
// cudaProfilerStart and cudaProfilerStop, so `ncu --profile-from-start off`
// profiles no warm-up. Each is also inside an NVTX range named
// "ROWSxINxOUT/implementation", for a reader of the report in the ncu UI.
//
// Timing is by CUDA events around each launch after warm-up, and the median is
// reported, not the mean: on a laptop GPU shared with a display, an occasional
// launch is delayed for reasons that have nothing to do with the kernel.

#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "gemm.cuh"
#include "microinfer/device_buffer.h"

namespace
{

  struct Shape
  {
    int rows, in_features, out_features;
  };

  struct Implementation
  {
    const char *name;
    void (*run)(const __half *, const __half *, __half *, int, int, int);
  };

  const Implementation kImplementations[] = {
      {"naive", gemm_study::naive},
      {"tiled", gemm_study::tiled},
      {"cublas", gemm_study::cublas},
  };

  void fill(microinfer::DeviceBuffer &buf, size_t count, float scale,
            std::mt19937 &rng)
  {
    std::normal_distribution<float> normal(0.0f, scale);
    std::vector<__half> host(count);
    for (auto &v : host)
    {
      v = __float2half(normal(rng));
    }
    gemm_study::cuda_check(cudaMemcpy(buf.raw(), host.data(),
                                      count * sizeof(__half),
                                      cudaMemcpyHostToDevice),
                           "upload");
  }

} // namespace

int main(int argc, char **argv)
{
  int repeat = 50;
  std::vector<Shape> shapes;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--repeat" && i + 1 < argc)
    {
      repeat = std::atoi(argv[++i]);
      continue;
    }
    Shape s{};
    if (std::sscanf(arg.c_str(), "%d,%d,%d", &s.rows, &s.in_features,
                    &s.out_features) != 3)
    {
      std::fprintf(stderr, "bad shape '%s', expected ROWS,IN,OUT\n",
                   arg.c_str());
      return 2;
    }
    shapes.push_back(s);
  }
  if (shapes.empty() || repeat < 1)
  {
    std::fprintf(
        stderr,
        "usage: gemm_bench [--repeat N] ROWS,IN,OUT [ROWS,IN,OUT ...]\n");
    return 2;
  }

  std::mt19937 rng(11);
  cudaEvent_t start, stop;
  gemm_study::cuda_check(cudaEventCreate(&start), "cudaEventCreate");
  gemm_study::cuda_check(cudaEventCreate(&stop), "cudaEventCreate");

  std::printf("rows,in_features,out_features,implementation,median_us\n");
  for (const Shape &s : shapes)
  {
    const size_t x_count = static_cast<size_t>(s.rows) * s.in_features;
    const size_t w_count = static_cast<size_t>(s.out_features) * s.in_features;
    microinfer::DeviceBuffer x(x_count * sizeof(__half));
    microinfer::DeviceBuffer w(w_count * sizeof(__half));
    microinfer::DeviceBuffer y(static_cast<size_t>(s.rows) * s.out_features *
                               sizeof(__half));
    fill(x, x_count, 1.0f, rng);
    fill(w, w_count, 1.0f / std::sqrt(static_cast<float>(s.in_features)), rng);

    for (const Implementation &impl : kImplementations)
    {
      const std::string range =
          std::to_string(s.rows) + "x" + std::to_string(s.in_features) + "x" +
          std::to_string(s.out_features) + "/" + impl.name;
      auto launch = [&]
      {
        impl.run(x.as<const __half>(), w.as<const __half>(), y.as<__half>(),
                 s.rows, s.in_features, s.out_features);
      };

      for (int i = 0; i < 3; ++i) // warm-up, and cuBLAS's handle and heuristics
      {
        launch();
      }
      gemm_study::cuda_check(cudaDeviceSynchronize(), "warm-up");

      std::vector<float> us(repeat);
      gemm_study::cuda_check(cudaProfilerStart(), "cudaProfilerStart");
      nvtxRangePushA(range.c_str());
      for (int i = 0; i < repeat; ++i)
      {
        gemm_study::cuda_check(cudaEventRecord(start), "cudaEventRecord");
        launch();
        gemm_study::cuda_check(cudaEventRecord(stop), "cudaEventRecord");
        gemm_study::cuda_check(cudaEventSynchronize(stop),
                               "cudaEventSynchronize");
        float ms = 0.0f;
        gemm_study::cuda_check(cudaEventElapsedTime(&ms, start, stop),
                               "cudaEventElapsedTime");
        us[i] = ms * 1000.0f;
      }
      nvtxRangePop();
      gemm_study::cuda_check(cudaProfilerStop(), "cudaProfilerStop");

      std::nth_element(us.begin(), us.begin() + repeat / 2, us.end());
      std::printf("%d,%d,%d,%s,%.2f\n", s.rows, s.in_features, s.out_features,
                  impl.name, us[repeat / 2]);
    }
  }
  return 0;
}
