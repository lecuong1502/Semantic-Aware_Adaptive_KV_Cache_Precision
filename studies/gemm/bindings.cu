// NumPy in, NumPy out, as at the engine's Seam B, so the study's kernels are
// tested exactly as the engine's projection is (tests/test_gemm_study.py).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

#include "gemm.cuh"
#include "microinfer/device_buffer.h"
#include "microinfer/staging.h"

namespace py = pybind11;

namespace
{

  using FloatArray =
      py::array_t<float, py::array::c_style | py::array::forcecast>;
  using gemm_study::Gemm;

  // Fp32 host arrays are rounded to fp16 on the way in and widened on the way
  // out, with the same staging helpers the engine uses.
  py::array_t<float> run(Gemm gemm, FloatArray x, FloatArray w)
  {
    if (x.ndim() != 2 || w.ndim() != 2)
    {
      throw std::invalid_argument(
          "x must be (rows, in_features) and w (out_features, in_features)");
    }
    if (x.shape(1) != w.shape(1))
    {
      throw std::invalid_argument("x has in_features " +
                                  std::to_string(x.shape(1)) + " and w has " +
                                  std::to_string(w.shape(1)));
    }
    const int rows = static_cast<int>(x.shape(0));
    const int in_features = static_cast<int>(x.shape(1));
    const int out_features = static_cast<int>(w.shape(0));
    py::array_t<float> y({x.shape(0), w.shape(0)});
    const size_t x_count = static_cast<size_t>(rows) * in_features;
    const size_t w_count = static_cast<size_t>(out_features) * in_features;
    const size_t y_count = static_cast<size_t>(rows) * out_features;
    if (y_count == 0)
    {
      return y;
    }
    const float *x_ptr = x.data();
    const float *w_ptr = w.data();
    float *y_ptr = y.mutable_data();
    {
      py::gil_scoped_release release;
      microinfer::DeviceBuffer dev_x(x_count * sizeof(__half));
      microinfer::DeviceBuffer dev_w(w_count * sizeof(__half));
      microinfer::DeviceBuffer dev_y(y_count * sizeof(__half));
      microinfer::upload_fp16(dev_x, x_ptr, x_count, "upload x");
      microinfer::upload_fp16(dev_w, w_ptr, w_count, "upload w");
      gemm(dev_x.as<const __half>(), dev_w.as<const __half>(),
           dev_y.as<__half>(), rows, in_features, out_features);
      microinfer::cuda_check(cudaDeviceSynchronize(), "GEMM execution");
      microinfer::download_fp16(y_ptr, dev_y, y_count, "download y");
    }
    return y;
  }

} // namespace

PYBIND11_MODULE(_gemm_study, m)
{
  m.doc() = "GEMM study (#11): hand-written kernels against cuBLAS. Off the "
            "inference path (ADR-0001).";
  m.def(
      "naive", [](FloatArray x, FloatArray w)
      { return run(gemm_study::naive, x, w); }, py::arg("x"), py::arg("w"),
      "y = x @ w.T, one thread per output reading global memory directly.");
  m.def(
      "tiled",
      [](FloatArray x, FloatArray w) { return run(gemm_study::tiled, x, w); },
      py::arg("x"), py::arg("w"), "y = x @ w.T, through shared-memory tiles.");
  m.def(
      "cublas", [](FloatArray x, FloatArray w)
      { return run(gemm_study::cublas, x, w); }, py::arg("x"), py::arg("w"),
      "y = x @ w.T through cublasGemmEx, configured as the engine's "
      "projection.");
  m.attr("tile") = gemm_study::kTile;
}
