#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

#include "microinfer/kernels.h"

namespace py = pybind11;

namespace {

// forcecast lets the caller pass any float dtype; c_style guarantees the
// contiguous layout the kernel indexes with.
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

py::array_t<float> rmsnorm(FloatArray x, FloatArray weight, float eps) {
  if (x.ndim() != 2) {
    throw std::invalid_argument("x must be 2-D (rows, hidden), got ndim=" +
                                std::to_string(x.ndim()));
  }
  if (weight.ndim() != 1) {
    throw std::invalid_argument("weight must be 1-D (hidden,), got ndim=" +
                                std::to_string(weight.ndim()));
  }

  const int rows = static_cast<int>(x.shape(0));
  const int hidden = static_cast<int>(x.shape(1));

  if (static_cast<int>(weight.shape(0)) != hidden) {
    throw std::invalid_argument(
        "weight length " + std::to_string(weight.shape(0)) +
        " does not match hidden size " + std::to_string(hidden));
  }

  py::array_t<float> out({x.shape(0), x.shape(1)});

  const float* x_ptr = x.data();
  const float* w_ptr = weight.data();
  float* out_ptr = out.mutable_data();

  {
    // The kernel does not touch Python objects, so other threads may run.
    py::gil_scoped_release release;
    microinfer::rmsnorm(x_ptr, w_ptr, out_ptr, rows, hidden, eps);
  }

  return out;
}

}  // namespace

PYBIND11_MODULE(_microinfer, m) {
  m.doc() = "MicroInfer CUDA kernels (Seam B: NumPy in, NumPy out)";

  m.def("cuda_driver_version", &microinfer::cuda_driver_version,
        "Driver version from cuDriverGetVersion. Proves libcuda is linked, "
        "which ADR-0007's allocator requires.");

  m.def("device_name", &microinfer::device_name, "Name of CUDA device 0.");

  m.def("rmsnorm", &rmsnorm, py::arg("x"), py::arg("weight"), py::arg("eps"),
        "RMSNorm over the last axis: x / sqrt(mean(x^2) + eps) * weight.\n"
        "Accumulates in fp32, stores fp16. Shapes are parameters; nothing is "
        "hardcoded to a model.");
}
