#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "microinfer/kernels.h"

namespace py = pybind11;

namespace
{

  // forcecast lets the caller pass any float dtype; c_style guarantees the
  // contiguous layout the kernel indexes with.
  using FloatArray =
      py::array_t<float, py::array::c_style | py::array::forcecast>;

  py::array_t<float> rmsnorm(FloatArray x, FloatArray weight, float eps)
  {
    if (x.ndim() != 2)
    {
      throw std::invalid_argument("x must be 2-D (rows, hidden), got ndim=" +
                                  std::to_string(x.ndim()));
    }
    if (weight.ndim() != 1)
    {
      throw std::invalid_argument("weight must be 1-D (hidden,), got ndim=" +
                                  std::to_string(weight.ndim()));
    }

    const int rows = static_cast<int>(x.shape(0));
    const int hidden = static_cast<int>(x.shape(1));

    if (static_cast<int>(weight.shape(0)) != hidden)
    {
      throw std::invalid_argument(
          "weight length " + std::to_string(weight.shape(0)) +
          " does not match hidden size " + std::to_string(hidden));
    }

    py::array_t<float> out({x.shape(0), x.shape(1)});

    const float *x_ptr = x.data();
    const float *w_ptr = weight.data();
    float *out_ptr = out.mutable_data();

    {
      // The kernel does not touch Python objects, so other threads may run.
      py::gil_scoped_release release;
      microinfer::rmsnorm(x_ptr, w_ptr, out_ptr, rows, hidden, eps);
    }

    return out;
  }

} // namespace

namespace
{

  py::array_t<float> tensor_to_numpy(const microinfer::DeviceTensor &t)
  {
    py::array_t<float> out(static_cast<py::ssize_t>(t.numel()));
    float *ptr = out.mutable_data();
    {
      py::gil_scoped_release release;
      t.download(ptr);
    }
    return out;
  }

  std::unique_ptr<microinfer::DeviceTensor> upload_fp16(FloatArray host)
  {
    const auto count = static_cast<size_t>(host.size());
    const float *ptr = host.data();
    py::gil_scoped_release release;
    return std::make_unique<microinfer::DeviceTensor>(ptr, count);
  }

} // namespace

PYBIND11_MODULE(_microinfer, m)
{
  m.doc() = "MicroInfer CUDA kernels (Seam B: NumPy in, NumPy out)";

  m.def("cuda_driver_version", &microinfer::cuda_driver_version,
        "Driver version from cuDriverGetVersion. Proves libcuda is linked, "
        "which ADR-0007's allocator requires.");

  m.def("device_name", &microinfer::device_name, "Name of CUDA device 0.");

  py::class_<microinfer::DeviceTensor>(m, "DeviceTensor",
                                       "Owned fp16 storage on the device.")
      .def_property_readonly("numel", &microinfer::DeviceTensor::numel)
      .def_property_readonly("nbytes", &microinfer::DeviceTensor::nbytes,
                             "Bytes occupied on the device. fp16, so two per "
                             "element — not the size of the host array it came "
                             "from.")
      .def("to_numpy", &tensor_to_numpy,
           "Copy back to the host as fp32. For tests; not a hot path.");

  m.def("upload_fp16", &upload_fp16, py::arg("host"),
        "Upload a float32 array to the device as fp16. The conversion happens "
        "on the host so the device never holds an fp32 copy.");

  m.def(
      "device_memory_info",
      []()
      {
        const auto info = microinfer::device_memory_info();
        py::dict d;
        d["free"] = info.free_bytes;
        d["total"] = info.total_bytes;
        return d;
      },
      "Free and total device memory as cudaMemGetInfo reports it — the same "
      "quantity NVML sees, and the one RQ2 turns on.");

  m.def("rmsnorm", &rmsnorm, py::arg("x"), py::arg("weight"), py::arg("eps"),
        "RMSNorm over the last axis: x / sqrt(mean(x^2) + eps) * weight.\n"
        "Accumulates in fp32, stores fp16. Shapes are parameters; nothing is "
        "hardcoded to a model.");
}
