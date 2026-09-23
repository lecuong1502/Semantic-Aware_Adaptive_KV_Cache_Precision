#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "microinfer/kernels.h"
#include "microinfer/paged_kv_cache.h"

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

  using IntArray =
      py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

  std::string shape_of(const py::array &a)
  {
    std::string s = "(";
    for (py::ssize_t i = 0; i < a.ndim(); ++i)
    {
      s += (i ? ", " : "") + std::to_string(a.shape(i));
    }
    return s + ")";
  }

  py::array_t<float> rope(FloatArray x, IntArray positions, double theta)
  {
    if (x.ndim() != 3)
    {
      throw std::invalid_argument(
          "x must be 3-D (seq, heads, head_dim), got shape " + shape_of(x));
    }
    if (positions.ndim() != 1 || positions.shape(0) != x.shape(0))
    {
      throw std::invalid_argument("positions must be 1-D with one entry per "
                                  "token: x has seq " +
                                  std::to_string(x.shape(0)) +
                                  ", positions has shape " +
                                  shape_of(positions));
    }
    if (x.shape(2) % 2 != 0)
    {
      throw std::invalid_argument(
          "head_dim must be even for rotate-half RoPE, got " +
          std::to_string(x.shape(2)));
    }
    if (!(theta > 0.0))
    {
      throw std::invalid_argument("theta must be positive, got " +
                                  std::to_string(theta));
    }
    const int32_t *pos_ptr = positions.data();
    for (py::ssize_t i = 0; i < positions.shape(0); ++i)
    {
      if (pos_ptr[i] < 0)
      {
        throw std::invalid_argument("positions must not be negative, got " +
                                    std::to_string(pos_ptr[i]) + " at index " +
                                    std::to_string(i));
      }
    }

    py::array_t<float> out({x.shape(0), x.shape(1), x.shape(2)});
    const float *x_ptr = x.data();
    float *out_ptr = out.mutable_data();
    const int seq = static_cast<int>(x.shape(0));
    const int heads = static_cast<int>(x.shape(1));
    const int head_dim = static_cast<int>(x.shape(2));
    {
      py::gil_scoped_release release;
      microinfer::rope(x_ptr, pos_ptr, out_ptr, seq, heads, head_dim, theta);
    }
    return out;
  }

  py::array_t<float> swiglu(FloatArray gate, FloatArray up)
  {
    if (gate.ndim() != 2 || up.ndim() != 2)
    {
      throw std::invalid_argument(
          "gate and up must be 2-D (rows, intermediate), got " +
          shape_of(gate) + " and " + shape_of(up));
    }
    if (gate.shape(0) != up.shape(0) || gate.shape(1) != up.shape(1))
    {
      throw std::invalid_argument("gate " + shape_of(gate) + " and up " +
                                  shape_of(up) + " must have the same shape");
    }

    py::array_t<float> out({gate.shape(0), gate.shape(1)});
    const float *g_ptr = gate.data();
    const float *u_ptr = up.data();
    float *out_ptr = out.mutable_data();
    const auto count = static_cast<size_t>(gate.size());
    {
      py::gil_scoped_release release;
      microinfer::swiglu(g_ptr, u_ptr, out_ptr, count);
    }
    return out;
  }

  py::array_t<float> linear(FloatArray x, FloatArray weight,
                            std::optional<FloatArray> bias)
  {
    if (x.ndim() != 2)
    {
      throw std::invalid_argument(
          "x must be 2-D (rows, in_features), got shape " + shape_of(x));
    }
    if (weight.ndim() != 2)
    {
      throw std::invalid_argument(
          "weight must be 2-D (out_features, in_features), got shape " +
          shape_of(weight));
    }
    if (weight.shape(1) != x.shape(1))
    {
      throw std::invalid_argument(
          "weight " + shape_of(weight) + " does not match x " + shape_of(x) +
          ": weight is (out_features, in_features) and x has in_features " +
          std::to_string(x.shape(1)));
    }
    if (x.shape(1) == 0)
    {
      throw std::invalid_argument("in_features must be at least 1");
    }
    if (bias && (bias->ndim() != 1 || bias->shape(0) != weight.shape(0)))
    {
      throw std::invalid_argument("bias must be 1-D (out_features,) with "
                                  "out_features " +
                                  std::to_string(weight.shape(0)) +
                                  ", got shape " + shape_of(*bias));
    }

    py::array_t<float> out({x.shape(0), weight.shape(0)});
    const float *x_ptr = x.data();
    const float *w_ptr = weight.data();
    const float *b_ptr = bias ? bias->data() : nullptr;
    float *out_ptr = out.mutable_data();
    const int rows = static_cast<int>(x.shape(0));
    const int in_features = static_cast<int>(x.shape(1));
    const int out_features = static_cast<int>(weight.shape(0));
    {
      py::gil_scoped_release release;
      microinfer::linear(x_ptr, w_ptr, b_ptr, out_ptr, rows, in_features,
                         out_features);
    }
    return out;
  }

  py::array_t<float> attention(FloatArray q, FloatArray k, FloatArray v)
  {
    for (const auto *a : {&q, &k, &v})
    {
      if (a->ndim() != 3)
      {
        throw std::invalid_argument(
            "q, k and v must be 3-D (tokens, heads, head_dim), got " +
            shape_of(q) + ", " + shape_of(k) + ", " + shape_of(v));
      }
    }
    if (k.shape(0) != v.shape(0) || k.shape(1) != v.shape(1) ||
        k.shape(2) != v.shape(2))
    {
      throw std::invalid_argument("k " + shape_of(k) + " and v " + shape_of(v) +
                                  " must have the same shape");
    }
    if (k.shape(1) == 0 || q.shape(1) % k.shape(1) != 0)
    {
      throw std::invalid_argument(
          "q has " + std::to_string(q.shape(1)) + " heads and k has " +
          std::to_string(k.shape(1)) +
          "; the KV head count must divide the query head count, so that "
          "every KV head serves the same number of query heads");
    }
    if (q.shape(2) != k.shape(2))
    {
      throw std::invalid_argument(
          "q has head_dim " + std::to_string(q.shape(2)) +
          " and k has head_dim " + std::to_string(k.shape(2)));
    }
    if (q.shape(0) > k.shape(0))
    {
      throw std::invalid_argument(
          "seq_q " + std::to_string(q.shape(0)) + " exceeds seq_k " +
          std::to_string(k.shape(0)) +
          ": queries align to the last keys, so there must be at least as "
          "many keys as queries");
    }

    py::array_t<float> out({q.shape(0), q.shape(1), q.shape(2)});
    const float *q_ptr = q.data();
    const float *k_ptr = k.data();
    const float *v_ptr = v.data();
    float *out_ptr = out.mutable_data();
    const int seq_q = static_cast<int>(q.shape(0));
    const int seq_k = static_cast<int>(k.shape(0));
    const int heads = static_cast<int>(q.shape(1));
    const int kv_heads = static_cast<int>(k.shape(1));
    const int head_dim = static_cast<int>(q.shape(2));
    {
      py::gil_scoped_release release;
      microinfer::attention(q_ptr, k_ptr, v_ptr, out_ptr, seq_q, seq_k, heads,
                            kv_heads, head_dim);
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

  // A page's bytes cross Seam B as any C-contiguous array whose size is the
  // page's; its dtype is the caller's business, as the page's contents are.
  void write_page(microinfer::PagedKVCache &cache, int layer, int page_index,
                  const py::array &host)
  {
    if (!(host.flags() & py::array::c_style))
    {
      throw std::invalid_argument(
          "a page is written from a C-contiguous array");
    }
    const void *ptr = host.data();
    const auto bytes = static_cast<std::size_t>(host.nbytes());
    py::gil_scoped_release release;
    cache.write({layer, page_index}, ptr, bytes);
  }

  py::array_t<uint8_t> read_page(const microinfer::PagedKVCache &cache,
                                 int layer, int page_index)
  {
    const microinfer::PageKey key{layer, page_index};
    const auto bytes = cache.page_bytes(cache.locate(key).tier);
    py::array_t<uint8_t> out(static_cast<py::ssize_t>(bytes));
    uint8_t *ptr = out.mutable_data();
    {
      py::gil_scoped_release release;
      cache.read(key, ptr, bytes);
    }
    return out;
  }

  std::vector<std::pair<int, int>>
  pages_in_slot_order(const microinfer::PagedKVCache &cache,
                      microinfer::Tier tier)
  {
    std::vector<std::pair<int, int>> out;
    for (const auto &key : cache.pages(tier))
    {
      out.emplace_back(key.layer, key.page_index);
    }
    return out;
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

  m.def("rope", &rope, py::arg("x"), py::arg("positions"), py::arg("theta"),
        "Rotary position embedding, rotate-half form, over x (seq, heads, "
        "head_dim) with one position per token.\n"
        "theta comes from config.json; the angle is formed in fp64 and the "
        "output stored fp16.");

  m.def("swiglu", &swiglu, py::arg("gate"), py::arg("up"),
        "silu(gate) * up, elementwise over two (rows, intermediate) arrays.\n"
        "Computed in fp32, stored fp16.");

  m.def("linear", &linear, py::arg("x"), py::arg("weight"),
        py::arg("bias") = py::none(),
        "x @ weight.T + bias through cublasGemmEx. weight is (out_features, "
        "in_features), as the checkpoint stores it; bias is optional.\n"
        "fp16 operands, fp32 accumulation, one fp16 rounding on store.");

  m.def("attention", &attention, py::arg("q"), py::arg("k"), py::arg("v"),
        "Causal attention with online softmax over q (seq_q, heads, head_dim) "
        "and k, v (seq_k, kv_heads, head_dim), seq_q <= seq_k;\n"
        "kv_heads must divide heads (grouped-query attention). Queries align "
        "to the last keys. No score matrix or mask is "
        "materialised.");

  py::dict tiles;
  tiles["query"] = microinfer::kAttentionTileQ;
  tiles["key"] = microinfer::kAttentionTileK;
  m.attr("attention_tiles") = tiles;

  using microinfer::PagedKVCache;
  using microinfer::PageKey;
  using microinfer::Tier;

  py::enum_<Tier>(m, "Tier", "Precision tier, ADR-0008.")
      .value("FP16", Tier::FP16)
      .value("INT8", Tier::INT8)
      .value("INT4", Tier::INT4)
      .value("INT2", Tier::INT2);

  py::register_exception<microinfer::PageNotFound>(m, "PageNotFound",
                                                   PyExc_KeyError);

  // No method returns a device address, and none may: a page moves whenever
  // its tier's tail is retracted (ADR-0007).
  py::class_<PagedKVCache>(
      m, "PagedKVCache",
      "The KV cache allocator of ADR-0007: one reserved virtual address range "
      "per tier, backed by granules that return to the driver as they empty.\n"
      "Pages are named by (layer, page_index) and packed from the low end of "
      "their tier; freeing one moves the tier's tail page into its slot.")
      .def(py::init<const std::array<std::size_t, microinfer::kTierCount> &,
                    const std::array<std::size_t, microinfer::kTierCount> &>(),
           py::arg("page_bytes"), py::arg("capacity_pages"),
           "Both are indexed by Tier: the size of one page, and the most pages "
           "the tier can hold. Reserves address space only.")
      .def(
          "allocate",
          [](PagedKVCache &c, int layer, int page_index, Tier tier)
          {
            py::gil_scoped_release release;
            c.allocate({layer, page_index}, tier);
          },
          py::arg("layer"), py::arg("page_index"), py::arg("tier"))
      .def(
          "free",
          [](PagedKVCache &c, int layer, int page_index)
          {
            py::gil_scoped_release release;
            c.free({layer, page_index});
          },
          py::arg("layer"), py::arg("page_index"))
      .def("write", &write_page, py::arg("layer"), py::arg("page_index"),
           py::arg("data"),
           "Copy a C-contiguous array of exactly the page's size into it.")
      .def("read", &read_page, py::arg("layer"), py::arg("page_index"),
           "The page's bytes, as uint8.")
      .def(
          "locate",
          [](const PagedKVCache &c, int layer, int page_index)
          {
            const auto loc = c.locate({layer, page_index});
            return std::make_pair(loc.tier, loc.slot);
          },
          py::arg("layer"), py::arg("page_index"),
          "(tier, slot) of a page, as the page table holds it now.")
      .def("__contains__", [](const PagedKVCache &c, std::pair<int, int> key)
           { return c.contains({key.first, key.second}); })
      .def("pages", &pages_in_slot_order, py::arg("tier"),
           "The tier's (layer, page_index) keys in slot order.")
      .def("page_bytes", &PagedKVCache::page_bytes, py::arg("tier"))
      .def("reserved_bytes", &PagedKVCache::reserved_bytes, py::arg("tier"))
      .def("mapped_bytes", &PagedKVCache::mapped_bytes, py::arg("tier"),
           "Device memory backing the tier now: whole granules.")
      .def_property_readonly("granule_bytes", &PagedKVCache::granule_bytes,
                             "From cuMemGetAllocationGranularity.");
}
