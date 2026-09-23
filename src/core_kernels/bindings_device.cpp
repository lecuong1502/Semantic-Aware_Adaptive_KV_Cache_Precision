// The device surface: what the engine's forward pass calls from Python.
//
// Python sequences these and does no arithmetic (ADR-0002). Every operand is
// already on the device, so a call is a launch and nothing crosses the bus
// except token ids, positions and, at the end, logits.
//
// Every op is bounds-checked against its operands before launch. A kernel
// writing past an allocation corrupts whatever the driver put next to it, and
// fails, if at all, somewhere else entirely.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_fp16.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "microinfer/check.h"
#include "microinfer/device_ops.h"
#include "microinfer/kernels.h"
#include "microinfer/kv_pages.h"

namespace py = pybind11;

namespace
{

  using microinfer::DeviceIndex;
  using microinfer::DeviceTensor;

  // A run of fp16 elements inside a DeviceTensor: the whole of it, or a view.
  // It holds a reference to the tensor, so a view cannot outlive what it looks
  // into. The KV cache is written through views: each step's keys and values
  // land at an offset into one per-layer buffer.
  struct Span
  {
    py::object owner;
    __half *ptr;
    size_t elements;

    size_t count() const { return elements; }
  };

  Span whole(py::object tensor)
  {
    auto &t = tensor.cast<DeviceTensor &>();
    return Span{tensor, static_cast<__half *>(t.data()), t.numel()};
  }

  // Every operand's size is checked against what the operation will touch.
  // Span counts elements and DeviceIndex counts indices; both call it count.
  template <typename Operand>
  void need(const Operand &operand, size_t elements, const char *what)
  {
    if (operand.count() < elements)
    {
      throw std::invalid_argument(
          std::string(what) + " holds " + std::to_string(operand.count()) +
          ", and the operation needs " + std::to_string(elements));
    }
  }

  // Sizes arrive as Python ints and are multiplied as size_t, where a negative
  // one wraps to something enormous. That would fail `need`, but an offset
  // computed from it would not: it would wrap to a pointer anywhere. So every
  // row count and offset is checked for sign before it is used.
  void non_negative(int value, const char *what)
  {
    if (value < 0)
    {
      throw std::invalid_argument(std::string(what) +
                                  " must not be negative, "
                                  "got " +
                                  std::to_string(value));
    }
  }

  size_t product(int a, int b) { return static_cast<size_t>(a) * b; }
  size_t product(int a, int b, int c) { return product(a, b) * c; }

  // fp32 logits need two fp16 slots each. The engine owns the storage, as
  // a DeviceTensor it sized, rather than this binding allocating it unseen:
  // every byte the engine takes appears in its footprint (Footprint.workspace).
  size_t scratch_elements(int rows, int vocab)
  {
    const size_t logits = product(rows, vocab) * sizeof(float);
    const size_t ids = static_cast<size_t>(rows) * sizeof(int32_t);
    return (logits + ids + sizeof(__half) - 1) / sizeof(__half);
  }

  float *logits_into(const Span &x, const Span &head, const Span &scratch,
                     int first_row, int rows, int hidden, int vocab)
  {
    non_negative(first_row, "first_row");
    non_negative(rows, "rows");
    need(x, product(first_row + rows, hidden), "x");
    need(head, product(vocab, hidden), "head");
    need(scratch, scratch_elements(rows, vocab), "scratch");
    float *out = reinterpret_cast<float *>(scratch.ptr);
    microinfer::device::linear_fp32_out(x.ptr + product(first_row, hidden),
                                        head.ptr, out, rows, hidden, vocab);
    return out;
  }

  // Rows [first_row, first_row + rows) of x through the (vocab, hidden) head,
  // as fp32 logits on the host.
  py::array_t<float> logits(const Span &x, const Span &head,
                            const Span &scratch, int first_row, int rows,
                            int hidden, int vocab)
  {
    py::array_t<float> out({rows, vocab});
    if (rows == 0)
    {
      return out;
    }
    const float *dev =
        logits_into(x, head, scratch, first_row, rows, hidden, vocab);
    microinfer::cuda_check(cudaMemcpy(out.mutable_data(), dev,
                                      product(rows, vocab) * sizeof(float),
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy logits device-to-host");
    return out;
  }

  // The greedy choice for each row: logits and argmax on the device, so only
  // the chosen ids cross the bus.
  py::array_t<int32_t> greedy(const Span &x, const Span &head,
                              const Span &scratch, int first_row, int rows,
                              int hidden, int vocab)
  {
    py::array_t<int32_t> out(rows);
    if (rows == 0)
    {
      return out;
    }
    const float *dev =
        logits_into(x, head, scratch, first_row, rows, hidden, vocab);
    int32_t *ids =
        reinterpret_cast<int32_t *>(reinterpret_cast<char *>(scratch.ptr) +
                                    product(rows, vocab) * sizeof(float));
    microinfer::device::argmax_rows(dev, rows, vocab, ids);
    microinfer::cuda_check(cudaMemcpy(out.mutable_data(), ids,
                                      rows * sizeof(int32_t),
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy ids device-to-host");
    // A row with no finite logit has no argmax: the kernel reports one past
    // the vocabulary. Handing that on as a token would put an id the model
    // does not have into the output, and embed would quietly read it as zero.
    const int32_t *chosen = out.data();
    for (int r = 0; r < rows; ++r)
    {
      if (chosen[r] >= vocab)
      {
        throw std::runtime_error(
            "row " + std::to_string(first_row + r) +
            " has no finite logit, so there is no greedy choice; the forward "
            "pass produced NaN or -inf throughout");
      }
    }
    return out;
  }

} // namespace

void bind_device(py::module_ &parent)
{
  auto m = parent.def_submodule(
      "device", "Kernels on device memory: the engine's forward pass. Python "
                "sequences these calls and does no arithmetic (ADR-0002).");

  py::class_<Span>(m, "Span", "A run of fp16 elements in a DeviceTensor.")
      .def(py::init(&whole), py::arg("tensor"))
      .def_property_readonly("count", &Span::count);
  py::implicitly_convertible<DeviceTensor, Span>();

  m.def(
      "view",
      [](py::object tensor, size_t offset, size_t count)
      {
        Span s = whole(tensor);
        if (offset > s.count() || count > s.count() - offset)
        {
          throw std::invalid_argument("view [" + std::to_string(offset) + ", " +
                                      std::to_string(offset + count) +
                                      ") of a tensor of " +
                                      std::to_string(s.count()) + " elements");
        }
        return Span{tensor, s.ptr + offset, count};
      },
      py::arg("tensor"), py::arg("offset"), py::arg("count"),
      "Elements [offset, offset + count) of a DeviceTensor.");

  m.def(
      "empty",
      [](size_t count) { return std::make_unique<DeviceTensor>(count); },
      py::arg("count"), "Uninitialised fp16 storage on the device.");

  py::class_<DeviceIndex>(m, "DeviceIndex", "int32 on the device.")
      .def_property_readonly("count", &DeviceIndex::count);
  m.def(
      "index",
      [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> host)
      {
        return std::make_unique<DeviceIndex>(host.data(),
                                             static_cast<size_t>(host.size()));
      },
      py::arg("host"), "Upload int32 token ids or positions.");

  m.def(
      "embed",
      [](const DeviceIndex &ids, const Span &table, const Span &out, int hidden,
         int vocab)
      {
        const int count = static_cast<int>(ids.count());
        need(table, product(vocab, hidden), "table");
        need(out, product(count, hidden), "out");
        microinfer::device::embed(ids.data(), table.ptr, out.ptr, count, hidden,
                                  vocab);
      },
      py::arg("ids"), py::arg("table"), py::arg("out"), py::arg("hidden"),
      py::arg("vocab"));

  m.def(
      "rmsnorm",
      [](const Span &x, const Span &weight, const Span &out, int rows,
         int hidden, float eps)
      {
        need(x, product(rows, hidden), "x");
        need(weight, hidden, "weight");
        need(out, product(rows, hidden), "out");
        microinfer::device::rmsnorm(x.ptr, weight.ptr, out.ptr, rows, hidden,
                                    eps);
      },
      py::arg("x"), py::arg("weight"), py::arg("out"), py::arg("rows"),
      py::arg("hidden"), py::arg("eps"));

  m.def(
      "linear",
      [](const Span &x, const Span &weight, std::optional<Span> bias,
         const Span &out, int rows, int in_features, int out_features)
      {
        need(x, product(rows, in_features), "x");
        need(weight, product(out_features, in_features), "weight");
        if (bias)
        {
          need(*bias, out_features, "bias");
        }
        need(out, product(rows, out_features), "out");
        microinfer::device::linear(x.ptr, weight.ptr,
                                   bias ? bias->ptr : nullptr, out.ptr, rows,
                                   in_features, out_features);
      },
      py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
      py::arg("rows"), py::arg("in_features"), py::arg("out_features"));

  m.def(
      "rope",
      [](const Span &x, const DeviceIndex &positions, const Span &out, int seq,
         int heads, int head_dim, double theta)
      {
        need(x, product(seq, heads, head_dim), "x");
        need(positions, seq, "positions");
        need(out, product(seq, heads, head_dim), "out");
        if (x.ptr == out.ptr)
        {
          // The kernel's operands are __restrict__.
          throw std::invalid_argument("rope cannot write over its input");
        }
        microinfer::device::rope(x.ptr, positions.data(), out.ptr, seq, heads,
                                 head_dim, theta);
      },
      py::arg("x"), py::arg("positions"), py::arg("out"), py::arg("seq"),
      py::arg("heads"), py::arg("head_dim"), py::arg("theta"));

  m.def(
      "attention",
      [](const Span &q, const Span &k, const Span &v,
         std::optional<Span> k_bias, const Span &out, int seq_q, int seq_k,
         int heads, int kv_heads, int head_dim, double theta)
      {
        if (seq_q > seq_k)
        {
          throw std::invalid_argument("seq_q " + std::to_string(seq_q) +
                                      " exceeds seq_k " +
                                      std::to_string(seq_k));
        }
        need(q, product(seq_q, heads, head_dim), "q");
        need(k, product(seq_k, kv_heads, head_dim), "k");
        need(v, product(seq_k, kv_heads, head_dim), "v");
        if (k_bias)
        {
          need(*k_bias, product(kv_heads, head_dim), "k_bias");
        }
        need(out, product(seq_q, heads, head_dim), "out");
        microinfer::device::attention(
            q.ptr, k.ptr, v.ptr, k_bias ? k_bias->ptr : nullptr, out.ptr, seq_q,
            seq_k, heads, kv_heads, head_dim, theta);
      },
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("k_bias"),
      py::arg("out"), py::arg("seq_q"), py::arg("seq_k"), py::arg("heads"),
      py::arg("kv_heads"), py::arg("head_dim"), py::arg("theta"),
      "Keys may be stored without their bias, which k_bias then completes "
      "(ADR-0009).");

  m.def(
      "add",
      [](const Span &a, const Span &b, const Span &out, size_t count)
      {
        need(a, count, "a");
        need(b, count, "b");
        need(out, count, "out");
        microinfer::device::add(a.ptr, b.ptr, out.ptr, count);
      },
      py::arg("a"), py::arg("b"), py::arg("out"), py::arg("count"),
      "out = a + b; out may be a or b.");

  m.def(
      "swiglu",
      [](const Span &gate, const Span &up, const Span &out, size_t count)
      {
        need(gate, count, "gate");
        need(up, count, "up");
        need(out, count, "out");
        if (out.ptr == gate.ptr || out.ptr == up.ptr)
        {
          // The kernel's operands are __restrict__.
          throw std::invalid_argument("swiglu cannot write over its input");
        }
        microinfer::device::swiglu(gate.ptr, up.ptr, out.ptr, count);
      },
      py::arg("gate"), py::arg("up"), py::arg("out"), py::arg("count"));

  m.def(
      "copy",
      [](const Span &source, const Span &target, size_t count)
      {
        need(source, count, "source");
        need(target, count, "target");
        microinfer::cuda_check(cudaMemcpy(target.ptr, source.ptr,
                                          count * sizeof(__half),
                                          cudaMemcpyDeviceToDevice),
                               "cudaMemcpy device-to-device");
      },
      py::arg("source"), py::arg("target"), py::arg("count"),
      "Copy count fp16 elements, device to device.");

  m.attr("page_tokens") = microinfer::kPageTokens;

  using microinfer::KVPages;
  py::class_<KVPages>(
      m, "KVPages",
      "The engine's KV cache on pages (#14): page i of layer l holds positions "
      "[i*P, (i+1)*P), keys then values. Pages come from a PagedKVCache as the "
      "sequence grows, and every launch resolves the page table afresh, so "
      "the allocator may move pages between launches (ADR-0007).")
      .def(py::init<microinfer::PagedKVCache &, int, int, size_t,
                    microinfer::Tier>(),
           py::arg("allocator"), py::arg("layers"), py::arg("page_tokens"),
           py::arg("row"), py::arg("tier"), py::keep_alive<1, 2>(),
           "row is kv_heads * head_dim. The allocator's page size at `tier` "
           "must be 2 * page_tokens * row fp16 elements.")
      .def("reserve", &KVPages::reserve, py::arg("tokens"),
           "Pages for positions [0, tokens) in every layer, allocating only "
           "those not yet held.")
      .def(
          "store",
          [](KVPages &c, int layer, const Span &keys, const Span &values,
             int start, int n)
          {
            non_negative(n, "n");
            const size_t row =
                c.page_bytes() / (2 * c.page_tokens() * sizeof(__half));
            need(keys, n * row, "keys");
            need(values, n * row, "values");
            c.store(layer, keys.ptr, values.ptr, start, n);
          },
          py::arg("layer"), py::arg("keys"), py::arg("values"),
          py::arg("start"), py::arg("n"))
      .def(
          "attention",
          [](KVPages &c, int layer, const Span &q, std::optional<Span> k_bias,
             const Span &out, int seq_q, int seq_k, int heads, int kv_heads,
             int head_dim, double theta)
          {
            if (seq_q > seq_k)
            {
              throw std::invalid_argument("seq_q " + std::to_string(seq_q) +
                                          " exceeds seq_k " +
                                          std::to_string(seq_k));
            }
            non_negative(seq_q, "seq_q");
            need(q, product(seq_q, heads, head_dim), "q");
            need(out, product(seq_q, heads, head_dim), "out");
            if (k_bias)
            {
              need(*k_bias, product(kv_heads, head_dim), "k_bias");
            }
            c.attention(layer, q.ptr, k_bias ? k_bias->ptr : nullptr, out.ptr,
                        seq_q, seq_k, heads, kv_heads, head_dim, theta);
          },
          py::arg("layer"), py::arg("q"), py::arg("k_bias"), py::arg("out"),
          py::arg("seq_q"), py::arg("seq_k"), py::arg("heads"),
          py::arg("kv_heads"), py::arg("head_dim"), py::arg("theta"))
      .def_property_readonly("page_tokens", &KVPages::page_tokens)
      .def_property_readonly("pages_per_layer", &KVPages::pages_per_layer)
      .def_property_readonly("capacity_tokens", &KVPages::capacity_tokens)
      .def_property_readonly("page_bytes", &KVPages::page_bytes);

  m.def("scratch_elements", &scratch_elements, py::arg("rows"),
        py::arg("vocab"),
        "fp16 elements of scratch that logits or greedy need for `rows` rows.");

  m.def("logits", &logits, py::arg("x"), py::arg("head"), py::arg("scratch"),
        py::arg("first_row"), py::arg("rows"), py::arg("hidden"),
        py::arg("vocab"),
        "fp32 logits for rows [first_row, first_row + rows) of x, to the "
        "host.");

  m.def("greedy", &greedy, py::arg("x"), py::arg("head"), py::arg("scratch"),
        py::arg("first_row"), py::arg("rows"), py::arg("hidden"),
        py::arg("vocab"),
        "The argmax token of each row's logits, computed on the device.");
}
