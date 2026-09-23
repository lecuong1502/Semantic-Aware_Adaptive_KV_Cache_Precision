#pragma once

#include <cstdint>
#include <string>

namespace microinfer
{

  // Driver API (libcuda), deliberately not the runtime API. Proving this
  // linkage is the point of the toolchain tracer bullet; ADR-0007's allocator
  // depends on it. Returns the driver version reported by cuDriverGetVersion.
  int cuda_driver_version();

  std::string device_name();

  // RMSNorm over the last axis: out = x / sqrt(mean(x^2) + eps) * weight.
  //
  // Host arrays are fp32; the kernel stores fp16 and accumulates in fp32, which
  // is the arrangement the engine uses throughout. Host-to-device and
  // device-to-host transfers happen inside, so the test-facing surface is plain
  // NumPy (Seam B, issue #1).
  //
  // `hidden` and `rows` are parameters. Nothing is hardcoded to any model:
  // Qwen2.5-0.5B has hidden 896 and Qwen2.5-1.5B has 1536 (ADR-0003).
  void rmsnorm(const float *x, const float *weight, float *out, int rows,
               int hidden, float eps);

  // Rotary position embedding, Qwen2's "rotate half" form: dimension j rotates
  // with j + head_dim/2 by pos * theta^(-2j/head_dim). x is (seq, heads,
  // head_dim) and positions is (seq,), one per token, so decode can rotate a
  // single token at any position. Applied to queries and keys separately,
  // since they differ in head count.
  //
  // theta is a parameter because it is config: Qwen2.5 uses 1e6 with no
  // scaling, which is why ADR-0003 chose it over Llama-3.2.
  void rope(const float *x, const int32_t *positions, float *out, int seq,
            int heads, int head_dim, double theta);

  // The MLP activation, silu(gate) * up, elementwise over `count` values.
  void swiglu(const float *gate, const float *up, float *out, size_t count);

  // A dense projection, out = x W^T + bias, through cublasGemmEx: fp16
  // operands, fp32 accumulation, one fp16 rounding at the end. ADR-0001:
  // cuBLAS is a BLAS, not an inference engine, and this wrapper is where the
  // boundary sits. x is (rows, in_features); weight is (out_features,
  // in_features), the layout the checkpoint stores; bias may be null.
  void linear(const float *x, const float *weight, const float *bias,
              float *out, int rows, int in_features, int out_features);

  // Attention tiles: queries per thread block, and keys per step of the online
  // softmax. Public so that tests can place sequence lengths relative to them
  // (a length below one tile, one that is not a multiple of it) instead of
  // guessing.
  constexpr int kAttentionTileQ = 16;
  constexpr int kAttentionTileK = 32;

  // Causal scaled dot-product attention with online softmax: out =
  // softmax(q k^T / sqrt(head_dim), causal) v, per head, never materialising
  // the score matrix or a mask. q and out are (seq_q, heads, head_dim); k and v
  // are (seq_k, kv_heads, head_dim), with seq_q <= seq_k.
  //
  // Causal alignment is bottom-right: query i is at position seq_k - seq_q + i
  // and sees keys at positions up to it. Prefill (seq_q == seq_k) and decode
  // (seq_q == 1) are the same rule.
  //
  // Grouped-query attention: kv_heads divides heads, and query head h reads KV
  // head h / (heads / kv_heads), HuggingFace's repeat_kv order. kv_heads ==
  // heads is ordinary multi-head attention. A head count that does not group,
  // or zero heads on one side only, throws std::invalid_argument.
  //
  // k_bias, if non-null, is (kv_heads, head_dim): the keys were stored
  // without their projection's bias, and key row j is completed as
  // k + RoPE(k_bias, j) with theta before use (ADR-0009).
  void attention(const float *q, const float *k, const float *v,
                 const float *k_bias, float *out, int seq_q, int seq_k,
                 int heads, int kv_heads, int head_dim, double theta);

  // Device memory as the *driver* sees it, from cudaMemGetInfo. This is the
  // same quantity NVML reports and the one RQ2 turns on, so the engine reads it
  // rather than tracking its own allocations and trusting the two to agree.
  struct MemoryInfo
  {
    size_t free_bytes;
    size_t total_bytes;
  };

  MemoryInfo device_memory_info();

  // An owned block of fp16 on the device. Weights arrive as fp32 from the host
  // and are stored fp16, which is what every kernel reads.
  class DeviceTensor
  {
  public:
    DeviceTensor(const float *host, size_t count);
    // Uninitialised storage: activations and the KV cache, which the forward
    // pass writes before it reads.
    explicit DeviceTensor(size_t count);
    ~DeviceTensor();
    DeviceTensor(const DeviceTensor &) = delete;
    DeviceTensor &operator=(const DeviceTensor &) = delete;

    size_t numel() const { return count_; }
    // Out of line so it can say sizeof(__half) rather than a literal 2. This
    // is the number Footprint.weights reports; it should not be a guess.
    size_t nbytes() const;
    void download(float *out) const;
    const void *data() const { return ptr_; }
    void *data() { return ptr_; }

  private:
    void *ptr_ = nullptr;
    size_t count_ = 0;
  };

  // int32 on the device, uploaded from the host: token ids and positions, which
  // the forward pass needs on the device and which are never fp16.
  class DeviceIndex
  {
  public:
    DeviceIndex(const int32_t *host, size_t count);
    ~DeviceIndex();
    DeviceIndex(const DeviceIndex &) = delete;
    DeviceIndex &operator=(const DeviceIndex &) = delete;

    size_t count() const { return count_; }
    const int32_t *data() const { return ptr_; }

  private:
    int32_t *ptr_ = nullptr;
    size_t count_ = 0;
  };

} // namespace microinfer
