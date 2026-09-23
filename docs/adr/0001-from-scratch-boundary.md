# From-scratch boundary: hand-written attention and KV cache, cuBLAS for dense GEMMs

The project claims a from-scratch inference engine (`MicroInfer`), but "from
scratch" has to stop somewhere. We hand-write every kernel that the research
contribution touches — attention (online softmax, causal masking, GQA), the
paged KV cache and its block table, INT8/INT4 quantize-dequantize, RoPE and
RMSNorm — and call cuBLAS for the dense projections (Q/K/V, MLP). No external
inference engine (vLLM, llama.cpp, TensorRT-LLM) is used at any point.

## Considered Options

- **A — hand-write everything, including a tiled GEMM.** Rejected. A
  hand-written GEMM is realistically 2–5x slower than cuBLAS, and every
  P50/P99 latency number in the paper would inherit that gap. A reviewer could
  not separate "overhead of the adaptive precision policy" from "slow matmul",
  which destroys the systems claim the paper is built on. It would also consume
  roughly 12–16 of the project's 18 weeks on the one component that carries no
  novelty.
- **C — use PyTorch for the forward pass, hand-write only the KV cache layer.**
  Rejected. PyTorch's caching allocator retains freed VRAM rather than
  returning it to the driver, so `nvmlDeviceGetMemoryInfo` would report
  PyTorch's reservation instead of genuine free memory — contaminating the
  exact signal RQ2 depends on. It also gives up the "engine built from
  scratch" contribution.

## Consequences

- cuBLAS is a link-time dependency of `core_kernels`. This is a BLAS, not an
  inference engine; the paper states the boundary explicitly rather than
  implying the projections were hand-written.
- A hand-written tiled GEMM is still built, but as an isolated study comparing
  naive → shared-memory-tiled → cuBLAS under `ncu`. It is a learning and
  benchmarking artifact, deliberately kept off the critical path and out of the
  inference path.

---

## Note from #11: the gap, measured

The study the Consequences promised is in `studies/gemm/`. The "2–5x" estimate
above turned out to be generous to a first hand-written GEMM. On the models'
own projections at 512 rows, a shared-memory-tiled kernel is 8–28x slower than
cuBLAS, and a naive one is 93–267x slower. In decode the tiled kernel is 5–21x
slower. DRAM bandwidth is the ceiling there, and cuBLAS reaches it. The
hand-written kernels stall well short of it: on latency at 2–7% occupancy
(naive), and on shared memory (tiled).

The causes differ by stage. The naive kernel is bound by the load/store queue,
the tiled kernel by shared memory, and cuBLAS by the tensor pipe. The
hand-written kernels use no tensor cores at all. The measurement strengthens
this decision; it does not revise it. The write-up is `studies/gemm/README.md`.
