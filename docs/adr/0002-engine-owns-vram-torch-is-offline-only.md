# The engine owns all VRAM; PyTorch is an offline dev dependency only

`MicroInfer` is a Python orchestrator over a pybind11 CUDA extension, and it
allocates every byte of device memory itself through `cudaMalloc`. PyTorch is
never imported in a process that runs the engine. It appears in exactly one
place — `tools/gen_golden.py`, run offline to produce reference activations
from a HuggingFace model at a fixed prompt and seed, written to `.npz` files
that the test suite compares against.

## Considered Options

- **torch as the in-process tensor container** (the conventional pybind11 CUDA
  extension pattern: hold weights and activations in `torch.Tensor`, pass
  `.data_ptr()` into hand-written kernels). Rejected. PyTorch's caching
  allocator retains freed device memory rather than returning it to the driver,
  so `nvmlDeviceGetMemoryInfo` would report PyTorch's reservation as used
  memory, indistinguishable from genuine external contention. RQ2's entire
  signal is that NVML reading. Correcting for it would mean polling
  `torch.cuda.memory_reserved()` and subtracting — a correction term that would
  have to be defended in the paper's method section, on a measurement the paper
  is built on.
- **A standalone C++/CUDA binary** with Python confined to the eval harness
  across a process boundary. Cleanest measurement of all, but requires
  hand-writing a tokenizer, a safetensors loader and sampling in C++ — roughly
  three weeks of plumbing carrying no novelty.

## Consequences

- Weights are copied to device with `cudaMemcpy`, and no torch tensor ever
  exists in the engine process. **The `safetensors` NumPy backend is not used —
  see the amendment below.**
- Golden reference tensors are generated once per (model, prompt set) pair and
  stored; regenerating them requires a machine with torch and the HF model, and
  is a deliberate, infrequent act.
- Python performs no arithmetic. It sequences kernel calls; every tensor
  operation happens in CUDA.

---

## Amendment: the safetensors container is parsed directly, not through its NumPy backend

This ADR said weights would load through "the `safetensors` numpy backend". They
do not, and #5 found out why: **Qwen2.5 ships bfloat16, and NumPy has no
bfloat16**. The NumPy backend cannot return a bf16 tensor for the same reason —
there is no array type to put it in. Taking the dependency would have bought a
loader that cannot load this project's only checkpoints.

`microinfer/weights.py` therefore reads the container itself. It is a small,
well-specified format: an 8-byte little-endian length, a JSON header mapping
each tensor to a dtype, a shape and a byte range, and the data. The decode from
bf16 is a shift, not an approximation — bfloat16 *is* the top sixteen bits of a
float32, so widening is exact.

Recorded rather than quietly done, because CONTRIBUTING requires a change that
contradicts an ADR to say so, and the original wording would otherwise read as
a description of code that does not exist.

### Consequences

- `safetensors` is not a dependency and should not become one for this purpose.
- The reader owns two checks the library would have provided: an unknown dtype
  is an error rather than a guess, and a truncated file is named as truncated
  rather than surfacing as a reshape error deep in the parser.
