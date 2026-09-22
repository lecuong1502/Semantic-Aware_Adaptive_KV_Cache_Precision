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

- Weights are loaded with the `safetensors` numpy backend and copied to device
  with `cudaMemcpy`. No torch tensor ever exists in the engine process.
- Golden reference tensors are generated once per (model, prompt set) pair and
  stored; regenerating them requires a machine with torch and the HF model, and
  is a deliberate, infrequent act.
- Python performs no arithmetic. It sequences kernel calls; every tensor
  operation happens in CUDA.
