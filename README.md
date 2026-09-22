# Semantic-Aware Adaptive KV Cache Precision

**Robust single-user LLM inference under unpredictable VRAM contention on consumer GPUs.**

A from-scratch inference engine — `MicroInfer` — that reallocates KV cache
precision at runtime, guided by attention-derived importance, when another
application on the same GPU takes video memory away.

> **Status: Milestone 0, in progress. There is no working engine yet.**
> The repository currently holds research notes, architectural decisions and a
> ticketed plan. See [Status](#status).

---

## The problem

Local LLM inference is increasingly ordinary — Ollama, LM Studio, llama.cpp on
laptops. Those laptops run browsers, chat apps and screen recorders at the same
time, and consumer GPUs give **no memory isolation** between processes. A
generation session that was comfortable a second ago can hit an out-of-memory
error because a browser tab started a video.

The adaptive-serving literature does not address this. FineServe, MorphServe,
MIRAGE, eLLM and KV-RM all assume a datacenter: a dedicated GPU, a scheduler
that can see the workload, and many concurrent requests to trade off against.
Locally there is one session, no scheduler to consult, and contention that
arrives without warning from processes the engine knows nothing about.

Prior KV cache compression work — KVQuant, KIVI, KVTuner — assigns precision
**statically**, offline, uniformly across the cache. This project assigns it
**at runtime, non-uniformly**, in response to memory pressure it did not cause
and cannot predict.

### Research questions

- **RQ1** — How volatile is available VRAM on a consumer laptop under realistic
  multitasking, and how does that interact with an in-progress generation?
- **RQ2** — Can a scheduler-agnostic runtime detect and react to VRAM pressure
  fast enough to avoid an OOM without losing the session?
- **RQ3** — Does allocating precision by semantic importance, rather than
  uniformly, preserve task quality better **at equal average compression**?

---

## Why the configuration is what it is

The mechanism is only meaningful when the bytes it can reclaim exceed the size
of a real contention spike. That is arithmetic, and it constrains the setup more
than model quality does.

Target machine: **RTX 4050 Laptop, 6141 MiB, compute capability 8.9**, CUDA 12.9.

Reclaimable by requantising the whole cache from FP16 to INT4, as a share of
that card:

| Model | KV / token | 8K | 16K | 32K |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 12 KiB | 1.2% | 2.3% | 4.7% |
| Qwen2.5-1.5B | 28 KiB | 2.7% | 5.5% | **10.9%** |
| Llama-3.2-1B | 32 KiB | 3.1% | 6.3% | 12.5% |

A browser tab with hardware acceleration takes 200–400 MiB; a screen share
around 200 MiB. At Qwen2.5-0.5B with an 8K context — the configuration most
people would reach for — the *entire* KV cache is 96 MiB and flattening it to
INT4 frees 72 MiB. The mechanism could not prevent a single OOM, and RQ2 would
be dead before a line of code was written.

Hence: **Qwen2.5-0.5B for development, Qwen2.5-1.5B at 32K context for
experiments** ([ADR-0003](docs/adr/0003-qwen25-family-small-for-dev-larger-for-experiments.md)).

### What a tier actually costs

Per page, for Qwen2.5-1.5B. Scale metadata is a flat 1280 bytes at every
quantised tier, so it does *not* shrink with bit width:

| Tier | Data | + metadata | Cache @ 32K | Effective bits |
|---|---:|---:|---:|---:|
| FP16 | 32768 B | — | 896 MiB | 16.00 |
| INT8 | 16384 B | 17664 B | 483 MiB | 8.63 |
| INT4 | 8192 B | 9472 B | 259 MiB | 4.63 |
| INT2 | 4096 B | 5376 B | 147 MiB | 2.63 |

Compression ratios reported anywhere in this project count metadata. INT4 is
4.63 effective bits, not 4.

---

## Architecture

```
                      Inference Orchestrator
                    (owns the decode loop)
                              |
        +---------------------+---------------------+
        |                     |                     |
   VRAM Pressure      Attention Importance    Precision
     Monitor                Scorer            Controller
   (NVML polling)      (per-page attention    (importance + pressure
    GREEN/YELLOW/RED     mass, EWMA)          -> requantisation plan)
        |                     |                     |
        +---------------------+---------------------+
                              |
                  Paged Mixed-Precision KV Cache
              (page table, VMM allocator, quant kernels)
                              |
                      MicroInfer CUDA kernels
              (attention, RoPE, RMSNorm, SwiGLU, quant)
```

Four decisions shape everything above:

**The from-scratch boundary stops at the BLAS.** Attention, the paged cache,
quantisation, RoPE and RMSNorm are hand-written CUDA. Dense projections call
cuBLAS. A hand-written GEMM would be 2–5x slower than cuBLAS and every latency
number in the paper would inherit that gap, leaving a reviewer unable to
separate policy overhead from a slow matmul
([ADR-0001](docs/adr/0001-from-scratch-boundary.md)).

**The engine owns all VRAM; PyTorch is offline-only.** PyTorch's caching
allocator retains freed device memory instead of returning it to the driver, so
`nvmlDeviceGetMemoryInfo` would report its reservation as used memory —
indistinguishable from the contention this project exists to detect. PyTorch
appears in exactly one tool, generating golden reference tensors offline
([ADR-0002](docs/adr/0002-engine-owns-vram-torch-is-offline-only.md)).

**Reclaimed bytes must reach the driver.** The cache reserves one virtual
address range per tier through the CUDA VMM API and releases physical granules
as they empty. A page "freed" inside a private pool is invisible to NVML, and
the response to memory pressure would be an unmeasurable no-op
([ADR-0007](docs/adr/0007-vmm-allocator-with-per-tier-address-ranges.md)).

**Correctness is gated on logits, not on text.** The engine's accumulation order
differs from HuggingFace's, so bit-reproducibility is unavailable; a 1e-4 logit
difference at a near-tie flips an argmax and diverges every token after it. The
gate is top-1 agreement ≥ 99% and mean KL < 1e-3
([ADR-0006](docs/adr/0006-correctness-gate-is-top1-logit-agreement.md)).

---

## Status

Milestone 0 builds the substrate: a correct, unoptimised, from-scratch inference
path with a paged KV cache whose pages can be held at four precision tiers. It
contains **no adaptivity** — a page's tier is set once, at allocation.

| Milestone | Scope | State |
|---|---|---|
| **0** | Paged, multi-tier inference path | In progress |
| 1 | Contention simulator, VRAM pressure monitor | Not started |
| 2 | Importance scorer, precision controller, runtime requantisation | Not started |
| 3 | NLP evaluation across four conditions, ablations | Not started |

Work is tracked as GitHub issues. The Milestone 0 spec is
[issue #1](../../issues/1); tickets carry native blocking links, so any issue
with no open blocker can be started.

---

## Building

**Prerequisites.** CUDA 12.x with `nvcc`, a C++17 host compiler, CMake ≥ 3.24,
Python ≥ 3.10, and an NVIDIA GPU with its driver installed. The extension links
the CUDA **driver** API (`libcuda`) as well as the runtime API — `cuMemCreate`
and `cuMemMap`, which the allocator needs, live only in the driver API.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # pinned versions the recorded results used
pip install -e ".[test]"         # builds the CUDA extension via scikit-build-core
pytest                           # kernel and toolchain tests
```

`pyproject.toml` defines what the package depends on; `requirements.txt` pins
the exact versions a result was produced with.

Target GPU architecture is detected from the card present at build time. Build
for a different target by overriding it:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" pip install -e .
```

Verified on Ubuntu 24.04 with CUDA 12.9, GCC 13.3, CMake 3.28.3, Python 3.12.3
and an RTX 4050 Laptop (compute capability 8.9).

**PyTorch is not a dependency and must not become one.** It appears only in the
offline golden tensor generator, which runs in a separate environment; a test
asserts that `torch` is absent from `sys.modules` after importing the engine
(ADR-0002).

## Repository map

| Path | What it holds |
|---|---|
| [`CONTEXT.md`](CONTEXT.md) | The project glossary. `page`, never `block` |
| [`docs/adr/`](docs/adr/) | Architectural decisions, with the rejected options |
| [`docs_research/`](docs_research/) | Research notes, paper outline, knowledge roadmap |
| [`docs/paper.tex`](docs/paper.tex) | Thesis source |
| [`docs/agents/`](docs/agents/) | Issue tracker and triage conventions for AI agents |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Conventions that are not negotiable, and why |

Read `CONTEXT.md` before writing anything. Several words in this domain are used
for more than one thing in the surrounding literature, and the glossary picks one
meaning for each.

---

## Limitations

Stated here rather than buried, because they are real.

- **Single GPU, single user, `batch_size=1`.** This is the premise, not a
  simplification to be lifted later.
- **`MicroInfer` is a research prototype.** It is not production-grade and is not
  trying to be.
- **The benefit scales with KV footprint fraction.** Aggressive grouped-query
  attention is the industry's own answer to KV cache size, and this method
  matters least on the models most likely to be deployed. Results are reported
  alongside that fraction rather than as a universal claim.
- **NVML polling has inherent detection latency.** The polling interval is a
  tunable trade-off and is itself subject to ablation.
- **Measurements come from a laptop** whose available memory and clocks depend on
  what else is running and on thermal state. Every recorded number carries its
  context.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
