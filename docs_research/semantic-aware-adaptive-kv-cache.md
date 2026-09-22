# Semantic-Aware Adaptive KV Cache Precision for Robust Single-User LLM Inference under VRAM Contention

*(Standalone project — built from scratch, no dependency on prior personal projects. Engine referred to here as `MicroInfer`; rename freely.)*

## 1. Paper Outline

### Title
**Semantic-Aware Adaptive KV Cache Precision for Robust Single-User LLM Inference under Unpredictable VRAM Contention on Consumer GPUs**

### Abstract (draft skeleton)
- Problem: Local, single-user LLM inference on consumer/laptop GPUs faces *unpredictable* VRAM contention from co-running applications (browsers, chat apps, background renderers), unlike datacenter serving where GPU allocation is controlled and workload-aware.
- Gap: Prior adaptive KV cache work (FineServe, MorphServe, MIRAGE, eLLM, KV-RM) assumes multi-tenant, scheduler-visible environments and optimizes uniformly across the cache.
- Contribution: (1) An empirical characterization of VRAM contention patterns on consumer laptops during everyday multitasking; (2) a lightweight runtime VRAM-pressure detector requiring no scheduler cooperation; (3) a semantic-aware adaptive precision policy that reallocates KV cache precision (FP16 → INT8 → INT4) guided by attention-derived importance scores rather than uniform degradation; (4) a task-stratified NLP evaluation showing where degradation hurts most (long-document QA, multi-turn dialogue, summarization) and how semantic-aware allocation mitigates it versus uniform quantization and versus the OOM/crash status quo.

### 1. Introduction
- Motivate with the *local inference* trend (Ollama, LM Studio, llama.cpp) — increasingly common on consumer laptops with limited, shared VRAM.
- Contrast explicitly with datacenter serving literature (cite FineServe/MorphServe/MIRAGE/eLLM/KV-RM as the closest prior work; state precisely what they assume that does not hold locally: dedicated GPU, workload visibility, multi-request scheduling).
- State the three research questions (RQs):
  - **RQ1**: How volatile is available VRAM on consumer laptops under realistic multitasking, and how does it interact with an in-progress single generation session?
  - **RQ2**: Can a scheduler-agnostic runtime detect and react to VRAM pressure fast enough to avoid OOM without a full session crash?
  - **RQ3**: Does allocating precision by semantic/attention importance (vs. uniform) preserve task-level NLP quality better under equal average compression?
- Summarize contributions and results preview.

### 2. Related Work
- **KV cache compression/quantization**: KVQuant, KIVI, KVTuner, Cocktail, KVmix, FlashInfer FP8-KV, LMDeploy TurboMind — position as *static/offline* precision assignment; your work is *runtime-reactive*.
- **Adaptive/runtime-aware serving**: FastCache, FineServe, MorphServe, MIRAGE, eLLM, KV-RM, Recency/Frequency Adaptive KV Caching — all datacenter/multi-tenant; explicitly table the assumptions each makes (dedicated/shared GPU, scheduler visibility, multi-request vs. single-session) to justify the gap.
- **Attention-guided importance/eviction**: Ada-KV, H2O-style attention-based eviction, AttentionRAG — reuse the idea of attention-derived importance, but apply it to *precision allocation under external memory pressure* rather than *eviction* or *offline compression budget*.
- **Consumer/edge inference systems**: llama.cpp, PowerInfer, ATSInfer, APEX — engineering systems that handle OOM by refusing/crashing or static offload; none handle *mid-session* precision adaptation triggered by *external* (non-model) VRAM contention.

### 3. Problem Characterization (RQ1)
- Experimental setup: laptop with an **RTX 4050 Laptop GPU, 6141 MiB** (not "4050/4060" — the 6 GiB figure is what makes the KV footprint arithmetic in ADR-0003 binding, and an 8 GiB 4060 would change every threshold below), fixed model + prompt set, background load generators (Chrome with N tabs, Discord screen-share, OBS recording, a game demo).
- Measurements: available VRAM over time (via NVML polling), variance, spike frequency/magnitude, correlation with common user actions.
- Deliverable: a small "VRAM contention trace" dataset + descriptive statistics — this is a citable empirical contribution even before the systems contribution.

### 4. System Design (RQ2) — see Section 2 of this document for full architecture
- VRAM Pressure Monitor
- Precision Controller (policy layer)
- Paged, Mixed-Precision KV Cache Manager
- Integration with the decode loop (MicroInfer, built from scratch — see Section 5)

### 5. Semantic-Aware Precision Policy (RQ3)
- Attention-derived importance scoring (per KV block).
- Mapping importance → precision tier (FP16 / INT8 / INT4).
- Re-scoring cadence and staleness handling across multi-turn topic shifts.
- Formal policy definition and pseudocode (see Section 3.3 below).

### 6. NLP Evaluation
- Tasks: multi-turn dialogue coherence, long-document QA (needle position-stratified), summarization fidelity.
- Datasets: MT-Bench or self-built multi-turn set; NarrativeQA or a self-built Vietnamese/English long-doc QA set; CNN/DailyMail or VietNews.
- Conditions compared: (a) no contention (upper bound), (b) contention → OOM/crash (status quo), (c) contention → uniform quantization, (d) contention → semantic-aware quantization (proposed).
- Metrics: anchor-fact recall/coherence score (dialogue), EM/F1 stratified by answer position (QA), ROUGE-L/BERTScore (summarization); plus systems metrics (P50/P99 latency, throughput, OOM rate).

### 7. Results (placeholder structure)
- RQ1 results: contention characterization plots.
- RQ2 results: detection latency, false positive/negative rate of pressure detection, OOM avoidance rate.
- RQ3 results: task metrics under each condition; correlation analysis between attention magnitude and quality drop.

### 8. Discussion & Limitations
- Single-GPU, single-model scope; generalization across model families/attention variants (MHA/GQA/MLA); overhead of importance scoring itself; the engine (MicroInfer) is a research prototype, not production-grade — note this honestly as a scope limitation, not a weakness to hide.

### 9. Conclusion & Future Work
- Extending to hybrid attention/SSM models; extending policy to weights (not just KV cache); on-device continual monitoring as an OS-level service.

### Target Venues
- Primary: MLSys, EuroSys, workshop tracks (ML for Systems @ NeurIPS/ICML), or ACL/EMNLP *Industry/System-Demonstration* track (given the NLP evaluation component).
- Fallback: arXiv preprint + well-documented GitHub repo — sufficient for a strong CV artifact even pre-acceptance.

---

## 2. System Architecture

### 2.1 High-Level Component Diagram (textual)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Host Application                          │
│  (chat UI / CLI calling MicroInfer for generation)                │
└───────────────────────────┬───────────────────────────────────────┘
                             │ generate(prompt, session_state)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Inference Orchestrator                       │
│  - Owns the decode loop                                          │
│  - Calls VRAM Monitor before/periodically during decode          │
│  - Calls Precision Controller when pressure event fires          │
│  - Calls Attention Scorer to refresh importance map               │
└───────┬───────────────────────┬───────────────────────┬──────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌───────────────────┐   ┌───────────────────────┐
│ VRAM Pressure  │     │ Attention Importance│  │ Precision Controller  │
│ Monitor        │     │ Scorer              │  │ (Policy Engine)        │
│ - NVML polling │     │ - reads attention   │  │ - importance→tier map  │
│ - threshold /  │     │   weights per step   │  │ - hysteresis/cooldown  │
│   trend detect │     │ - per-block score    │  │ - emits re-quant plan  │
└───────┬────────┘     │ - decay/staleness    │  └──────────┬─────────────┘
        │              └──────────┬───────────┘             │
        │                         │                          │
        └───────────┬─────────────┴──────────────┬───────────┘
                     ▼                            ▼
        ┌─────────────────────────────────────────────────┐
        │        Paged Mixed-Precision KV Cache Manager     │
        │  - block table (page → precision tier)             │
        │  - FP16 / INT8 / INT4 (de)quantize kernels          │
        │  - in-place re-quantization of existing blocks      │
        └───────────────────────┬───────────────────────────┘
                                 ▼
                  ┌───────────────────────────┐
                  │   MicroInfer CUDA Kernels   │
                  │ (attention, GEMM, softmax)  │
                  │   — built from scratch      │
                  └───────────────────────────┘
```

### 2.2 Data Flow per Decode Step
1. Orchestrator requests the next token; before dispatching the attention kernel, it checks a lightweight flag set by the VRAM Monitor thread (no blocking call on the hot path — monitor runs on a separate thread/timer, e.g., every 50–100 ms).
2. If a pressure flag is set, the Precision Controller is invoked *between* decode steps (never mid-kernel) to avoid partial/inconsistent cache state.
3. The Controller consults the current importance map (refreshed every `R` steps, see 3.3) and produces a **re-quantization plan**: a list of `(block_id, current_tier, target_tier)`.
4. The KV Cache Manager applies the plan: for blocks being downgraded, it quantizes in place using the engine's own quantization kernels (built in Milestone 2, see 4.1); for blocks eligible to be upgraded (pressure relieved), it restores from a small retained buffer or recomputes if not retained (tunable trade-off, see 3.4).
5. Decode continues with the updated paged block table.

### 2.3 Module Boundaries (for implementation)
| Module | Responsibility | Key interfaces |
|---|---|---|
| `vram_monitor` | Poll NVML, maintain rolling free-memory stats, expose pressure events | `PressureEvent poll()`, `subscribe(callback)` |
| `attention_scorer` | Accumulate per-block attention mass, expose importance scores | `update(attn_weights, block_ids)`, `get_scores() -> Dict[block_id, float]` |
| `precision_controller` | Map scores + pressure level to a re-quant plan | `plan(scores, pressure_level, current_tiers) -> List[BlockAction]` |
| `kv_cache_manager` | Own paged blocks, execute (de)quantization, update block table | `requantize(block_id, target_tier)`, `alloc_block()`, `evict_block()` |
| `core_kernels` | Attention, GEMM, softmax, INT8/INT4 quantize-dequantize — **built from scratch for this project** | minimal, well-tested primitives; no external inference engine dependency |
| `eval_harness` | Run NLP tasks under each of the 4 conditions, collect metrics | `run_condition(task, condition) -> Metrics` |
| `contention_simulator` | Launch/stop synthetic GPU memory hogs for controlled experiments | `start_load(pattern)`, `stop_load()` |

---

## 3. Detailed Design

### 3.1 VRAM Pressure Monitor
- **Signal source**: `nvidia-ml-py` (NVML bindings) polling `nvmlDeviceGetMemoryInfo` on a background thread, independent of the CUDA stream used for inference.
- **Pressure levels** (3-tier, simple and explainable):
  - `GREEN`: free VRAM > `T_high` (e.g., > 20% of total)
  - `YELLOW`: `T_low` < free VRAM ≤ `T_high` (e.g., 10–20%)
  - `RED`: free VRAM ≤ `T_low` (e.g., < 10%) — imminent OOM risk
- **Debounce/hysteresis**: require the level to persist for `K` consecutive polls (e.g., 3 polls at 50 ms = 150 ms) before firing a transition event, to avoid thrashing on transient spikes.
- **Output**: an event queue the Orchestrator drains between decode steps; never blocks the decode loop.

### 3.2 Attention Importance Scorer
- **What to accumulate**: for each decode step, the attention weights from the query token to all cached KV positions (extracted from the attention kernel's intermediate softmax output — since the kernel is being built from scratch, expose this as an optional debug/telemetry output rather than adding overhead to the fast path by default).
- **Aggregation**: maintain an exponentially-weighted moving average (EWMA) per KV block:
  `score[block] = α * mean_attn_to(block) + (1 - α) * score[block]_prev`
  with `α` tuned (e.g., 0.2) to balance responsiveness vs. stability across multi-turn topic shifts.
- **Layer weighting**: optionally weight deeper layers more heavily (empirically deeper-layer attention correlates more with semantic relevance); expose as a tunable vector `w_layer`, default uniform for the first implementation to keep the baseline simple, then ablate.
- **Staleness handling**: on detecting a topic shift heuristic (e.g., a new user turn with low lexical/embedding overlap with the recent window), force a full re-score pass instead of relying on EWMA decay alone.

### 3.3 Precision Controller (Policy Engine)
- **Inputs**: pressure level (`GREEN`/`YELLOW`/`RED`), per-block importance scores, current per-block tier, a target compression ratio implied by the pressure level.
- **Policy (initial, simple, ablatable)**:
  1. Rank blocks by importance score ascending (least important first).
  2. Under `YELLOW`: downgrade the bottom `X%` of blocks by one tier (FP16→INT8), skipping the most recent `N` tokens (recency floor — recent context is almost always relevant) and any block flagged as containing an anchor fact if such a lightweight heuristic is enabled.
  3. Under `RED`: more aggressive — downgrade bottom `Y%` (`Y > X`) potentially two tiers (FP16→INT4) or push to CPU as a last resort before refusing/truncating.
  4. Under `GREEN` after a prior downgrade: opportunistically upgrade blocks back toward FP16 in importance order, budget-permitting, to recover quality once contention passes.
- **Pseudocode**:
```
function plan(scores, pressure, tiers, recency_floor):
    candidates = blocks not in recency_floor, sorted by scores ascending
    if pressure == RED:
        target_fraction = Y
        max_downgrade_steps = 2
    elif pressure == YELLOW:
        target_fraction = X
        max_downgrade_steps = 1
    else:
        return upgrade_plan(candidates_desc_by_score, tiers)  # opportunistic recovery

    n = ceil(len(candidates) * target_fraction)
    actions = []
    for block in candidates[:n]:
        new_tier = downgrade(tiers[block], steps=max_downgrade_steps)
        actions.append((block, tiers[block], new_tier))
    return actions
```
- **Hysteresis on the policy itself**: do not re-plan more often than every `M` decode steps (e.g., 8–16) to bound re-quantization overhead.

### 3.4 Paged Mixed-Precision KV Cache Manager
- Owns a paged KV cache, built from scratch (see Milestone 1, Section 4.1): each page has a `precision_tier` field in its **page table** entry, alongside the usual logical→physical page index. (vLLM calls this a *block table*; `CONTEXT.md` fixes the project's term as *page table*. The rest of this document still uses the older word in places.)
- **Downgrade path**: apply the engine's own INT8/INT4 quantization kernel in place, free the now-unused higher-precision bytes back to the allocator.
- **Upgrade path**: two options to implement and compare:
  - *(a) Recompute-on-upgrade*: re-run the affected prefix's forward pass for that block — costly but always correct.
  - *(b) Retain-buffer*: keep a small CPU-side or compressed FP16 shadow copy for recently-downgraded blocks (bounded LRU buffer) so upgrade is a fast copy-back instead of recompute — introduces its own memory cost, worth an ablation.
- **Allocator interaction**: precision tier changes the byte footprint of a page, and the freed bytes must return to the **driver**. `nvmlDeviceGetMemoryInfo` reports what the driver has handed out, so bytes recycled inside a private pool are invisible to the only measurement RQ2 has. The cache therefore reserves one virtual address range per tier through the CUDA VMM API — `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap` — packs pages from the low end of each range, and releases granules with `cuMemUnmap`/`cuMemRelease` as they fall empty. See ADR-0007.
- **Rejected: over-provisioning page slots per tier.** An earlier draft of this section proposed it as "simpler for a first implementation". It is the one approach that *cannot* work: an INT4 page written into an FP16-sized slot wastes the difference, nothing returns to the driver, NVML never moves, and the response to RED pressure becomes a no-op that cannot be measured. Adopting it would mean withdrawing RQ2 from the thesis. Kept on the record rather than deleted, because it is an attractive shortcut and someone will propose it again.

### 3.5 Contention Simulator (for controlled experiments)
- A standalone utility that allocates/frees GPU memory in configurable patterns (step function, sawtooth, random spikes) to reproduce RQ1's empirical traces deterministically for RQ2/RQ3 experiments — decouples "does the mechanism work" from "waiting for a real Chrome tab to spike memory."

---

## 4. "Vibe Coding" Implementation Guide (working with an AI coding assistant)

This section is written so you can hand each part, in order, to an AI coding assistant (e.g., Claude Code) as a scoped task — small enough to verify, large enough to be useful. Because the engine is now built from scratch, the build order starts one level lower than before.

### 4.1 Suggested build order

**Milestone 0 — Minimal working inference path (no adaptivity yet)**
1. A basic decoder-only transformer forward pass — a Python orchestrator over hand-written CUDA kernels via `pybind11`, with cuBLAS for the dense projections (ADR-0001) — plain FP16 attention, no paging yet, single fixed-size KV cache buffer.
   - **Model is not a free choice.** Qwen2.5-0.5B-Instruct for development, Qwen2.5-1.5B-Instruct at 32K context for the experiments (ADR-0003). The earlier "0.5–1.5B parameter model" range does not survive the arithmetic: at 0.5B with an 8K context the *entire* KV cache is 96 MiB, and flattening all of it to INT4 reclaims 72 MiB — against contention spikes of 200–400 MiB from a browser tab. The mechanism could not avoid a single OOM at that scale, and RQ2 would be dead before a line of code was written. Only the long-context column clears the threshold.
   - Goal: correct, unoptimized, verified against a HuggingFace reference on a fixed prompt and seed. **The gate is top-1 logit agreement ≥ 99% with mean KL < 1e-3, not matching generated text** — cuBLAS accumulates along a different path, so a 1e-4 logit difference at a near-tie flips an argmax and diverges everything after it (ADR-0006).
2. Add a **paged** KV cache (fixed precision, FP16 only) — logical block table, physical page pool, page allocation/free. Validate: output identical to the non-paged version.
3. Add INT8 and INT4 quantize/dequantize kernels for KV cache blocks, usable *statically* (i.e., quantize once at allocation, no runtime switching yet). Validate: perplexity delta on a held-out set vs. FP16, matches published expectations for the quantization scheme chosen (e.g., roughly in line with KIVI/KVQuant-reported degradation).

**Milestone 1 — Contention & monitoring (no adaptivity yet, just observing)**
4. **Contention Simulator** — self-contained, easy to test, and unblocks controlled experiments early.
5. **VRAM Pressure Monitor** — wrap NVML polling, unit test against the simulator.

**Milestone 2 — Adaptivity**
6. **Attention Importance Scorer** — hook into the attention kernel from Milestone 0 to extract per-block scores; validate against a known synthetic attention pattern first (feed a hand-crafted attention matrix and check the aggregation math) before hooking into the real kernel.
7. **Precision Controller** — pure policy logic, no CUDA; fully unit-testable with mocked scores/pressure levels.
8. **Runtime re-quantization in the KV Cache Manager** — extend the static quantization from Milestone 0 step 3 to support *in-place, mid-session* tier changes. This is the riskiest, most novel part — implement the downgrade path first (reuses the static quantize kernel), upgrade path second.
9. **Orchestrator wiring** — connect the decode loop to the above without adding synchronization stalls; profile with `nsys`/`ncu` before/after to confirm the monitor doesn't add measurable per-step overhead.

**Milestone 3 — Evaluation**
10. **Eval harness** — task datasets + metrics + the 4-condition experiment runner.

This order matters: Milestone 0 alone (a correct, from-scratch paged+quantized inference path) is already a substantial, presentable engineering artifact on its own — treat it as a checkpoint you could stop at and still have something demonstrable, before committing to the full adaptive system.

### 4.2 Prompting pattern per module (example for the Precision Controller)
When handing a module to an AI coding assistant, give it:
- The **interface** (function signatures from the table in 2.3).
- The **pseudocode** (3.3) as a spec, not as literal code to copy — ask it to implement, then to write unit tests covering: hysteresis behavior, recency floor exclusion, GREEN-state upgrade ordering, and edge cases (empty block list, all blocks already at min tier).
- Explicitly ask it to **keep this module free of CUDA/NVML dependencies** so it stays unit-testable in isolation — this is a good general practice to state up front (decoupling policy logic from hardware I/O).

### 4.3 Verification discipline (important for a research artifact, especially building from scratch)
- For each module, ask the assistant to generate **both** the implementation and a small **correctness test** before moving to integration — this matters more here than in typical app development, because silent correctness bugs in a re-quantization path can produce plausible-looking but wrong text (hard to catch by eye).
- For every new CUDA kernel (Milestone 0 especially, since there is no prior reference implementation to lean on), always validate numerically against a PyTorch/HuggingFace reference on a fixed seed/prompt before trusting any benchmark numbers — a benchmark showing a "speedup" from a broken kernel is a common failure mode to guard against explicitly, and it's easier to fall into when starting from scratch than when extending known-correct code.
- Keep a running **benchmark log** (config, hardware, git commit hash, results) from the very first working version — this becomes the paper's experiment log and saves you from re-deriving results later.

### 4.4 Repo layout

Written to match what the repository actually is, which the flat layout in an
earlier draft of this section predates. The shape follows from ADR-0002: a
Python orchestrator over a CUDA extension, with PyTorch nowhere in the engine
process.

```
├── pyproject.toml                 # scikit-build-core + CMake; torch is not a dependency
├── CMakeLists.txt                 # links CUDA::cudart AND CUDA::cuda_driver (ADR-0007)
├── requirements.txt               # pinned environment; torch must never appear (ADR-0002)
├── CONTEXT.md                     # glossary — the unit is a `page`, never a `block`
├── CONTRIBUTING.md
├── .clang-format                  # Allman, 2-space, namespaces indented
├── .pre-commit-config.yaml        # clang-format + the two engine invariants
├── docs/
│   ├── adr/                       # architectural decisions, with rejected options
│   └── agents/                    # issue tracker and triage conventions
├── src/
│   ├── microinfer/                # Python orchestrator. Sequences kernels; does no arithmetic
│   └── core_kernels/              # hand-written CUDA
│       ├── rmsnorm.cu  rope.cu  swiglu.cu
│       ├── attention.cu           # online softmax, causal mask, GQA
│       ├── paged_cache.cu         # page table + VMM allocator (ADR-0007)
│       ├── quant.cu               # INT8 / INT4 / INT2 quantise-dequantise (ADR-0005, ADR-0008)
│       ├── gemm.cu                # thin cuBLAS wrapper (ADR-0001)
│       ├── bindings.cpp           # pybind11. NumPy in/out at the test-facing surface (Seam B)
│       └── include/microinfer/
├── tests/
│   ├── test_*.py                  # Seam A (Engine) and Seam B (extension)
│   └── golden/                    # reference tensors: generated offline, stored, regenerated rarely
├── tools/
│   ├── gen_golden.py              # the ONLY place torch and transformers appear (ADR-0002)
│   └── check_engine_invariants.py # enforces that rule on commit
├── studies/                       # tiled GEMM vs cuBLAS — deliberately off the inference path (ADR-0001)
├── experiments/logs/              # append-only benchmark log: config, hardware, commit
└── graduation_thesis/             # paper source
```

Two boundaries in that tree carry decisions rather than taste. `tools/` is the
only directory where PyTorch may be imported, because its caching allocator
would otherwise corrupt the NVML reading RQ2 depends on. And `studies/` is
unreachable from the engine's import graph, so a hand-written GEMM that is
slower than cuBLAS cannot contaminate a latency measurement.

## 5. Knowledge Roadmap (Prerequisite Learning Plan)

This section lists the knowledge required to execute this project and write the paper, organized in the order it is needed (matches the build milestones in Section 4).

### 5.1 CUDA / GPU Programming (foundation)

**Programming model**
- Thread/block/grid hierarchy; warp as the SIMT execution unit (32 threads) — the single most important concept for performance reasoning.
- Warp divergence: threads in a warp taking different branches serialize execution — relevant to causal-mask implementation.
- Occupancy: active warps per SM vs. theoretical max; bounded by registers/thread, shared memory/block, threads/block. High occupancy does not always mean fastest — a common misconception to avoid.

**Memory hierarchy**
| Memory | Speed | Scope | Typical use |
|---|---|---|---|
| Register | Fastest | Per-thread | Loop accumulators |
| Shared memory | Very fast | Per-block | GEMM tiles, K/V tiles in attention |
| L2 cache | Fast | Device-wide | Automatic, but layout-aware code benefits |
| Global memory (HBM) | Slowest (relative) | Device-wide | Main tensor storage — usual bottleneck |

- Memory coalescing: adjacent threads in a warp accessing adjacent addresses merge into fewer transactions — directly relevant to how you lay out paged KV cache blocks.
- Roofline model: Arithmetic Intensity = FLOPs / Bytes accessed; decode-time attention (batch=1) is typically memory-bound (motivates KV cache compression); large GEMMs are typically compute-bound. Compute AI for each kernel you write.

**Kernels to implement and optimize**
- Tiled GEMM: naive → shared-memory tiled → compare against cuBLAS with `ncu` to see the gap.
- Online softmax: incremental max/sum tracking (core of Flash-Attention-style kernels), avoids materializing the full attention matrix.
- Reduction via warp shuffle (`__shfl_down_sync`) instead of shared memory + `__syncthreads()` where possible — used in softmax, layernorm, and the EWMA importance-score aggregation in this project.

**Profiling tools**
- Nsight Systems (`nsys`): kernel timeline, CPU-GPU overlap, idle gaps.
- Nsight Compute (`ncu`): per-kernel occupancy, memory throughput %, warp stall reasons (`sm__throughput`, `dram__throughput`, `achieved_occupancy`).

**Python/CUDA bridging**
- `pybind11` to expose `.cu` kernels to Python; managing memory handoff between PyTorch tensors and raw CUDA pointers (`.data_ptr()`).

**Suggested pace**: Weeks 1–2 basics (vector add, transpose, naive matmul) → Week 3 tiled GEMM vs. cuBLAS → Week 4 reduction/softmax → Weeks 5–6 full attention kernel + pybind11 binding, validated against a PyTorch reference.

### 5.2 Transformer Architecture & Inference Internals

- Self-attention formula and the `1/sqrt(d_k)` scaling rationale; multi-head reshape/transpose (a common source of bugs when hand-writing kernels); causal masking implementation without materializing an explicit mask matrix.
- Attention variants and their KV cache impact: **MHA** (largest cache), **MQA** (shared K/V, smallest cache, lower quality), **GQA** (grouped, used in LLaMA-2/Mistral — recommended choice for this project since it's the most realistic modern default and directly shapes paged-block sizing).
- RoPE (Rotary Position Embedding): applied to Q/K pre-attention; important failure mode to check — quantize/dequantize error interacting with RoPE-transformed vectors during correctness validation.
- **KV cache sizing formula** (memorize, used repeatedly to motivate compression in the paper):
  `KV cache size = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × bytes_per_element`
- **PagedAttention** (vLLM): block table (logical→physical page mapping), on-demand page allocation instead of pre-allocating for max length; this project extends the block table with a `precision_tier` field.
- Quantization basics: symmetric vs. asymmetric (asymmetric usually better for KV cache due to skewed distributions); per-tensor vs. per-channel vs. per-token granularity (per-token common for KV cache); outlier sensitivity and its effect on scale — measure perplexity degradation as the required baseline number before task-level evaluation.
- Attention weights as an importance signal: know the limitation ("attention is not explanation" debate) and the **attention sink** phenomenon (the first/BOS token absorbs disproportionate attention regardless of semantic relevance) — must be accounted for, or the importance map will spuriously always protect the first token.

**Suggested pace**: Week 7 read "Attention Is All You Need" + an illustrated-transformer explainer → Week 8 implement single-head attention, validate against `torch.nn.functional.scaled_dot_product_attention` → Week 9 read the PagedAttention paper, implement a basic block table → Week 10 static INT8 KV quantization + perplexity baseline.

### 5.3 Operating Systems & Systems Knowledge

- GPU memory is managed by the driver per-process context; consumer GPUs (unlike datacenter MIG/MPS-isolated setups) provide **no memory isolation guarantee** between co-running processes — this is the core justification for the whole project.
- Unified Memory / oversubscription (`cudaMallocManaged` vs. `cudaMalloc`) — know this exists to explain why it's *not* the chosen approach (PCIe page-fault overhead) in favor of adaptive quantization.
- NVML API: `nvmlDeviceGetMemoryInfo()` (core signal), `nvmlDeviceGetUtilizationRates()` (optional, to distinguish compute-busy vs. memory-busy); Python via `pynvml`. Known limitations to state honestly in the paper: polling-based (inherent detection latency), and polling interval is itself a tunable trade-off worth an ablation.
- Concurrency: Python `threading` (CUDA calls release the GIL during kernel execution, so threading remains effective for the monitor) or `asyncio`; C++ `std::thread`/`std::atomic` if the core is C++. Avoid races between the Attention Scorer writing scores and the Precision Controller reading them — double-buffering is a clean solution. CUDA streams: understand the concept for a possible future optimization where re-quantization overlaps with the next decode step's compute.
- Control-systems vocabulary applied to the Precision Controller: **hysteresis/debounce** (require a signal to persist for K consecutive polls before acting, to avoid thrashing), basic **feedback loop** framing (measure → decide → act → re-measure), **rate limiting** (cap re-planning frequency). Using this terminology correctly signals system-design maturity to reviewers.
- Benchmarking discipline: report **P50/P99 latency**, not just averages (occasional re-quantization spikes matter more to UX than the mean suggests); measure prefill and decode throughput separately (different bottleneck regimes).

**Suggested pace**: Week 11 NVML docs + a simple polling script, observe real VRAM changes while opening/closing other apps → Week 12 build the Contention Simulator → Week 13 VRAM Pressure Monitor with hysteresis + unit tests against synthetic patterns → Week 14 integrate into the decode loop, measure overhead with `nsys`.

### 5.4 NLP Evaluation Methodology

- **Perplexity**: sanity-check metric for quantization degradation (FP16 vs. INT8 vs. INT4) on a held-out set (e.g., WikiText-2/C4 subset); doesn't localize *where* degradation happens — motivates task-specific evaluation.
- **ROUGE-L**: LCS-based lexical overlap for summarization; known limitation — penalizes valid paraphrases. **BERTScore**: embedding-based semantic similarity, complements ROUGE; report both, as is standard practice. **EM/F1** (SQuAD-style): standard extractive-QA metrics, with EM requiring normalized exact match and F1 being token-overlap based.
- **Needle-in-a-haystack** benchmark design: insert a fact at a controlled position within a long context, measure retrieval accuracy stratified by position and context length. Directly useful here because you control which paged block the needle falls into, letting you measure the effect of that specific block's precision tier being downgraded — the most direct test of RQ3.
- **LLM-as-judge**: known biases to disclose — self-preference bias, position bias (mitigate by randomizing answer order in pairwise comparisons); treat as a proxy, ideally cross-validated with a small human-evaluation sample; use a clear rubric and chain-of-thought judging to improve consistency.
- **Multi-turn dialogue evaluation** (MT-Bench-style): design an **anchor-fact recall** task specifically — an early turn states a fact, a later turn (far enough away to fall in a likely-downgraded block) asks for it back; measure both exact recall and overall coherence via LLM-as-judge.
- **Controlled experimental design**: when comparing uniform vs. semantic-aware quantization, hold the **average compression ratio equal** — otherwise a quality difference could simply reflect a different compression level, not a better allocation strategy. Prepare ablations in advance (recency floor on/off, layer weighting on/off, re-scoring frequency, EWMA α) — reviewers will ask for these regardless.

**Suggested pace**: Week 15 perplexity pipeline (FP16/INT8/INT4 static baseline) → Week 16 build a small (50–100 sample) needle-in-haystack set (prioritize this — it's the most direct RQ3 evidence) → Week 17 set up the LLM-as-judge pipeline and sanity-check its consistency → Week 18 run the full 4-condition experiment across all three tasks, plus ablations.

### 5.5 Research Skills

- **Reading papers efficiently — the 3-pass method**: Pass 1 (5–10 min) — abstract, intro, figures/results only, to answer "what does this paper contribute." Pass 2 (30–60 min) — method and related work in depth. Pass 3 (as needed) — experimental setup/appendix detail, usually needed only when directly reproducing or comparing against a specific baseline. Do not read linearly start-to-end on a first pass. Tools: Semantic Scholar/Google Scholar for citation graphs, Connected Papers for visual topic mapping.
- **Controlled experimental design**: change exactly one variable at a time; keep model, dataset, prompts, seed, and hardware fixed across compared conditions. Choose baselines that are genuinely strong/plausible (a "uniform quantization" baseline, not just "OOM crash," is what gives the comparison scientific value).
- **Writing Related Work well**: avoid a flat "Paper A does X, Paper B does Y" list. Instead, group papers by approach, summarize each group's core idea, and explicitly state which assumption each group makes that does *not* hold in your setting — this "gap statement" is the actual scholarly contribution of the section. Template: *"While [Method X] achieves strong results in [context], it assumes [assumption] which does not hold in [your setting], motivating our approach."*
- **Basic statistics**: report confidence intervals or standard error, not a single mean (N ≥ 5 runs as a practical minimum for systems experiments); understand why multiple seeds/runs matter for both sampling-based generation and for noisy latency/throughput measurements (background load, thermal throttling — directly relevant to this project's laptop setting); prefer **paired** comparisons (same question set across conditions) over unpaired, since paired tests are more sensitive for this experimental design.
- **Scientific writing**: prioritize clarity and precision over ornate language; back every substantive claim with evidence (a number or a citation); follow IMRaD discipline — Introduction sells the idea, Method describes objectively, Results states facts without interpretation, Discussion is where interpretation belongs.

**Suggested pace**: ongoing throughout, but concentrate effort on Related Work drafting once Section 2 of the paper outline is being written (after Milestone 0 is working and the literature comparison table is being finalized), and on statistical rigor once Milestone 3 experiments begin producing numbers.

### 5.6 Summary Timeline (all sections combined)

| Weeks | Focus | Output |
|---|---|---|
| 1–6 | CUDA fundamentals → attention kernel | Working, validated single-head attention kernel |
| 7–10 | Transformer internals, paged cache, static quantization | Milestone 0 complete: correct, paged, quantized inference path |
| 11–14 | VRAM monitoring, contention simulation, control-loop design | Milestone 1–2 core mechanism working, overhead-profiled |
| 15–18 | NLP evaluation pipelines and datasets, full experiments | Milestone 3 complete: 4-condition results + ablations |
| Ongoing | Paper reading, Related Work, statistical rigor, writing | Draft paper sections, ready for internal review/submission |
