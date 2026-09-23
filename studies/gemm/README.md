# GEMM study: naive and tiled kernels against cuBLAS

The CUDA learning objective of the research notes' §5.1, and issue #11. It is
kept **off the inference path** by ADR-0001. The engine never calls this code,
and `tests/test_studies_isolation.py` asserts that the engine's import graph,
build and wheel cannot reach it.

## What is compared

Three implementations of the engine's projection: `y = x Wᵀ`, where `x` is
`(rows, in_features)`, `W` is `(out_features, in_features)` as the checkpoint
stores it, the operands are fp16, the accumulation is fp32 and the result is
rounded to fp16 once. Only the schedule differs.

| stage | what it does |
|---|---|
| **naive** | One thread per output, walking the whole reduction in global memory. A 32×32 thread block. |
| **tiled** | 32×32 tiles of `x` and `W` staged in shared memory with coalesced loads. The `W` tile is padded against bank conflicts. Still one output per thread. |
| **cuBLAS** | `cublasGemmEx`, configured exactly as `src/core_kernels/linear.cu` configures it. A test holds the two bit-identical. |

Both hand-written kernels pass the same ADR-0006 gate as the engine's
projection, at every projection of both models (`tests/test_gemm_study.py`).
Their slowness is the finding. A wrong answer would have made the timings
meaningless.

The shapes are every distinct projection of Qwen2.5-0.5B and 1.5B, read from
the model cards. Each is measured at one row (decode) and at 512 rows
(prefill). `v_proj`, `up_proj` and `o_proj` repeat `k_proj`, `gate_proj` and
`q_proj`.

## Results

Recorded in `results/`:

- `2026-09-23T044626Z-a0c0423-timing.json` holds CUDA-event timings, the
  median of 50 launches after warm-up, at the GPU's own clocks.
- `2026-09-23T045057Z-5a0572e-ncu.json` holds Nsight Compute 2025.2.1 metrics,
  profiled at commit `a0c0423` with ncu's default base-clock lock and cache
  flush.

Hardware: RTX 4050 Laptop GPU, 6141 MiB, 24 MiB L2, driver 580.178.04.

The GPU was **not exclusive**. Xorg, GNOME Shell, a browser and VS Code held it
throughout, and both records list them. The ratios below are within-run
comparisons, so they are less exposed to this than the absolute times.

### Prefill, 512 rows (event timing)

| model | projection | rows x in x out | naive µs | tiled µs | cuBLAS µs | naive ÷ cuBLAS | tiled ÷ cuBLAS |
|---|---|---|---:|---:|---:|---:|---:|
| 0.5B | q_proj | 512 x 896 x 896 | 8,899.6 | 834.6 | 43.0 | 207x | 19.4x |
| 0.5B | k_proj | 512 x 896 x 128 | 1,527.8 | 133.1 | 14.1 | 108x | 9.4x |
| 0.5B | gate_proj | 512 x 896 x 4864 | 46,909.4 | 4,066.3 | 186.4 | 252x | 21.8x |
| 0.5B | down_proj | 512 x 4864 x 896 | 46,034.1 | 4,190.9 | 210.1 | 219x | 19.9x |
| 1.5B | q_proj | 512 x 1536 x 1536 | 25,177.1 | 2,329.4 | 105.5 | 239x | 22.1x |
| 1.5B | k_proj | 512 x 1536 x 256 | 4,457.5 | 393.6 | 27.6 | 161x | 14.3x |
| 1.5B | gate_proj | 512 x 1536 x 8960 | 142,154.8 | 14,662.7 | 547.8 | 260x | 26.8x |
| 1.5B | down_proj | 512 x 8960 x 1536 | 149,332.0 | 15,583.2 | 689.1 | 217x | 22.6x |

### Decode, 1 row (event timing)

| model | projection | rows x in x out | naive µs | tiled µs | cuBLAS µs | naive ÷ cuBLAS | tiled ÷ cuBLAS |
|---|---|---|---:|---:|---:|---:|---:|
| 0.5B | q_proj | 1 x 896 x 896 | 70.7 | 87.0 | 14.1 | 5x | 6.2x |
| 0.5B | k_proj | 1 x 896 x 128 | 34.8 | 45.1 | 8.4 | 4x | 5.4x |
| 0.5B | gate_proj | 1 x 896 x 4864 | 214.0 | 421.7 | 27.6 | 8x | 15.3x |
| 0.5B | down_proj | 1 x 4864 x 896 | 467.8 | 537.6 | 24.6 | 19x | 21.9x |
| 1.5B | q_proj | 1 x 1536 x 1536 | 96.3 | 169.6 | 15.0 | 6x | 11.3x |
| 1.5B | k_proj | 1 x 1536 x 256 | 46.1 | 59.4 | 9.2 | 5x | 6.4x |
| 1.5B | gate_proj | 1 x 1536 x 8960 | 579.3 | 1,160.9 | 176.1 | 3x | 6.6x |
| 1.5B | down_proj | 1 x 8960 x 1536 | 982.0 | 1,427.5 | 157.5 | 6x | 9.1x |

### Where the time goes (Nsight Compute)

Each figure is the range across the eight shapes of its regime.

| regime | implementation | achieved occupancy | SM throughput | memory throughput | DRAM throughput | leading stall (cycles per issue) |
|---|---|---:|---:|---:|---:|---|
| prefill | naive | 65–65% | 10–12% | 79–98% | 0–1% | `lg_throttle` 172.2–175.3 |
| prefill | tiled | 67–67% | 67–90% | 67–90% | 2–11% | `mio_throttle` 21.4–21.8 |
| prefill | cublas | 8–16% | 26–49% | 24–62% | 24–62% | `math_pipe_throttle` 5.5–13.1 |
| decode | naive | 2–7% | 1–7% | 7–53% | 3–29% | `long_scoreboard` 13.2–23.1 |
| decode | tiled | 67–67% | 16–83% | 16–83% | 3–15% | `mio_throttle` 21.4–21.8 |
| decode | cublas | 13–24% | 12–36% | 29–94% | 29–94% | `long_scoreboard` 5.7–9.0 |

## The gap, and its causes

**The gap is far wider than ADR-0001 assumed.** The ADR rejected a
hand-written GEMM on the estimate that one would be 2–5x slower than cuBLAS.
Both stages here fall well outside that range. In prefill, the tiled kernel is
**9–27x** slower (19–27x for every projection but the narrow `k_proj`), and
the naive kernel is **108–260x** slower. The ADR's range belongs to a far more
developed kernel than either of these. So the measurement strengthens the
decision; it does not revise it.

### Prefill: each stage is bound by a different unit, and none of them is DRAM

- **naive: the load/store queue.** `threadIdx.x` runs along `out_features`, so
  at each step of the reduction a warp's 32 threads read 32 rows of `W`, each
  `in_features × 2` bytes from the next. Every warp-wide load becomes 32
  separate transactions. The global-memory instruction queue is full, and a
  warp waits **~174 cycles per issued instruction** on `lg_throttle`.
  Occupancy is a healthy 65%, and memory throughput reads 79–98% of peak. Yet
  DRAM sits at 0–1%, because `W` is served from L1/L2 again and again. The
  memory system is busy moving the same bytes one sector at a time. It is not
  starved for bandwidth. Achieved compute is about 0.1 TFLOP/s.
- **tiled: shared memory.** Coalesced tile loads remove the global-memory
  bottleneck and make the kernel about 10x faster. The inner loop then does
  two shared-memory loads per fused multiply-add. The kernel therefore
  saturates the MIO pipe that serves shared memory (`mio_throttle`, ~21 cycles
  per issue, SM and memory throughput both 67–90%), while the FMA units wait.
  It reaches about 1 TFLOP/s. The known next stage is register blocking: each
  thread computes a small block of outputs, so each shared-memory value feeds
  several FMAs. That stage is not in this study.
- **cuBLAS: tensor cores.** The kernels chosen are `ampere_fp16_s1688gemm` and
  `s16816gemm` tensor-core GEMMs, at 19–26 TFLOP/s on every shape but `k_proj`. The
  leading stall is `math_pipe_throttle`: the tensor pipe is the busy unit,
  which is where a GEMM should be bound. Achieved occupancy is only 8–16%.
  Each thread holds 102–254 registers of accumulator tiles, which caps
  residency. This is the research notes' warning in numbers: high occupancy
  does not mean fast. Both hand-written kernels run at 4–8x cuBLAS's occupancy
  and are 9–260x slower.

A large part of the prefill gap is therefore not schedule but hardware. The
hand-written kernels do scalar fp32 FMAs, and cuBLAS uses tensor cores. Closing
the gap would take register blocking, then tensor cores (WMMA or `mma.sync`),
then software pipelining of the loads. That is the 12–16 weeks ADR-0001 refused
to spend.

### Decode: a GEMV, and square tiles are the wrong shape for it

With one row, the projection is a matrix-vector product. It must read every
weight once and does two FLOPs per weight, so it is bound by DRAM bandwidth
whoever writes it.

- **cuBLAS** switches to a non-tensor-core GEMV kernel
  (`internal::kernel<...>`) for most shapes, and reaches **93–95% of DRAM peak**
  on the 1.5B's `gate_proj` and `down_proj`. For those shapes the gap is only
  3–9x, because there is less to win.
- **naive** launches 32×32 blocks, and with one row 31 of each block's 32 warps
  exit at once. Achieved occupancy falls to **2–7%**. The few warps left stall
  on `long_scoreboard`, waiting on global loads with nothing to hide the
  latency behind.
- **tiled is slower than naive** here. Every warp of a block stays resident,
  because all of them load tiles and reach `__syncthreads()`, so occupancy
  reads 67%. But 31 of the 32 rows of each tile are zero padding. The kernel
  does 32x the shared-memory work for the outputs it produces and stalls on
  `mio_throttle` exactly as in prefill. Its occupancy counts warps that do no
  useful work. A tile shaped for GEMM is a poor choice for GEMV, and a decode
  kernel would need a different decomposition: a warp per output feature,
  reducing along `in_features`.

**Caveat on the decode timings.** cuBLAS's event-timed decode reaches 316–355
GB/s on the 0.5B's `gate_proj` and `down_proj`. That is more than this card's
DRAM can deliver. Those weights (8.7 MB) fit in the 24 MiB L2, and a repeated
launch finds them there. In real decode each layer reads different weights and
the whole model far exceeds L2, so these small-shape decode timings are
optimistic for every implementation. ncu flushes caches before each profiled
kernel, and its DRAM figures are the ones that reflect inference.

## Reproducing

```
.venv/bin/python studies/gemm/profile.py time          # event timings, no privilege
.venv/bin/python studies/gemm/profile.py ncu-command   # prints the one sudo command
sudo ...                                               # as printed; writes results/gemm-<commit>.ncu-rep
.venv/bin/python studies/gemm/profile.py summarize studies/gemm/results/gemm-<commit>.ncu-rep
```

Performance counters need root on this machine (`RmProfilingAdminOnly: 1`).
Only `ncu` runs under sudo, and summarising the report needs no privilege. The
summary records both the commit it was summarised at and the commit that was
profiled, and it refuses to run if the measured sources changed between them.
The `.ncu-rep` files are not committed (about 30 MiB each). The JSON records
are.

These records follow the fields #13's benchmark log asks for. When #13 lands,
they should be migrated into it rather than kept beside it.
