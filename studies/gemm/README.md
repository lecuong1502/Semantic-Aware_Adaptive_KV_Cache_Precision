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

- `2026-09-23T050005Z-c1482f0-timing.json` holds CUDA-event timings: the
  median of 50 launches after 3 warm-up launches, on inputs from seed 11. They
  run at whatever clocks the driver chooses, because locking clocks needs root.
  The record states the clocks before and after the run. Before is the idle
  P-state; after is 2640 MHz at P0.
- `2026-09-23T045057Z-5a0572e-ncu.json` holds Nsight Compute 2025.2.1 metrics,
  profiled at commit `a0c0423` with ncu's default base-clock lock and cache
  flush. The kernels are unchanged between that commit and the timing's: the
  change between them is a comment, and one launcher template replacing two
  identical launchers.
- `2026-09-23T044626Z-a0c0423-timing.json` is an earlier timing at `a0c0423`.
  It does not record its warm-up count or seed and is superseded. It is kept
  because results are appended, never rewritten.

Every table and every figure below comes from the timing record and the ncu
record above. TFLOP/s and GB/s figures are derived from the event timings.
ncu's own durations are at locked base clocks and give lower rates, for
example 13–17 TFLOP/s for cuBLAS in prefill against 19–26 from the events, on
every projection but `k_proj`.

Hardware: RTX 4050 Laptop GPU, 6141 MiB, 24 MiB L2, driver 580.178.04.

The GPU was **not exclusive**. Xorg, GNOME Shell, a browser and VS Code held it
throughout, and both records list them. The ratios below are within-run
comparisons, so they are less exposed to this than the absolute times.

### Prefill, 512 rows (event timing)

| model | projection | rows x in x out | naive µs | tiled µs | cuBLAS µs | naive ÷ cuBLAS | tiled ÷ cuBLAS |
|---|---|---|---:|---:|---:|---:|---:|
| 0.5B | q_proj | 512 x 896 x 896 | 8,917.2 | 835.6 | 43.7 | 204x | 19.1x |
| 0.5B | k_proj | 512 x 896 x 128 | 1,528.8 | 133.8 | 16.5 | 93x | 8.1x |
| 0.5B | gate_proj | 512 x 896 x 4864 | 47,134.4 | 4,076.4 | 186.5 | 253x | 21.9x |
| 0.5B | down_proj | 512 x 4864 x 896 | 47,366.1 | 4,220.9 | 190.8 | 248x | 22.1x |
| 1.5B | q_proj | 512 x 1536 x 1536 | 25,761.1 | 2,262.0 | 105.3 | 245x | 21.5x |
| 1.5B | k_proj | 512 x 1536 x 256 | 4,456.4 | 394.2 | 27.6 | 161x | 14.3x |
| 1.5B | gate_proj | 512 x 1536 x 8960 | 147,272.9 | 15,680.7 | 550.6 | 267x | 28.5x |
| 1.5B | down_proj | 512 x 8960 x 1536 | 148,920.3 | 15,391.7 | 675.1 | 221x | 22.8x |

### Decode, 1 row (event timing)

| model | projection | rows x in x out | naive µs | tiled µs | cuBLAS µs | naive ÷ cuBLAS | tiled ÷ cuBLAS |
|---|---|---|---:|---:|---:|---:|---:|
| 0.5B | q_proj | 1 x 896 x 896 | 74.8 | 87.8 | 14.3 | 5x | 6.1x |
| 0.5B | k_proj | 1 x 896 x 128 | 35.7 | 45.9 | 9.0 | 4x | 5.1x |
| 0.5B | gate_proj | 1 x 896 x 4864 | 214.0 | 420.9 | 27.6 | 8x | 15.2x |
| 0.5B | down_proj | 1 x 4864 x 896 | 466.9 | 537.6 | 25.6 | 18x | 21.0x |
| 1.5B | q_proj | 1 x 1536 x 1536 | 97.3 | 170.0 | 15.4 | 6x | 11.1x |
| 1.5B | k_proj | 1 x 1536 x 256 | 46.0 | 59.4 | 9.2 | 5x | 6.4x |
| 1.5B | gate_proj | 1 x 1536 x 8960 | 572.4 | 1,144.8 | 154.6 | 4x | 7.4x |
| 1.5B | down_proj | 1 x 8960 x 1536 | 978.7 | 1,421.7 | 148.5 | 7x | 9.6x |

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
**8–28x** slower (19–28x for every projection but the narrow `k_proj`), and
the naive kernel is **93–267x** slower. The ADR's range belongs to a far more
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
  and are 8–267x slower.

A large part of the prefill gap is therefore not schedule but hardware. The
hand-written kernels do scalar fp32 FMAs, and cuBLAS uses tensor cores. Closing
the gap would take register blocking, then tensor cores (WMMA or `mma.sync`),
then software pipelining of the loads. That is the 12–16 weeks ADR-0001 refused
to spend.

### Decode: a GEMV, and square tiles are the wrong shape for it

With one row, the projection is a matrix-vector product. It must read every
weight once and does two FLOPs per weight, so DRAM bandwidth is its ceiling
whoever writes it. Only cuBLAS reaches that ceiling. The two hand-written
kernels stop well short of it (DRAM at 3–29%), for reasons of their own.

- **cuBLAS** switches to a non-tensor-core GEMV kernel
  (`internal::kernel<...>`) for most shapes, and reaches **93–95% of DRAM peak**
  on the 1.5B's `gate_proj` and `down_proj`. For those shapes the gap is only
  4–10x, because there is less to win.
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

**Caveat on the decode timings.** cuBLAS's event-timed decode reaches 315–340
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
The `.ncu-rep` files are not committed (about 30 MiB each, 32 MB for this
study's). The JSON records
are.

These records follow the fields #13's benchmark log asks for. When #13 lands,
they should be migrated into it rather than kept beside it.
