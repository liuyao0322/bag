# CUDA Strassen implementation

The implementation computes square, column-major FP32 `C <- A*B+C`, with
separate A/B/C allocations and leading dimension N. It includes the complete
seven-product algorithm, fused primitive kernels, stream dependencies and a
hybrid two-level algorithm. Custom arithmetic uses FP32 FMA; the benchmark
uses cuBLAS PEDANTIC math with TF32 disabled.

## Seven products

Split A, B and C into four N/2 by N/2 blocks. In the code's product order:

| Product | Multiplication | Updates to the existing C |
|---|---|---|
| Q1 | `(A21+A22) B11` | `C21 += Q1; C22 -= Q1` |
| Q2 | `(A11+A12) B22` | `C12 += Q2; C11 -= Q2` |
| Q3 | `(A21-A11) (B11+B12)` | `C22 += Q3` |
| Q4 | `(A12-A22) (B21+B22)` | `C11 += Q4` |
| Q5 | `A11 (B12-B22)` | `C12 += Q5; C22 += Q5` |
| Q6 | `A22 (B21-B11)` | `C11 += Q6; C21 += Q6` |
| Q7 | `(A11+A22) (B11+B22)` | `C11 += Q7; C22 += Q7` |

[engine.cuh](engine.cuh) maps these operands directly to the original matrix
blocks. Each [primitive](simt.cuh) fuses input combinations, multiplication
and output accumulation, without a global intermediate product matrix for
the one-level algorithm. `FourBatchedAtomic` submits Q1/Q2, Q3/Q4, Q5/Q6 and
Q7 in four concurrent streams. Paired products use `grid.z=2`; each CTA still
computes one product. Atomics protect overlapping C updates, and CUDA events
join all workers back into the caller stream.

## One-level API

```cpp
#include "engine.cuh"

strassen_cuda::Workspace workers;  // Create outside repeated timed calls.
strassen_cuda::one_level(
    strassen_cuda::Backend::ColumnTile8,
    strassen_cuda::Schedule::FourBatchedAtomic,
    a, b, c, n, caller_stream, workers);
// Synchronize caller_stream before reading/freeing C or destroying workers.
```

N must be positive and divisible by 8. The selected `ColumnTile8` backend
uses masked partial tiles when needed. Other backends have additional tile
constraints, checked by `valid_shape(backend, n/2)`. A workspace owns reusable
streams/events and is not reentrant; independent concurrent callers need
separate workspaces. The API preserves nonzero initial C and repeated calls.

The validated small-size kernel uses 64x64 CTA tiles, K steps of 16, 128
threads, 4x8 per-thread output tiles, shared-memory input double buffering,
deferred input sums, XOR B layout and coalesced atomic output updates. It
uses 16 KiB shared memory. `ColumnTile8` is intended for the measured small
sizes; `Huge` and hybrid configurations retain separate large-size results.

## Hybrid two-level API

```cpp
#include "hybrid.cuh"

strassen_cuda::Workspace workers;
strassen_cuda::HybridWorkspace scratch(n);
strassen_cuda::hybrid_two_level(
    strassen_cuda::Backend::Huge,
    strassen_cuda::Schedule::SevenAtomic,
    a, b, c, n, caller_stream, workers, scratch);
```

[hybrid.cuh](hybrid.cuh) prepares the two input combinations for each outer
product in compact scratch matrices, clears a scratch output, invokes inner
one-level Strassen, then merges that result into C. All seven outer products
run in caller-stream order and reuse three half-size matrices. The inner
call supports `SevenAtomic` or `FourBatchedAtomic`. Extra scratch occupies
`3*N*N` bytes. N must be divisible by 16 and the quarter-size leaf must meet
the backend's tile constraints; `Huge` requires N divisible by 512. Keep both
workspaces alive until caller-stream work has finished.

## Build and reproduce

Use CUDA 12.2 and a CUDA GPU. On the cluster, build and run inside a GPU
allocation. A clean checkout includes the original primitive dependencies
under `benchmark_results/strassen_fp32_a100_20261017/source/primitive_headers`;
the build runner automatically uses that archive when local headers are absent.

```bash
srun --jobid=<allocation> bash scripts/bench_strassen_onelevel.sh validation \
  benchmark_results/my_strassen/build --mode endtoend \
  --sizes 1280,1536,1792,2048 --backends column_tile64_k16_r4x8 \
  --schedules four_batched_atomic --rounds 7 --target-ms 300
```

For independent inputs, replay the same immutable build with a new seed:

```bash
srun --jobid=<allocation> bash scripts/replay_strassen_onelevel.sh \
  benchmark_results/my_strassen/build benchmark_results/my_strassen/seed1009 \
  --mode endtoend --sizes 1280,1536,1792,2048 \
  --backends column_tile64_k16_r4x8 --schedules four_batched_atomic \
  --seed-offset 1009 --rounds 7 --target-ms 300
```

The benchmark times all products, transformations, output updates and stream
dependencies. Allocation, input generation, initialization before an independent
trial and stream creation are outside the steady-state interval. Hybrid scratch
clears and merges remain inside it. Numerical checks validate one and three
accumulated calls: exhaustive at N<=128 and deterministic samples including
corners at larger N, with relative L2 <=3e-5 and scaled maximum error <=1e-4.

See [one-level build/API details](../../scripts/STRASSEN_ONELEVEL.md),
[hybrid reproduction](../../scripts/STRASSEN_BATCHING.md),
[small-size evidence](../../benchmark_results/strassen_primitive_20261017/REPORT.md)
and [large-size/hybrid evidence](../../benchmark_results/strassen_batching_20261017/REPORT.md).
