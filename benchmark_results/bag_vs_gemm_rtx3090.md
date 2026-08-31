# Triton BAG vs. ordinary GEMM on RTX 3090

## Scope and measurement contract

- GPU: NVIDIA GeForce RTX 3090 on cluster node `g06`.
- Data type: FP32 inputs and outputs.
- Precision modes: IEEE FP32 dot products and TF32 tensor-core dot products.
- Layouts: row-major and column-major.
- Shapes: square N=512, 1024, 2048, 4096, and 8192.
- BAG: one recursion level, seven `(N/2)^3` primitive products.
- Reference: `torch.mm(a, b, out=output)` backed by cuBLAS.
- Timed values: median CUDA-event latency from `triton.testing.do_bench`.
- Excluded from timing: tensor allocation, workspace allocation, CUDA stream
  creation, random input generation, Triton JIT, cuBLAS initialization, and
  initial GPU clock warmup.

The reported speedup is:

```text
speedup = ordinary_GEMM_median_ms / BAG_median_ms
```

Values above 1 mean BAG is faster; values below 1 mean ordinary GEMM is
faster. `bag_classical_equivalent_tflops` uses `2*N^3` logical FLOPs so the
two algorithms are directly comparable. BAG's seven products actually
execute `7/8` of the conventional GEMM multiplication work.

## Aggregate results

Each row is the geometric mean of row/column-major and IEEE/TF32 cases.

| N | BAG median (ms) | GEMM median (ms) | GEMM/BAG speedup |
|---:|---:|---:|---:|
| 512 | 0.201473 | 0.020759 | 0.103038x |
| 1024 | 0.212300 | 0.092992 | 0.438023x |
| 2048 | 0.861283 | 0.667939 | 0.775516x |
| 4096 | 6.081674 | 4.792459 | 0.788016x |
| 8192 | 38.463014 | 35.987420 | 0.935637x |

Across all 20 cases, BAG latency geometric mean is **1.538423 ms** and the
speedup geometric mean is **0.481222x**. This aggregate is dominated by the
small launch-bound shapes. At N=8192, BAG reaches 91.4-92.5% of ordinary IEEE
GEMM performance and 94.7-95.8% of ordinary TF32 GEMM performance.

## Multi-stream and dependency batches

The auto scheduler uses two dependency batches only at N=2048:

```text
prepare SA/SB
    -> batch 1: M1 | M2 | M3 -> complete SA/SB | M4   (four streams)
    -> event join
    -> batch 2: M5 | M6 | M7                          (three streams)
    -> event join
    -> inverse basis transform on the caller stream
```

M1-M4 overwrite four different transformed output quadrants. M5-M7 share
`C_psi12`, so the second batch uses the primitive's atomic accumulation mode.
The caller stream waits for the final worker events before launching the
inverse transform, preserving normal CUDA stream semantics without a host
synchronization.

The streams are created by `allocate_bag_workspace` before the timed callable.
Relative to the serial iteration, the N=2048 BAG latency geometric mean falls
from **0.922183 ms** to **0.861283 ms**, a **6.60%** improvement. Forcing the
same schedule at N=512/1024 costs roughly 0.22 ms in event fork/join overhead,
so those buckets remain serial. N>=4096 also remains serial because individual
sub-GEMMs already fill the GPU and the materialized workspace path has real
dependencies.

## Accuracy

- IEEE: maximum relative L2 error `3.140339e-06`; maximum absolute error
  `3.452301e-04`.
- TF32: maximum relative L2 error `1.095635e-03`; maximum absolute error
  `5.009460e-02`.

All 20 benchmark cases passed their configured numerical thresholds. The
published test suite additionally covers non-aligned sizes, sliced positive
strides, a non-default caller stream, repeated multi-stream batches, and the
primitive's concurrent atomic updates from two CUDA streams.

The complete per-case measurements are in
`benchmark_results/bag_vs_gemm_rtx3090.csv`.
