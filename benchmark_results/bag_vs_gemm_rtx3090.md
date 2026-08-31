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
| 512 | 0.028662 | 0.021714 | 0.757575x |
| 1024 | 0.113289 | 0.094463 | 0.833819x |
| 2048 | 0.697425 | 0.670907 | 0.961977x |
| 4096 | 4.801546 | 4.819184 | 1.003673x |
| 8192 | 36.737450 | 35.773945 | 0.973773x |

Across all 20 cases, BAG latency geometric mean is **0.832332 ms** and the
speedup geometric mean is **0.901037x**. Relative to the original grouped
implementation (`0.885580 ms`), the optimized implementation is **1.064x
faster** and reduces geometric-mean latency by **6.01%**. Relative to the
earlier dependency-batched implementation (`1.538423 ms`), it is **1.849x
faster**. The per-size aggregate reaches parity with ordinary GEMM at N=4096
and 97.4% of it at N=8192.

## Grouped/batched Triton launch

The auto scheduler uses grouped execution for N<=8192 when the preallocated
workspace contains its three `(7,N/2,N/2)` buffers:

```text
materialize only eight genuinely transformed operands
    -> one 2-D Triton grid: M1 | M2 | M3 | M4 | M5 | M6 | M7
    -> direct decode of seven products into C
```

The product dimension is the second grid axis, so all seven independent GEMMs
are issued by one Triton launch. Products are written to disjoint batch slices;
the decode kernel therefore needs no atomic operations. Workspace allocation,
including all batch buffers, occurs before warmup and timing. The timed auto
path creates no streams. Six unchanged quadrant operands are read directly
from A/B instead of being copied into grouped storage. The GEMM tile, BK,
pipeline depth, and pointwise program width are statically bucketed by matrix
size and IEEE/TF32 mode; offline tuning is not part of runtime execution.

The final row-major IEEE/TF32 latency is `0.0317/0.0246 ms` at N=512 and
`0.1300/0.0983 ms` at N=1024. At N=2048, TF32 row-major reaches `1.027x` of
ordinary GEMM. At N=4096, TF32 reaches `1.053x` in both layouts; at N=8192 it
reaches `1.015x/1.012x` for column/row-major.

## Arithmetic upper bound versus measured end-to-end speedup

The zero-overhead arithmetic upper bound remains `8/7 = 1.142857x`: BAG
executes seven half-size GEMMs where a classical blocked GEMM executes eight.
It is distinct from an end-to-end measurement that includes transforms,
product storage, decoding, and launches.

The current independent final-run cuBLAS measurements give the following
finite-shape, GEMM-only bound `T_gemm(N) / (7*T_gemm(N/2))` at N=8192: IEEE
column/row-major `1.0956x/1.0934x`, and TF32 column/row-major
`1.0295x/1.0259x`. A prior same-GPU run measured IEEE column-major at
`1.1325x`, which remains the previously observed finite-shape upper bound near
1.14x; run-to-run GEMM timing variation does not replace the arithmetic
`8/7` ceiling.

Separately, one explicit hybrid end-to-end run measured seven cuBLAS leaves at
`40.444928 ms` plus `2.772990 ms` of transformations/host gaps, for
`43.217918 ms` total. Its approximately 1.02x result is one finite-shape
end-to-end observation, not a replacement for the 1.14x arithmetic upper
bound. Closing the gap requires reducing or overlapping the O(N^2) work and
matching the favorable half-size GEMM throughput.

A genuine seven-stream all-atomic Triton variant was also measured. It was
slower (`2.284818 ms` all-case geomean) because the large sub-GEMMs compete for
the same SMs, tensor cores, caches, and memory bandwidth; atomic correctness
does not create extra execution capacity. It is retained only as an explicit
diagnostic schedule.

## Accuracy

- IEEE: maximum relative L2 error `3.139858e-06`; maximum absolute error
  `3.385544e-04`.
- TF32: maximum relative L2 error `1.095635e-03`; maximum absolute error
  `5.009460e-02`.

All 20 benchmark cases passed their configured numerical thresholds. The
published test suite additionally covers non-aligned sizes, sliced positive
strides, grouped column-major execution, a non-default caller stream, repeated
multi-stream and seven-stream calls, and the primitive's concurrent atomic
updates from two CUDA streams.

The complete per-case measurements are in
`benchmark_results/bag_vs_gemm_rtx3090.csv`.
