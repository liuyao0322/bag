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
| 512 | 0.031314 | 0.021685 | 0.692521x |
| 1024 | 0.121122 | 0.093688 | 0.773502x |
| 2048 | 0.729086 | 0.671485 | 0.920995x |
| 4096 | 5.031416 | 4.825449 | 0.959064x |
| 8192 | 39.148390 | 36.166375 | 0.923828x |

Across all 20 cases, BAG latency geometric mean is **0.885580 ms** and the
speedup geometric mean is **0.847458x**. Relative to the previous
dependency-batched implementation (`1.538423 ms`), the grouped implementation
is **1.737x faster** and reduces geometric-mean latency by **42.44%**. At
N=4096 it reaches 93.6-94.0% of ordinary IEEE GEMM performance and 98.0-98.1%
of ordinary TF32 GEMM performance.

## Grouped/batched Triton launch

The auto scheduler uses grouped execution for N<=4096 when the preallocated
workspace contains its three `(7,N/2,N/2)` buffers:

```text
materialize seven left/right pairs
    -> one 2-D Triton grid: M1 | M2 | M3 | M4 | M5 | M6 | M7
    -> direct decode of seven products into C
```

The product dimension is the second grid axis, so all seven independent GEMMs
are issued by one Triton launch. Products are written to disjoint batch slices;
the decode kernel therefore needs no atomic operations. Workspace allocation,
including all batch buffers, occurs before warmup and timing. The timed auto
path creates no streams.

The benefit is largest where launch latency previously dominated: row-major
IEEE/TF32 N=512 now takes `0.0348/0.0266 ms`, and N=1024 takes
`0.1393/0.1044 ms`. At N=2048 the TF32 column-major case reaches `0.984x` of
ordinary GEMM; at N=4096 the TF32 cases reach `0.980x/0.981x`.

## Arithmetic upper bound versus measured end-to-end speedup

The zero-overhead arithmetic upper bound remains `8/7 = 1.142857x`: BAG
executes seven half-size GEMMs where a classical blocked GEMM executes eight.
It is distinct from an end-to-end measurement that includes transforms,
product storage, decoding, and launches.

The same final-run cuBLAS measurements give the following finite-shape,
GEMM-only bound `T_gemm(N) / (7*T_gemm(N/2))` at N=8192: IEEE
column/row-major `1.1325x/1.0875x`, and TF32 column/row-major
`1.0288x/1.0373x`. The IEEE column-major result confirms a measured leaf-GEMM
bound close to 1.14x.

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

- IEEE: maximum relative L2 error `3.140339e-06`; maximum absolute error
  `3.452301e-04`.
- TF32: maximum relative L2 error `1.095635e-03`; maximum absolute error
  `5.009460e-02`.

All 20 benchmark cases passed their configured numerical thresholds. The
published test suite additionally covers non-aligned sizes, sliced positive
strides, grouped column-major execution, a non-default caller stream, repeated
multi-stream and seven-stream calls, and the primitive's concurrent atomic
updates from two CUDA streams.

The complete per-case measurements are in
`benchmark_results/bag_vs_gemm_rtx3090.csv`.
