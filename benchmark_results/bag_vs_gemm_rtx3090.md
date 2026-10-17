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
| 512 | 0.026070 | 0.021695 | 0.832173x |
| 1024 | 0.108612 | 0.093323 | 0.859241x |
| 2048 | 0.707315 | 0.673029 | 0.951525x |
| 4096 | 4.795489 | 4.820585 | 1.005233x |
| 8192 | 36.905721 | 35.785073 | 0.969635x |

Across all 20 cases, BAG latency geometric mean is **0.812665 ms** and the
speedup geometric mean is **0.921138x**. Relative to the original grouped
implementation (`0.885580 ms`), the optimized implementation is **1.089x
faster** and reduces geometric-mean latency by **8.23%**. Relative to the
earlier dependency-batched implementation (`1.538423 ms`), it is **1.893x
faster**. The per-size aggregate reaches parity with ordinary GEMM at N=4096
and 97.0% of it at N=8192; the latter ratio is sensitive to independent
large-IEEE cuBLAS timing variation.

## Grouped/batched Triton launch

The auto scheduler uses grouped execution for N<=8192 when the preallocated
workspace contains its three `(7,N/2,N/2)` buffers:

```text
materialize only eight genuinely transformed operands
    -> one 2-D Triton grid: M1 | M2 | M3 | M4 | M5 | M6 | M7
    -> N=512/1024 TF32: last product CTA decodes its spatial tile into C
    -> other buckets: independent direct decode of seven products into C
```

The product dimension is the second grid axis, so all seven independent GEMMs
are issued by one Triton launch. Products are written to disjoint batch
slices. For N=512/1024 TF32, the prepare launch also clears 32/64 scalar
counters in an otherwise unused workspace plane. Each product CTA publishes
one GPU-scope acquire/release arrival after storing its tile; the seventh CTA
loads the seven product tiles and writes all four output blocks. These 224/448
scalar atomics only elect the last completer—they do not atomically accumulate
matrix elements—so this specialization removes the third launch without
changing floating-point reduction order. All other buckets retain the
independent non-atomic decode kernel.

Workspace allocation, including all batch buffers, occurs before warmup and
timing. The timed auto path creates no streams or CUDA events. Six unchanged
quadrant operands are read directly from A/B instead of being copied into
grouped storage. The GEMM tile, BK, pipeline depth, pointwise program width,
and last-completer policy are statically bucketed by size and IEEE/TF32 mode;
offline tuning is not part of runtime execution.

The final row-major IEEE/TF32 latency is `0.0287/0.0225 ms` at N=512 and
`0.1280/0.0922 ms` at N=1024. Column-major results are `0.0317/0.0225 ms` and
`0.1280/0.0922 ms`, respectively. At N=4096, TF32 reaches
`1.047x/1.055x` for column/row-major; at N=8192 it reaches
`1.010x/1.013x`.

## Arithmetic upper bound versus measured end-to-end speedup

The zero-overhead arithmetic upper bound remains `8/7 = 1.142857x`: BAG
executes seven half-size GEMMs where a classical blocked GEMM executes eight.
It is distinct from an end-to-end measurement that includes transforms,
product storage, decoding, and launches.

An earlier independent same-GPU run gave the following finite-shape, GEMM-only
bound `T_gemm(N) / (7*T_gemm(N/2))` at N=8192: IEEE column/row-major
`1.0956x/1.0934x`, and TF32 column/row-major `1.0295x/1.0259x`. Another
same-GPU run measured IEEE column-major at
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
published test suite additionally passed all 28 BAG numerical/interface cases
and all 26 primitive cases. Coverage includes non-aligned sizes, sliced
positive strides, grouped column-major execution, a non-default caller stream,
repeated multi-stream and seven-stream calls, and concurrent primitive atomic
updates from two CUDA streams.

The complete per-case measurements are in
`benchmark_results/bag_vs_gemm_rtx3090.csv`.
