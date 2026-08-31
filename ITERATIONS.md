# Triton BAG GEMM Optimization Log

## Summary

| Version | Direction | Signal result | Correct | Decision |
|---|---|---:|---:|---|
| baseline | Current grouped tiled `tl.dot` kernel | 2.185636 ms geomean; 0.365529x vs unfused | yes | baseline |
| iter-1 | Reinterpret column-major as transposed row-major GEMM | 1.358151 ms geomean; 0.588348x vs unfused | yes | keep; new best |

## Baseline

- Direction: Current implementation before GEMM-body optimization.
- Hypothesis: Column-major operands are mapped poorly for Triton's preferred
  row-major dot operand layouts; fixed shape buckets also leave throughput on
  the table.
- Result: fused geomean 2.185636 ms and speedup geomean 0.365529x on the
  six-shape column-major IEEE signal set. Individual fused medians were
  0.5796/2.1089/12.9966 ms for square 1024/2048/4096, 1.9471 ms tall,
  2.0449 ms wide, and 1.7234 ms large-K. `CORRECT=True`.
- Analysis: The baseline is 2.74x slower than the preallocated unfused path by
  geometric mean. Prior row-major measurements were materially faster, making
  physical-layout reinterpretation the highest-confidence first direction.

## Iteration 1

- Direction: Detect conventional column-major views and compute the equivalent
  transposed problem `C^T=(V+epsilon*W)^T @ (X+delta*Y)^T`, swapping operands,
  coefficients, dimensions, and strides without moving data.
- Hypothesis: Presenting physically row-major operands to `tl.dot` will improve
  global-load vectorization and dot operand lowering while preserving exact
  logical addresses and arithmetic.
- Result: fused geomean fell from 2.185636 to 1.358151 ms, a 1.609x kernel
  speedup (37.9% lower latency). Relative-to-unfused geomean improved from
  0.365529x to 0.588348x. Square 4096 improved from 12.9966 to 8.5699 ms.
  `CORRECT=True` across all six signal shapes.
- Analysis: The large, broad improvement confirms physical layout was the
  dominant first-order issue. The kernel still trails cuBLAS, so the next
  direction is tile/warp/pipeline tuning on the now-canonical row-major GEMM.
