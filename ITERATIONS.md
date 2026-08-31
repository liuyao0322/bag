# Triton BAG GEMM Optimization Log

## Summary

| Version | Direction | Signal result | Correct | Decision |
|---|---|---:|---:|---|
| baseline | Current grouped tiled `tl.dot` kernel | 2.185636 ms geomean; 0.365529x vs unfused | yes | baseline |
| iter-1 | Reinterpret column-major as transposed row-major GEMM | 1.358151 ms geomean; 0.588348x vs unfused | yes | keep; new best |
| iter-2 | Rectangular `128x64x32`, 8-warps, 2-stage main tiles | 1.422233 ms geomean; 0.560589x vs unfused | yes | reject; regression |
| iter-3 | Size-specific rectangular standard tile + square large tile | 1.309330 ms geomean; 0.609487x vs unfused | yes | keep; new best |
| iter-4 | Reduce standard rectangular tile from 8 to 4 warps | 1.370751 ms geomean; 0.583013x vs unfused | yes | reject; regression |

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

## Iteration 2

- Direction: Replace the standard `64x64x32` tile and the large
  `128x128x16` four-input tile with `128x64x32`, 8 warps, and 2 pipeline
  stages to restore K=32 while reducing per-thread accumulators.
- Hypothesis: Higher A reuse and fewer K-loop iterations would outweigh the
  extra output programs, while two stages would fit the four input streams in
  shared memory.
- Result: fused geomean regressed from 1.358151 to 1.422233 ms (4.7% slower),
  and speedup geomean fell to 0.560589x. Square-1024 improved from 0.3308 to
  0.2857 ms and large-K from 0.8561 to 0.8172 ms, but square-2048 regressed to
  1.5974 ms and square-4096 to 11.7540 ms. `CORRECT=True`.
- Analysis: The rectangular K=32 tile is useful for 1024-scale and large-K
  problems, but loses too much N reuse and/or pipeline depth on large square
  matrices. Keep it as a size-specific candidate rather than a global policy.

## Iteration 3

- Direction: Retain `128x64x32`, 8-warps, 2-stage tiles only below the large
  square threshold, and restore `128x128x16`, 8-warps, 3-stage tiles when both
  M and N are at least 2048.
- Hypothesis: A size-specific policy can preserve iteration 2's 1024/large-K
  gains without its large-square regression.
- Result: fused geomean improved from the iteration-1 best of 1.358151 to
  1.309330 ms (3.6% lower, 1.037x faster), and speedup geomean reached
  0.609487x. Relative to baseline, the kernel is now 1.669x faster.
  `CORRECT=True` on all signal shapes.
- Analysis: The result validates separate throughput regimes. Large squares
  need the reuse of 128x128 tiles despite K=16, while 1024 and large-K benefit
  from more programs, fewer K iterations, and lower accumulator pressure.

## Iteration 4

- Direction: Keep the iteration-3 tiles but reduce the standard
  `128x64x32` configuration from 8 warps to 4, increasing per-thread ILP.
- Hypothesis: The conventional Triton four-warp mapping might schedule the
  IEEE dot loop more efficiently than eight warps.
- Result: fused geomean regressed from 1.309330 to 1.370751 ms (4.7% slower).
  Square-1024 rose from 0.2857 to 0.3420 ms and large-K from 0.8136 to
  0.8755 ms; unaffected buckets moved only within normal noise.
  `CORRECT=True`.
- Analysis: Four warps likely create excessive per-thread accumulator/register
  pressure for four-input fusion. Eight warps are decisively better for this
  rectangular tile and should be retained.
