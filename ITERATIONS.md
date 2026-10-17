# Triton BAG GEMM Optimization Log

## Summary

| Version | Direction | Signal result | Correct | Decision |
|---|---|---:|---:|---|
| baseline | Current grouped tiled `tl.dot` kernel | 2.185636 ms geomean; 0.365529x vs unfused | yes | baseline |
| iter-1 | Reinterpret column-major as transposed row-major GEMM | 1.358151 ms geomean; 0.588348x vs unfused | yes | keep; new best |
| iter-2 | Rectangular `128x64x32`, 8-warps, 2-stage main tiles | 1.422233 ms geomean; 0.560589x vs unfused | yes | reject; regression |
| iter-3 | Size-specific rectangular standard tile + square large tile | 1.309330 ms geomean; 0.609487x vs unfused | yes | keep; new best |
| iter-4 | Reduce standard rectangular tile from 8 to 4 warps | 1.370751 ms geomean; 0.583013x vs unfused | yes | reject; regression |
| iter-5 | Expand tall/wide fused tiles to 64 on the short edge | 1.158568 ms geomean; 0.692941x vs unfused | yes | keep; new best |
| iter-6 | Reduce large-square pipeline from 3 to 2 stages | 1.205704 ms geomean; 0.661972x vs unfused | yes | reject; regression |
| iter-7 | Compile-time unmasked fast path for exact tiles | benchmark failed to compile | no result | fix API usage and retry |
| iter-8 | Corrected exact-tile unmasked load path | 1.175801 ms geomean; 0.679223x vs unfused | yes | reject; no gain |
| iter-9 | Canonical-contiguous address specialization | 1.155336 ms geomean; 0.692094x vs unfused | yes | reject; below 3% threshold |
| iter-10 | Use 128x128x16 tiles for large tall/wide GEMMs | 1.091716 ms geomean; 0.733427x vs unfused | yes | keep; new best |
| iter-11 | Extend fused 128x128x16 tiles down to short edge 1024 | 0.991254 ms geomean; 0.807483x vs unfused | yes | keep; new best |
| iter-12 | Test 128x128x8 with four pipeline stages | benchmark failed to compile | no result | reject; unsupported K tile |
| iter-13 | Reduce fused 128x128 program grouping from 8 to 4 | 0.991427 ms geomean; 0.807330x vs unfused | yes | reject; no gain |
| iter-14 | Increase fused 128x128 CTA from 8 to 16 warps | 0.958616 ms geomean; 0.834934x vs unfused | yes | keep; new best |
| iter-15 | Increase fused 128x128 CTA from 16 to 32 warps | 1.229815 ms geomean; 0.649229x vs unfused | yes | reject; major regression |
| final | Restore and independently remeasure iteration 14 | 0.963284 ms geomean; 0.831807x vs unfused | yes | final |
| baseline-512 | Existing 128x64x32 bucket on four 512 modes | 0.077348 ms geomean; 0.534826x vs unfused | yes | 512 phase baseline |
| iter-16 | Compact 32x32x16 tile for medium-small fused GEMMs | 0.046449 ms geomean; 0.871591x vs unfused | yes | keep; new 512 best |
| iter-17 | Increase compact tile BK from 16 to 32 with two stages | 0.047569 ms geomean; 0.865065x vs unfused | yes | reject; regression |
| iter-18 | Increase compact 32x32 tile from 4 to 8 warps | 0.060066 ms geomean; 0.685031x vs unfused | yes | reject; major regression |
| iter-19 | Deepen compact 32x32x16 pipeline from 3 to 4 stages | 0.053670 ms geomean; 0.782033x vs unfused | yes | reject; regression |
| iter-20 | Reduce compact 32x32 tile from 4 to 2 warps | 0.052497 ms geomean; 0.793737x vs unfused | yes | reject; regression |
| iter-21 | Test compact rectangular 32x64x16 tile | 0.047456 ms geomean; 0.870281x vs unfused | yes | reject; below best |
| iter-22 | Reduce compact-tile GROUP_M from 8 to 4 | 0.046449 ms geomean; 0.885925x vs unfused | yes | tie; retain GROUP_M=8 |
| iter-23 | Test 16x64x16 I/O-oriented tile | 0.058420 ms geomean; 0.718442x vs unfused | yes | reject; major regression |
| iter-24 | Reduce best compact-tile pipeline from 3 to 2 stages | 0.049067 ms geomean; 0.855399x vs unfused | yes | reject; regression |
| final-512 | Restore and remeasure iteration 16 | 0.047132 ms geomean; 0.890515x vs unfused | yes | final 512 winner |
| iter-25 | One-level BAG with two quarter-block transforms and seven fused primitives | 2.240755 ms geomean; 0.330219x vs cuBLAS GEMM | yes | baseline for end-to-end BAG phase |
| iter-26 | Non-atomic output accumulation for same-stream BAG calls | 2.069139 ms geomean; 0.356438x vs cuBLAS GEMM | yes | keep; new end-to-end best |
| iter-27 | Materialize terminal four-term inputs to remove right-combination IEEE cliff | 1.853898 ms geomean; 0.397718x vs cuBLAS GEMM | yes | keep; new end-to-end best |
| iter-28 | Traverse pointwise transforms in physical contiguous order | 1.580899 ms geomean; 0.469512x vs cuBLAS GEMM | yes | keep; new end-to-end best |
| iter-29 | Reuse workspace to materialize M1/M2 inputs for N>=4096 | 1.552825 ms geomean; 0.477185x vs cuBLAS GEMM | yes | keep; new end-to-end best |
| iter-30 | Two dependency-batched multi-stream waves for N=2048 | 1.538423 ms geomean; 0.481222x vs cuBLAS GEMM | yes | keep; targeted N=2048 win |

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

## Iteration 5 — expand tall/wide tiles

- Change: restored the iteration-3 standard path (`128x64x32`, 8 warps, 2 stages) and expanded aspect-ratio-specialized tiles from `128x32`/`32x128` to `128x64`/`64x128`, using 8 warps for the fused-input path.
- Hypothesis: doubling work along the short output dimension keeps the per-thread accumulator footprint roughly constant while increasing operand reuse and amortizing launch/scheduling overhead for tall and wide GEMMs.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **1.158568 ms**.
- Geometric-mean speedup vs unfused: **0.692941x**.
- Change vs baseline (2.185636 ms): **1.8868x faster**.
- Change vs previous best, iteration 3 (1.309330 ms): **11.52% faster**.
- Per-shape fused latency (ms): square-1024 `0.2857`, square-2048 `1.3302`, square-4096 `8.5791`, tall `0.9610`, wide `0.9452`, large-k `0.8167`.
- Outcome: **new best**; the improvement is concentrated in tall and wide shapes, while square and large-K performance remains effectively unchanged.
- Trajectory: `trajectory/iter-5-20260831-141626`.

## Iteration 6 — reduce large-square pipeline depth

- Change: reduced only the fused large-square `128x128x16` path from 3 pipeline stages to 2.
- Hypothesis: a smaller shared-memory footprint could improve residency or reduce pipeline overhead for the large-square GEMM body.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **1.205704 ms**.
- Geometric-mean speedup vs unfused: **0.661972x**.
- Change vs iteration-5 best (1.158568 ms): **4.07% slower**.
- Per-shape fused latency (ms): square-1024 `0.2857`, square-2048 `1.5493`, square-4096 `9.5898`, tall `0.9503`, wide `0.9344`, large-k `0.8151`.
- Outcome: **reject**. The 2048 and 4096 square cases regress by about 16% and 12%, respectively, proving that the third pipeline stage is needed to hide the four fused input streams' load latency.
- Trajectory: `trajectory/iter-6-20260831-141734`.

## Iteration 7 — exact-tile unmasked fast path

- Change: restored the iteration-5 large-square pipeline and added `EVEN_M`, `EVEN_N`, and `EVEN_K` compile-time flags intended to remove boundary predicates on exact tiles.
- Hypothesis: eliminating masks from the GEMM K loop and output atomics for aligned benchmark shapes would reduce predicate/address-protection overhead while preserving the masked fallback for arbitrary dimensions.
- Correctness/performance: **not measured** because Triton compilation failed before launch.
- Failure: Triton 3.6 rejects `tl.load(..., mask=None, other=0.0)`; `other` must be omitted when no mask is supplied.
- Outcome: implementation/API failure, not a performance verdict. Retain the direction and retry with explicitly separated masked and unmasked `tl.load` calls.
- Trajectory: `trajectory/iter-7-20260831-141839` was not created because the benchmark exited before its snapshot step.

## Iteration 8 — corrected exact-tile unmasked path

- Change: separated masked and unmasked input loads into compile-time branches, omitting both `mask` and `other` for exact tiles. The iteration-5 three-stage large-square path was restored.
- Hypothesis: removing all input boundary predicates for aligned signal shapes would shorten the hot K loop.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **1.175801 ms**.
- Geometric-mean speedup vs unfused: **0.679223x**.
- Change vs iteration-5 best (1.158568 ms): **1.49% slower**.
- Per-shape fused latency (ms): square-1024 `0.2928`, square-2048 `1.3517`, square-4096 `8.7388`, tall `0.9431`, wide `0.9544`, large-k `0.8489`.
- Outcome: **reject**. The compiler already handles the regular masks effectively; explicit branching increases specialization/code complexity without a repeatable speedup.
- Trajectory: `trajectory/iter-8-20260831-141852`.

## Iteration 9 — canonical-contiguous address specialization

- Change: removed the iteration-8 exact-tile branches and specialized canonical row-major physical views so inner unit strides do not participate in address multiplications. Arbitrary positive strides retain the general formula.
- Hypothesis: simpler pointer expressions for the column-major-transposed benchmark path would lower integer address-generation overhead inside the K loop.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **1.155336 ms**.
- Geometric-mean speedup vs unfused: **0.692094x**.
- Change vs iteration-5 best (1.158568 ms): **0.28% faster**, below the required 3% meaningful-improvement threshold.
- Per-shape fused latency (ms): square-1024 `0.2857`, square-2048 `1.3353`, square-4096 `8.5606`, tall `0.9523`, wide `0.9370`, large-k `0.8161`.
- Outcome: **reject as inconclusive/no material gain**. Runtime strides are evidently already simplified or amortized by Triton's compiler, so the extra specialization is not justified.
- Trajectory: `trajectory/iter-9-20260831-142018`.

## Iteration 10 — square tiles for large tall/wide GEMMs

- Change: returned to the iteration-5 general address path and selected the large-square `128x128x16`, 8-warp, 3-stage GEMM body for fused tall/wide problems whose short edge is at least 512.
- Hypothesis: the additional reuse from doubling the short tile edge again would outweigh the lower K tile and increased accumulator footprint.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **1.091716 ms**.
- Geometric-mean speedup vs unfused: **0.733427x**.
- Change vs iteration-5 best (1.158568 ms): **5.77% faster**.
- Change vs baseline (2.185636 ms): **2.0020x faster**.
- Per-shape fused latency (ms): square-1024 `0.2847`, square-2048 `1.3384`, square-4096 `8.5340`, tall `0.7987`, wide `0.8028`, large-k `0.8120`.
- Outcome: **new best**. Tall and wide latency improve another 16.9% and 15.1% over iteration 5, respectively, without degrading the other shape buckets.
- Trajectory: `trajectory/iter-10-20260831-142124`.

## Iteration 11 — extend square fused tiles to 1024

- Change: lowered the short-edge threshold for the fused `128x128x16`, 8-warp, 3-stage path from 2048 to 1024; the no-input-combination policy remains unchanged.
- Hypothesis: the reuse and pipeline advantages observed on large square/aspect-ratio shapes should also benefit 1024-square and 1024x1024 large-K GEMMs.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.991254 ms**.
- Geometric-mean speedup vs unfused: **0.807483x**.
- Change vs iteration-10 best (1.091716 ms): **9.20% faster**.
- Change vs baseline (2.185636 ms): **2.2050x faster**.
- Per-shape fused latency (ms): square-1024 `0.2212`, square-2048 `1.3517`, square-4096 `8.5760`, tall `0.7941`, wide `0.7962`, large-k `0.5852`.
- Outcome: **new best**. Square-1024 improves 22.3%, while large-K improves 27.9% and reaches 0.945x of the strong unfused path.
- Trajectory: `trajectory/iter-11-20260831-142155`.

## Iteration 12 — smaller K tile with deeper pipeline

- Change: changed fused 128x128 configurations from `BK=16, stages=3` to `BK=8, stages=4`.
- Hypothesis: reducing per-stage shared-memory data and increasing prefetch depth could improve latency hiding despite doubling K-loop iterations.
- Correctness/performance: **not measured** because Triton compilation failed.
- Failure: Triton 3.6's NVIDIA `tl.dot` lowering requires the K dimension to be at least 16; `BK=8` triggers `Input shapes should have ... K >= 16`.
- Outcome: **reject**. BK=16 is the minimum supported dot tile on this stack, so this direction is exhausted.
- Trajectory: no snapshot was created because compilation failed before the benchmark snapshot step.

## Iteration 13 — smaller program grouping

- Change: restored the iteration-11 `128x128x16`, 8-warp, 3-stage body and reduced `GROUP_M` from 8 to 4 for its fused shape buckets.
- Hypothesis: shorter M groups might improve L2 locality or tail scheduling across square and aspect-ratio problems.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.991427 ms**.
- Geometric-mean speedup vs unfused: **0.807330x**.
- Change vs iteration-11 best (0.991254 ms): **0.02% slower**.
- Per-shape fused latency (ms): square-1024 `0.2212`, square-2048 `1.3558`, square-4096 `8.3835`, tall `0.8018`, wide `0.7987`, large-k `0.5898`.
- Outcome: **reject/no material difference**. The current shapes are not sensitive to GROUP_M=4 versus 8 at the required 3% threshold.
- Trajectory: `trajectory/iter-13-20260831-142252`.

## Iteration 14 — increase per-CTA warps

- Change: restored `GROUP_M=8` and increased fused `128x128x16` configurations from 8 to 16 warps while retaining three pipeline stages.
- Hypothesis: the four-input, three-stage tile likely permits only one CTA per SM due to shared-memory use; 16 warps should expose more independent work inside that CTA and reduce per-thread accumulator pressure.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.958616 ms**.
- Geometric-mean speedup vs unfused: **0.834934x**.
- Change vs iteration-11 best (0.991254 ms): **3.29% faster**.
- Change vs baseline (2.185636 ms): **2.2800x faster**.
- Per-shape fused latency (ms): square-1024 `0.2007`, square-2048 `1.3332`, square-4096 `8.5699`, tall `0.7465`, wide `0.7567`, large-k `0.5990`.
- Outcome: **new best**. The added warps materially improve 1024-square and aspect-ratio cases; large-K regresses slightly, suggesting a separate K-dominant policy may be useful.
- Trajectory: `trajectory/iter-14-20260831-142323`.

## Iteration 15 — maximum-size CTA

- Change: increased fused `128x128x16` configurations from 16 to 32 warps (1024 threads per CTA).
- Hypothesis: maximizing active warps in the one-resident-CTA regime might further hide memory and instruction latency.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **1.229815 ms**.
- Geometric-mean speedup vs unfused: **0.649229x**.
- Change vs iteration-14 best (0.958616 ms): **28.3% slower**.
- Per-shape fused latency (ms): square-1024 `0.2478`, square-2048 `1.7326`, square-4096 `11.4739`, tall `0.9385`, wide `0.9339`, large-k `0.8013`.
- Outcome: **reject**. A full 1024-thread CTA introduces substantial scheduling/communication overhead; 16 warps is the optimum tested point on this axis.
- Trajectory: `trajectory/iter-15-20260831-142356`.

## Final signal verification

- Restored source: iteration 14, commit `9f3c931`.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.963284 ms**.
- Geometric-mean speedup vs unfused: **0.831807x**.
- Change vs baseline (2.185636 ms): **2.269x faster / 55.93% lower latency**.
- Per-shape fused latency (ms): square-1024 `0.2007`, square-2048 `1.3312`, square-4096 `8.6148`, tall `0.7506`, wide `0.7629`, large-k `0.6062`.
- The final rerun is within 0.49% of iteration 14's signal geomean, confirming the selected implementation.
- Relative to the strong preallocated unfused sequence, the fused primitive reaches 0.832x geometric-mean speedup (about 20.2% slower); this remaining gap includes two required atomic output accumulations and IEEE FP32 dot execution.
- Trajectory: `trajectory/final-20260831-142428`.

## Square-512 optimization phase

## End-to-end BAG phase

### Iteration 25 — minimal-workspace one-level BAG baseline

- Change: implemented one BAG recursion level using two materialized quarter
  blocks (`SA=A12-A21+A22`, `SB=B12-B21+B22`), seven fused primitive calls,
  direct transformed-output accumulation, and an in-place inverse transform.
- Hypothesis: reducing GEMM work from eight to seven half-size products while
  avoiding full transformed A/B copies could offset transform and launch
  overhead at sufficiently large N.
- Correctness: 28 BAG cases passed across IEEE/TF32 and row-major,
  column-major, and sliced layouts; the original primitive's 26-case suite
  also passed. The benchmark reported `CORRECT=True` on all 20 signal cases.
- BAG geometric-mean latency: **2.240755 ms**.
- Geometric-mean speedup vs ordinary preallocated cuBLAS GEMM: **0.330219x**.
- Large row-major TF32 speedups were `0.681x`, `0.710x`, and `0.808x` at
  N=2048/4096/8192; large row-major IEEE remained `0.387x` to `0.389x`.
- Analysis: the 12.5% reduction in multiplication count is visible only at
  the largest TF32 bucket. Custom sub-GEMM throughput and six atomic output
  updates dominate; small N is additionally launch-bound.
- Outcome: correct end-to-end baseline, not performance winner. Test a
  non-atomic sequential accumulation specialization while preserving the
  public primitive's atomic default.
- Trajectory: `trajectory/iter-25-20260831-145757`.

### Iteration 26 — same-stream non-atomic accumulation

- Change: retained atomic accumulation as the public primitive default, but
  specialized the seven serial BAG calls to load/add/store output tiles.
- Hypothesis: BAG launches are ordered on one CUDA stream, so atomics provide
  no correctness benefit internally and add avoidable output-update cost.
- Correctness: quick BAG regression and all 26 primitive tests passed,
  including the unchanged default atomic updates from two CUDA streams. The
  full benchmark reported `CORRECT=True` on 20 cases.
- BAG geometric-mean latency: **2.069139 ms**, 7.66% lower than iteration 25.
- Geometric-mean speedup vs cuBLAS GEMM: **0.356438x**, up from 0.330219x.
- Row-major TF32 reached `0.750x`, `0.760x`, and `0.842x` at
  N=2048/4096/8192. Row-major IEEE improved at N=1024/2048 but remained
  about `0.38x` for N=4096/8192.
- Analysis: removing atomics materially helps small and TF32 buckets, but the
  large IEEE plateau confirms the custom GEMM bodies are now dominant.
- Outcome: keep as the new best. Next replace only the plain overwrite M3/M4
  products with cuBLAS while retaining Triton primitives for fused products.
- Trajectory: `trajectory/iter-26-20260831-150040`.

### Iteration 27 — materialize terminal four-term combinations

- Change: after M3 consumes the three-term SA/SB buffers, a linear Triton
  kernel subtracts A11/B11 in place. M6 and M7 then consume these completed
  four-term inputs without fused GEMM-loop subtraction.
- Hypothesis: event profiling showed a physical-right-only IEEE combination
  taking about 73.6 ms versus 6-8 ms for every peer at half-size 4096; moving
  that subtraction to a bandwidth-bound pass should remove the lowering cliff.
- Correctness: the 18-case quick BAG suite passed and the full benchmark
  reported `CORRECT=True` across 20 cases with unchanged errors.
- BAG geometric-mean latency: **1.853898 ms**, 10.40% below iteration 26 and
  17.26% below the end-to-end baseline.
- Geometric-mean speedup vs cuBLAS GEMM: **0.397718x**.
- At N=8192, IEEE improved from `115.3761/128.9400 ms` to
  `52.3080/66.8774 ms` for row/column-major; row-major speedup reached 0.850x.
  Row-major TF32 reached 0.888x at N=8192.
- Analysis: the pathological right-combination code generation is eliminated.
  Column-major pointwise transforms now account for roughly 20 ms at N=8192,
  versus under 2 ms row-major, exposing the next dominant layout issue.
- Outcome: keep as new best. Make pointwise kernels traverse physical
  contiguous order for column-major storage.
- Trajectory: `trajectory/iter-27-20260831-150336`.

### Iteration 28 — layout-aware pointwise traversal

- Change: specialized the preparation, four-term completion, and inverse
  kernels so column-major tensors map adjacent lanes along physical rows;
  row-major and generic-stride traversal is unchanged.
- Hypothesis: component events attributed about 20.25 ms at N=8192
  column-major to pointwise/host gaps versus 1.83 ms row-major, indicating
  warp-wide leading-dimension accesses instead of coalesced accesses.
- Correctness: the 18-case quick BAG suite and all 20 benchmark cases passed.
- BAG geometric-mean latency: **1.580899 ms**, 14.72% below iteration 27 and
  29.45% below the end-to-end baseline.
- Geometric-mean speedup vs cuBLAS GEMM: **0.469512x**.
- N=8192 column-major improved to `51.8164 ms` IEEE and `31.9729 ms` TF32;
  row/column speedups are now symmetric at roughly 0.87x IEEE and 0.90x TF32.
- Analysis: physical-order mapping eliminates the column-major bandwidth
  pathology. Remaining large-N gap is dominated by the seven GEMM calls;
  N=512/1024 remain launch-bound because one BAG level creates ten launches.
- Outcome: keep as new best. Investigate fusing the small pointwise kernels
  and size-gating BAG so launch-bound buckets use ordinary GEMM.
- Trajectory: `trajectory/iter-28-20260831-150450`.

### Iteration 29 — materialize large M1/M2 inputs

- Change: for N>=4096, reordered the independent BAG products so M6/M7
  consume the four-term workspace first, then reused those two buffers to
  materialize M1 and M2 input pairs. Their O(N^3) loops now use the faster
  plain-product primitive; smaller N retains fused pairs and fewer launches.
- Hypothesis: two coalesced O(N^2) passes should cost less than repeatedly
  loading and combining four inputs across two large O(N^3) GEMM loops.
- Correctness: a large-size pre-screen and all 20 formal benchmark cases
  passed with unchanged IEEE/TF32 error metrics.
- BAG geometric-mean latency: **1.552825 ms**, 1.78% below iteration 28 and
  30.70% below iteration 25.
- Geometric-mean speedup vs cuBLAS GEMM: **0.477185x**.
- N=8192 reached `0.924x/0.936x` of IEEE GEMM and `0.958x/0.947x` of TF32
  GEMM for column/row-major respectively. N=4096 improved to about 0.78x IEEE
  and 0.80x TF32.
- Analysis: materialization is increasingly favorable as N grows, bringing
  the largest bucket within 4-8% of cuBLAS while retaining seven products.
  Small buckets remain dominated by fixed launch overhead.
- Outcome: keep as new best. Preserve the honest all-BAG benchmark rather
  than hiding small sizes behind a GEMM fallback.
- Trajectory: `trajectory/iter-29-20260831-150740`.

### Iteration 30 — dependency-batched multi-stream launch

- Change: preallocated four worker streams in `BagWorkspace`. For N=2048,
  batch 1 launches M1/M2/M3/M4 on four streams with distinct overwrite
  outputs; batch 2 launches M5/M6/M7 on three streams using atomic updates for
  the shared C-psi-12 block. Caller/worker dependencies use CUDA events and
  the inverse transform is joined back onto the caller's current stream.
- Measurement boundary: stream objects and workspace are created before the
  benchmark callable. Timed execution includes real event fork/join and GPU
  waits, but excludes stream creation, allocation, JIT, and library startup.
- Pre-screen: forcing multi-stream at N=512/1024 regressed latency from about
  0.20-0.24 ms to 0.43-0.45 ms due event overhead; therefore auto scheduling
  enables it only at N=2048 and keeps serial iter-29 scheduling elsewhere.
- Correctness: quick IEEE/TF32 tests passed across row/column/sliced layouts,
  including two consecutive batched calls from a non-default caller stream;
  all 20 formal benchmark cases reported `CORRECT=True`.
- BAG geometric-mean latency: **1.538423 ms**, 0.93% below iteration 29 and
  31.34% below the end-to-end baseline.
- Geometric-mean speedup vs cuBLAS GEMM: **0.481222x**.
- N=2048 latency was `1.0373/1.0445 ms` IEEE and `0.7199/0.7055 ms` TF32 for
  column/row-major. Relative to the serial pre-screen this is approximately a
  5-8% targeted improvement; other size buckets intentionally remain serial.
- Analysis: multi-stream helps only where a sub-GEMM underfills the RTX 3090
  enough to amortize seven event dependencies. It cannot reduce launch count,
  so smaller buckets require graph/grouped submission rather than more streams.
- Outcome: keep the auto-gated path because it materially improves N=2048
  without changing other buckets, despite the all-suite geomean gain being
  below the general 3% threshold.
- Trajectory: `trajectory/iter-30-20260831-151531`.

### Final multi-stream verification

- Selected source: iteration 30, commit `35cebab`; no code changes were made
  before the independent final run.
- Measurement boundary: `BagWorkspace` and its four CUDA streams were created
  before warmup and before `triton.testing.do_bench`; creation is not included.
- Correctness: `CORRECT=True` across all 20 layout/precision/size cases.
- BAG geometric-mean latency: **1.599316 ms**; speedup geomean vs cuBLAS was
  **0.465460x**. The difference from iter-30's 1.538423 ms is concentrated in
  the first column-major IEEE N=512/1024 cases; N=2048 multi-stream latency
  remained stable at `1.0476/1.0532 ms` IEEE and `0.7178/0.7009 ms` TF32.
- Large serial buckets also remained stable: N=8192 was about 49.0-49.5 ms
  IEEE and 30.31-30.34 ms TF32.
- Decision: retain iteration 30. It provides a repeatable targeted N=2048
  improvement while auto-gating event overhead out of all other buckets.
- Trajectory: `trajectory/final-20260831-151619`.

### Published end-to-end BAG validation

- Published iteration 30 byte-for-byte from `solution/bag_triton` to
  `src/bag_triton`, including the depth-1 BAG API, internal same-stream
  non-atomic specialization, and dependency-batched auto scheduler.
- Published correctness: all **28** BAG cases passed across IEEE/TF32 and
  row-major/column-major/sliced layouts, including non-default caller-stream
  replay. All **26** primitive cases also passed, including concurrent atomic
  accumulation from two CUDA streams.
- Result artifacts: `benchmark_results/bag_vs_gemm_rtx3090.csv` and
  `benchmark_results/bag_vs_gemm_rtx3090.md`.
- The user requirement that CUDA stream creation remain outside measurement is
  enforced by constructing all four streams in the preallocated workspace;
  the benchmark callable receives that workspace and never creates streams.

### Iteration 31 — fully concurrent seven-stream atomic schedule

- Change: added two independent quarter-matrix buffers for
  `UA=SA-A11`/`UB=SB-B11`, generated `SA/SB/UA/UB` and zeroed all four C-psi
  blocks in one Triton preparation kernel, then launched M1--M7 on seven
  pre-created CUDA streams with atomic signed accumulation.
- Hypothesis: removing the M3-before-M6/M7 dependency and making every leaf
  product eligible for simultaneous execution could expose unused RTX 3090 SM
  capacity and move end-to-end BAG toward the arithmetic-only 8/7 bound.
- Measurement boundary: all stream and workspace allocation was outside the
  timed callable; preparation, output zeroing, seven primitive launches,
  atomics, stream fork/join, inverse transform, and required dependencies were
  included.
- Correctness: `CORRECT=True` across all 20 formal cases. A separate quick
  suite passed 18 numerical cases plus repeated execution on a non-default
  caller stream.
- BAG geometric-mean latency: **2.284818 ms**.
- Geometric-mean speedup vs cuBLAS GEMM: **0.325027x**.
- Change vs iteration-30 best (1.538423 ms): **48.51% slower**.
- Representative N=8192 latency: IEEE column/row
  `61.0601/61.4656 ms`; TF32 column/row `47.4532/40.3139 ms`.
- Analysis: seven half-size GEMMs contend for the same already-saturated SM,
  tensor-core, cache, and memory resources. Full atomic output adds ten signed
  atomic tile updates, including four contenders on C-psi12; the extra UA/UB
  materialization and stream fork/join overhead compound the loss. Atomic
  correctness removes output races but does not create execution capacity.
- Outcome: **reject as the auto schedule**, while retaining the explicit
  schedule temporarily for diagnosability. The next direction is to improve
  leaf-GEMM efficiency through a hybrid or grouped/batched implementation.
- Trajectory: `trajectory/iter-31-20260831-153322`.

### Iteration 32 — hybrid cuBLAS leaves with direct Triton decode

- Change: added an explicitly labelled hybrid schedule. Triton materializes
  independent M3/M6/M7 inputs and both M1/M2 input pairs; seven serial cuBLAS
  GEMMs store their products in four output quadrants plus three dead
  workspace buffers; one Triton kernel decodes the seven products directly
  into the final C basis.
- Algebraic decode: `C11=M4+M5`, `C12=M3+M5-M6-M2`,
  `C21=M1-M3-M5-M7`, and `C22=M1+M2-M3-M5`.
- Hypothesis: matching cuBLAS leaf throughput and removing intermediate C-psi
  merge passes would expose the practical upper bound after mandatory O(N^2)
  BAG transformations.
- Measurement boundary: all buffers were preallocated; input transforms,
  seven leaf GEMMs, the direct decode, and all required execution were timed.
  This result is explicitly hybrid and is not reported as a pure Triton
  primitive result.
- Correctness: `CORRECT=True` across all 20 formal cases; the quick interface
  suite also passed the hybrid schedule.
- BAG geometric-mean latency: **1.153518 ms**.
- Geometric-mean speedup vs cuBLAS GEMM: **0.648018x**.
- Change vs iteration-30 pure-Triton best (1.538423 ms): **25.02% faster**.
- N=8192 speedup: IEEE column/row `1.020x/0.985x`; TF32 column/row
  `0.953x/0.959x`.
- Component result at N=8192 IEEE row-major: seven cuBLAS leaves totalled
  `40.444928 ms`; transformations and host gaps totalled `2.772990 ms`, down
  from `3.560448 ms` before direct decode.
- Analysis: this measured hybrid point does not reach 8/7. Seven half-size
  cuBLAS calls total about 40.5 ms, and input materialization/direct decoding
  adds about 2.8 ms. The approximately 1.02x result is an end-to-end
  observation for this run, not a global or arithmetic ceiling.
- Outcome: **keep as an explicit diagnostic/performance backend**, not as the
  claimed pure-Triton primitive implementation. Continue with a grouped Triton
  leaf kernel to remove launch overhead in the small buckets.
- Trajectory: `trajectory/iter-32-20260831-154247`.

### Iteration 33 — one-launch grouped Triton BAG products

- Change: added three preallocated `(7,N/2,N/2)` buffers. One Triton kernel
  materializes all seven left/right input pairs, a two-dimensional Triton grid
  computes all seven independent GEMMs in one launch, and the direct-decode
  kernel writes the final C basis. No CUDA streams or atomic output updates are
  needed on this path.
- Hypothesis: batching the seven leaf products into one GPU launch would retain
  BAG's 7/8 arithmetic while eliminating the seven-launch and event overhead
  that dominated the N=512/1024 buckets.
- Auto policy: select grouped execution for N<=4096 when grouped workspace is
  available; retain the serial large-matrix path at N=8192. All grouped buffers
  are allocated before timing.
- Correctness: `CORRECT=True` across all 20 formal cases. The separate quick
  suite also passed the grouped schedule in column-major storage.
- BAG geometric-mean latency: **0.883871 ms**.
- Geometric-mean speedup vs cuBLAS GEMM: **0.843987x**.
- Change vs iteration-30 pure-Triton best (1.538423 ms): **42.55% lower
  latency / 1.740x faster**.
- Change vs iteration-32 hybrid geomean (1.153518 ms): **23.38% lower**, driven
  by the launch-bound small and medium buckets while remaining pure Triton.
- Representative grouped results: N=512 IEEE/TF32 row-major
  `0.0348/0.0266 ms`; N=2048 `0.8970/0.5990 ms`; N=4096
  `6.2510/4.0458 ms`.
- N=4096 reaches `0.930--0.941x` in IEEE and `0.974--0.979x` in TF32.
  N=8192 remains on the prior serial path at about `48.6--49.3 ms` IEEE and
  `30.22--30.28 ms` TF32.
- Analysis: launch batching fixes the small-bucket anomaly and nearly closes
  the medium TF32 gap. At N=8192, GEMM throughput and O(N^2) transforms still
  dominate, so batching cannot produce the arithmetic-only 8/7 ratio.
- Outcome: **new pure-Triton winner**. Continue with full correctness and a
  final independent benchmark before publishing.
- Trajectory: `trajectory/iter-33-20260831-154625`.

### Final grouped-Triton verification

- Selected source: iteration 33, commit `0a9d559`.
- Correctness before final timing: all **28** BAG cases and all **26** primitive
  cases passed, including sliced positive-stride layouts, non-default caller
  streams, repeated seven-stream atomic calls, grouped column-major execution,
  and concurrent primitive atomic accumulation.
- Independent formal correctness: `CORRECT=True` across all 20 benchmark
  layout/precision/size cases.
- BAG geometric-mean latency: **0.885580 ms**.
- Geometric-mean speedup vs cuBLAS GEMM: **0.847458x**.
- Reproducibility vs iteration 33: latency differs by only **0.19%**
  (`0.883871 -> 0.885580 ms`).
- Best near-parity buckets: TF32 N=2048 column/row `0.984x/0.949x`; TF32
  N=4096 column/row `0.980x/0.981x`.
- Measurement boundary: grouped input/product buffers were allocated before
  warmup and timing. The timed path contains input materialization, one batched
  seven-product Triton launch, direct decode, and required execution; it creates
  no CUDA streams.
- Decision: publish iteration 33 as the new pure-Triton implementation. The
  measured gap to 8/7 is explained by finite-size GEMM scaling and mandatory
  O(N^2) transforms, not by omitted synchronization or allocation overhead.
- Trajectory: `trajectory/final-20260831-154749`.

### Published grouped-Triton validation

- Published the iteration-33 grouped kernel and four-stage large plain-product
  pipeline from `solution/bag_triton` to `src/bag_triton`.
- Default `allocate_bag_workspace` now provisions grouped buffers automatically
  through N=4096; callers may pass `grouped=False` to choose the lower-memory
  fallback explicitly. Allocation remains outside `bag_matmul` timing.
- Published result artifacts were refreshed from the independent final run:
  `benchmark_results/bag_vs_gemm_rtx3090.csv` and
  `benchmark_results/bag_vs_gemm_rtx3090.md`.
- The all-case pure-Triton result is **0.885580 ms** BAG latency geomean and
  **0.847458x** speedup geomean versus ordinary cuBLAS GEMM.

### Post-publication upper-bound clarification

- User correction: a previous approximately 1.14x upper bound must not be
  replaced by iteration 32's approximately 1.02x hybrid measurement.
- Correct distinction: `8/7=1.142857x` is the zero-overhead arithmetic upper
  bound; `T_gemm(N)/(7*T_gemm(N/2))` is the finite-shape GEMM-only bound; the
  reported hybrid ratio is a specific end-to-end observation after BAG
  overhead.
- From the same final CSV at N=8192, the GEMM-only bounds are IEEE
  column/row-major `1.1325x/1.0875x` and TF32 column/row-major
  `1.0288x/1.0373x`. Thus the IEEE column-major data does retain a measured
  leaf-GEMM upper bound close to 1.14x.
- Optimization implication: preserve 1.14x as the target ceiling and focus on
  reducing/overlapping input materialization and direct-decode traffic; do not
  treat 1.02x as evidence that further improvement is impossible.

### Iteration 34 — sparse grouped inputs and precision-specific GEMM buckets

- Change: retained the one-launch grouped Triton schedule, but materialized
  only the eight BAG operands that are genuine linear combinations. The
  grouped GEMM now reads unchanged A11/A12/A21 and B11/B12/B21 quadrants
  directly through uniform product-id branches, eliminating six quarter-matrix
  stores and their later reads. An offline, overwrite-safe Triton configuration
  search selected static size/precision buckets; no online autotuning remains.
- Auto policy: extended preallocated grouped execution from N<=4096 through
  N<=8192. Workspace allocation and JIT remain outside the timed callable;
  preparation, seven Triton products, and direct decoding remain inside it.
- Hypothesis: removing redundant O(N^2) operand copies and selecting different
  BK/pipeline depths for IEEE versus TF32 would reduce both transform traffic
  and leaf-GEMM latency without reintroducing streams or atomic accumulation.
- Correctness: the 18-case quick numerical/interface suite passed before the
  formal run, including row-major, column-major, sliced layouts, both
  precisions, non-default caller streams, and all retained explicit schedules.
  The formal benchmark reported `CORRECT=True` for all 20 signal cases.
- BAG geometric-mean latency: **0.831471 ms**, 5.93% below iteration 33's
  **0.883871 ms** and 6.11% below the published rerun's **0.885580 ms**.
- Geometric-mean speedup versus ordinary cuBLAS GEMM: **0.892095x**, up from
  iteration 33's **0.843987x** and the published **0.847458x**.
- Large buckets: at N=8192, IEEE column/row reached **0.995x/0.999x** and TF32
  reached **1.013x/1.011x**. At N=4096, TF32 reached **1.052x/1.046x**.
- Analysis: sparse input materialization and precision-specific tiles provide
  a broad, repeatable end-to-end win. The largest shapes are now approximately
  parity or modestly faster than ordinary GEMM, while N=512/1024 remains
  dominated by fixed preparation/decode and launch overhead rather than the
  seven-product arithmetic count.
- Outcome: **new best**. Retain this implementation and continue with memory
  footprint reduction and launch/epilogue work, prioritizing the small buckets
  without weakening the all-shape reporting contract.
- Trajectory: `trajectory/iter-34-20260831-162705`.

### Iteration 35 — wide pointwise preparation and decode programs

- Change: kept the iteration-34 grouped GEMM body unchanged, while changing
  the grouped input-preparation and direct-decode kernels from `BLOCK=256`,
  four warps to `BLOCK=2048`, eight warps for N>=2048. N=512/1024 retain the
  smaller pointwise configuration. Allocation, JIT, and all rejected CUDA
  Graph experiments remain outside the selected implementation.
- Hypothesis: the transform and decode passes traverse physically contiguous
  storage and touch many arrays per element; wider programs should amortize
  CTA scheduling and address work on large matrices without affecting product
  reuse or numerical behavior.
- Correctness: all 20 formal layout/precision/size cases reported
  `CORRECT=True` with unchanged IEEE and TF32 error levels.
- BAG geometric-mean latency: **0.828137 ms**, 0.40% below iteration 34's
  **0.831471 ms** and 6.31% below iteration 33's **0.883871 ms**.
- Geometric-mean speedup versus ordinary cuBLAS GEMM: **0.899713x**. The
  reference GEMM run varied materially at N=8192 IEEE, so the selection is
  based on BAG latency rather than treating the ratio change as kernel work.
- Per-bucket effect: N=2048 BAG latency was `0.8356/0.8612 ms` IEEE and
  `0.5724/0.5765 ms` TF32 for column/row-major; N=8192 was
  `47.1081/46.6033 ms` IEEE and `28.1441/28.1436 ms` TF32.
- Analysis: the improvement is small in the all-suite geomean because only
  three size buckets change, but pre-screens across 512/1024/2048/4096-wide
  blocks consistently favored 2048 elements with eight warps for the large
  pointwise passes. The GEMM body remains the dominant large-N cost.
- Outcome: **keep as a targeted new best**; do not extend the wide pointwise
  program to N<=1024, where it did not provide a stable benefit.
- Trajectory: `trajectory/iter-35-20260831-163827`.

### Baseline 512

- Scope: square-512 across column-major/row-major and IEEE/TF32, always with
  both input combinations and two atomic output accumulations.
- Existing configuration: `128x64x32`, 8 warps, 2 stages, 32 CTAs.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.077348 ms**.
- Geometric-mean speedup vs unfused: **0.534826x**.
- Per-mode fused latency (ms): IEEE column-major `0.0901`, IEEE row-major
  `0.0819`, TF32 column-major `0.0696`, TF32 row-major `0.0696`.
- Analysis: 32 CTAs underfill the RTX 3090's 82 SMs, while the four-input,
  two-stage tile carries an estimated 98,304-byte pipeline payload and thus
  cannot recover latency hiding through multi-CTA residency.
- Trajectory: `trajectory/baseline-20260831-143739`.

### Iteration 16 — compact high-grid tile

- Change: added a fused near-square bucket up to 512 using `32x32x16`, 4
  warps, and 3 stages.
- Hypothesis: increasing the 512 output grid from 32 to 256 CTAs and reducing
  the estimated four-input pipeline footprint from 98,304 to 24,576 bytes
  would outweigh the loss of per-CTA input reuse.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.046449 ms**.
- Geometric-mean speedup vs unfused: **0.871591x**.
- Change vs 512 baseline (0.077348 ms): **1.665x faster / 39.95% lower
  latency**.
- Per-mode fused latency (ms): IEEE column-major `0.0553`, IEEE row-major
  `0.0502`, TF32 column-major `0.0410`, TF32 row-major `0.0410`.
- Outcome: **new 512 best**. The broad improvement confirms that whole-GPU
  underfill and shared-memory-constrained residency dominated the old bucket.
- Trajectory: `trajectory/iter-16-20260831-143818`.

### Iteration 17 — wider K tile with shallower pipeline

- Change: retained the `32x32`, 4-warp output tile but changed
  `BK=16/stages=3` to `BK=32/stages=2`.
- Hypothesis: halving K-loop iterations while keeping the estimated pipeline
  footprint near 32 KiB could improve instruction efficiency.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.047569 ms**.
- Geometric-mean speedup vs unfused: **0.865065x**.
- Change vs iteration-16 best (0.046449 ms): **2.41% slower**.
- Per-mode fused latency (ms): IEEE column-major `0.0563`, IEEE row-major
  `0.0492`, TF32 column-major `0.0430`, TF32 row-major `0.0430`.
- Outcome: **reject**. Fewer K-loop iterations do not compensate for the
  shallower pipeline and larger per-stage loads, especially in TF32.
- Trajectory: `trajectory/iter-17-20260831-143847`.

### Iteration 18 — more warps per compact CTA

- Change: restored `BK=16/stages=3` and increased the `32x32` tile from 4
  to 8 warps.
- Hypothesis: more intra-CTA parallelism and fewer accumulators per thread
  might improve latency hiding.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.060066 ms**.
- Geometric-mean speedup vs unfused: **0.685031x**.
- Change vs iteration-16 best (0.046449 ms): **29.32% slower**.
- Per-mode fused latency (ms): IEEE column-major `0.0758`, IEEE row-major
  `0.0758`, TF32 column-major `0.0471`, TF32 row-major `0.0481`.
- Outcome: **reject**. The small output tile does not contain enough work to
  amortize eight-warp cooperation; four warps is decisively better.
- Trajectory: `trajectory/iter-18-20260831-143913`.

### Iteration 19 — deeper compact-tile pipeline

- Change: restored four warps and increased `32x32x16` from 3 to 4 stages.
- Hypothesis: the approximately 32-KiB four-stage footprint could preserve
  multi-CTA residency while prefetching farther ahead.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.053670 ms**.
- Geometric-mean speedup vs unfused: **0.782033x**.
- Change vs iteration-16 best (0.046449 ms): **15.55% slower**.
- Per-mode fused latency (ms): IEEE column-major `0.0686`, IEEE row-major
  `0.0686`, TF32 column-major `0.0420`, TF32 row-major `0.0420`.
- Outcome: **reject**. Four stages add pipeline/synchronization overhead beyond
  what this short GEMM can amortize; three stages remains the optimum tested.
- Trajectory: `trajectory/iter-19-20260831-143941`.

### Stall reassessment after iteration 19

- NCU remains unavailable due the previously observed `ERR_NVGPUCTRPERM`;
  per the one-probe policy it was not retried.
- Iterations 17–19 show that iteration 16's `BK=16`, four warps, and three
  stages are each locally important: wider K, more warps, and deeper
  pipelining all regress.
- Triton's official I/O-bound matmul configuration space uses 16/32-sized M
  tiles, 32/64-sized N tiles, and two warps when N is at most 64. This motivated
  testing the untried two-warp and compact rectangular directions.

### Iteration 20 — two-warp compact tile

- Change: restored three stages and reduced `32x32x16` from four to two
  warps.
- Hypothesis: fewer cooperative threads and greater per-thread ILP could reduce
  overhead for this short, I/O/atomic-heavy GEMM.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.052497 ms**.
- Geometric-mean speedup vs unfused: **0.793737x**.
- Change vs iteration-16 best (0.046449 ms): **13.02% slower**.
- Per-mode fused latency (ms): IEEE column-major `0.0604`, IEEE row-major
  `0.0543`, TF32 column-major `0.0481`, TF32 row-major `0.0481`.
- Outcome: **reject**. Two warps leave insufficient latency-hiding capacity;
  four warps is the optimum tested warp count.
- Trajectory: `trajectory/iter-20-20260831-144027`.

### Iteration 21 — compact rectangular tile

- Change: tested `32x64x16`, four warps, and three stages, producing 128
  output CTAs with an estimated 36-KiB pipeline payload.
- Hypothesis: doubling N-side output reuse might outweigh halving the launch
  grid while retaining enough programs to cover the 82-SM GPU.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.047456 ms**.
- Geometric-mean speedup vs unfused: **0.870281x**.
- Change vs iteration-16 best (0.046449 ms): **2.17% slower**.
- Per-mode fused latency (ms): IEEE column-major `0.0573`, IEEE row-major
  `0.0502`, TF32 column-major `0.0420`, TF32 row-major `0.0420`.
- Outcome: **reject**. The additional reuse nearly ties but does not beat the
  256-CTA square tile; whole-GPU work supply remains more valuable.
- Trajectory: `trajectory/iter-21-20260831-144100`.

### Iteration 22 — smaller program super-group

- Change: restored the iteration-16 tile and reduced `GROUP_M` from 8 to 4.
- Hypothesis: a smaller grouped-ordering working set might improve L2 locality
  or tail scheduling on the 16x16 output grid.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.046449 ms**, exactly tied with iteration
  16 at benchmark resolution.
- Geometric-mean speedup vs unfused: **0.885925x**; the change from iteration
  16 is caused by reference timing variation, not lower fused latency.
- Per-mode fused latency (ms): IEEE column-major `0.0553`, IEEE row-major
  `0.0502`, TF32 column-major `0.0410`, TF32 row-major `0.0410`.
- Outcome: **tie/reject for simplicity**. Retain `GROUP_M=8`, which matches
  the established launch policy and is not slower.
- Trajectory: `trajectory/iter-22-20260831-144131`.

### Second stall reassessment after iteration 22

- Iterations 20–22 ruled out fewer warps, compact rectangular reuse, and
  smaller program grouping.
- The remaining official I/O-bound configuration dimension not yet covered was
  a 16-row output tile; it preserves a 256-CTA grid while changing dot operand
  and atomic-output thread mapping.

### Iteration 23 — 16-row I/O-oriented tile

- Change: tested `16x64x16`, four warps, and three stages.
- Hypothesis: retaining 256 CTAs while assigning a wider contiguous N segment
  to each program might improve output atomics and right-operand loads.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.058420 ms**.
- Geometric-mean speedup vs unfused: **0.718442x**.
- Change vs iteration-16 best (0.046449 ms): **25.77% slower**.
- Per-mode fused latency (ms): IEEE column-major `0.0717`, IEEE row-major
  `0.0717`, TF32 column-major `0.0471`, TF32 row-major `0.0481`.
- Outcome: **reject**. The asymmetric tile loses too much operand reuse and/or
  produces a worse dot layout; the square compact tile remains superior.
- Trajectory: `trajectory/iter-23-20260831-144210`.

### Iteration 24 — shallower best-tile pipeline

- Change: restored the best `32x32x16`, four-warp, `GROUP_M=8` tile and
  reduced the pipeline from three to two stages.
- Hypothesis: a 16-KiB estimated pipeline footprint could improve CTA
  residency and reduce synchronization overhead.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.049067 ms**.
- Geometric-mean speedup vs unfused: **0.855399x**.
- Change vs iteration-16 best (0.046449 ms): **5.64% slower**.
- Per-mode fused latency (ms): IEEE column-major `0.0573`, IEEE row-major
  `0.0573`, TF32 column-major `0.0420`, TF32 row-major `0.0420`.
- Outcome: **reject**. Three stages are required to hide four-input load
  latency; with iterations 19 and 24, stages 2/3/4 are now fully compared.
- Trajectory: `trajectory/iter-24-20260831-144240`.

### Final square-512 verification

- Restored source: iteration 16, commit `688bf32`.
- Selected configuration: `32x32x16`, four warps, three stages,
  `GROUP_M=8`.
- Correctness: `CORRECT=True`.
- Fused geometric-mean latency: **0.047132 ms**.
- Geometric-mean speedup vs unfused: **0.890515x**.
- Change vs 512 baseline (0.077348 ms): **1.641x faster / 39.07% lower
  latency**.
- Per-mode fused latency (ms): IEEE column-major `0.0563`, IEEE row-major
  `0.0522`, TF32 column-major `0.0410`, TF32 row-major `0.0410`.
- The final fused geomean is within 1.47% of iteration 16 and the implementation
  remains correct, confirming the selected winner.
- Stop rationale: all principal configuration directions were tested with at
  least three distinct axes—tile/grid shape (16, 21, 23), reduction/pipeline
  shape (17, 19, 24), warp count (18, 20), and launch grouping (22). None
  exceeded iteration 16, so the viable configuration search is exhausted.
- Trajectory: `trajectory/final-20260831-144315`.

### Published 512 validation

- Published the iteration-16 winner from `solution/` to `src/`.
- Correctness: all **26** cases passed, including the new non-aligned
  `(511,509,513)` boundary case in row-major, column-major, and sliced
  positive-stride layouts, plus concurrent two-stream atomic accumulation.
- Full verdict: all 32 benchmark cases passed correctness with fused latency
  geomean **0.354190 ms** and fused-vs-unfused geomean **0.899699x**.
- Relative to the previously published optimized kernel, the full-suite fused
  latency improves by **1.077x** even though only the 512 bucket changed.
- Formal square-512 improvement: IEEE `0.079872 -> 0.051200 ms` (**1.560x**)
  and TF32 `0.070656 -> 0.040960 ms` (**1.725x**) for both layouts.

### Final sparse-grouped BAG verification and publication

- Selected implementation: iteration 34's sparse operand materialization and
  precision/size-specific grouped GEMM buckets plus iteration 35's wide
  pointwise preparation/decode programs for N>=2048.
- Rejected follow-ups: tile-resident seven-product decoding, direct K-loop
  operand generation, CUDA Graph replay, unit-inner-stride specialization,
  `maxnreg=64`, and semi-sparse materialization. They either regressed the
  measured suite or produced unstable/no material gains; `maxnreg=64` caused
  catastrophic register spilling and was removed immediately.
- Correctness before publication: all **28** full BAG cases and all **26** full
  primitive cases passed. Coverage includes IEEE/TF32, row-major,
  column-major, sliced positive strides, non-aligned shapes, non-default caller
  streams, repeated multistream/seven-stream schedules, and concurrent atomic
  primitive updates.
- Independent final result: **0.832332 ms** BAG latency geometric mean and
  **0.901037x** speedup geometric mean versus ordinary cuBLAS GEMM;
  `CORRECT=True` on all 20 benchmark cases. The final latency is within 0.51%
  of iteration 35's `0.828137 ms`, confirming the selected implementation.
- Aggregate by N (`GEMM/BAG`): N=512 `0.757575x`, N=1024 `0.833819x`,
  N=2048 `0.961977x`, N=4096 `1.003673x`, and N=8192 `0.973773x`.
- Large TF32 results: N=4096 is `1.053x/1.053x` and N=8192 is
  `1.015x/1.012x` for column/row-major. IEEE remains more sensitive to
  run-to-run cuBLAS timing at N=8192, so BAG latency and the complete CSV are
  reported alongside every ratio.
- Published `solution/bag_triton/bag.py` byte-for-byte to
  `src/bag_triton/bag.py`; a source-root quick replay passed all 18 numerical
  cases and retained explicit schedule/interface checks.
- Refreshed `benchmark_results/bag_vs_gemm_rtx3090.csv` and the accompanying
  Markdown report from `trajectory/final-20260831-164812`.
- The arithmetic-only upper bound remains `8/7=1.142857x`; the prior measured
  IEEE column-major finite-shape bound near `1.1325x` is retained separately
  from this final run's end-to-end result.

### Iteration 36 — increase N=512 grouped-GEMM work supply

- Change: specialize the `half<=256` grouped leaf GEMM from `64x64x32`, four
  warps, four stages to `32x64x16`, four warps, three stages. The N=512 launch
  grows from 112 to 224 CTAs across the seven products while preserving the
  sparse prepare and direct decode kernels.
- Hypothesis: the former 1.37-CTA-per-SM grid under-supplies the RTX 3090's 82
  SMs; doubling spatial work supply and shortening the K pipeline should hide
  latency better than the larger tile's extra reuse.
- Correctness: `CORRECT=True` on all 20 formal benchmark cases covering five
  sizes, both layouts, and IEEE/TF32.
- Full-suite BAG geometric-mean latency: **0.818945 ms**; geometric-mean
  `GEMM/BAG` speedup: **0.915502x**.
- Change versus iteration 35 (0.828137 ms): **1.11% lower latency**. Change
  versus the previously published independent final (0.832332 ms): **1.61%
  lower latency**.
- N=512 BAG latencies: IEEE column/row **0.0317/0.0287 ms**, TF32 column/row
  **0.0236/0.0225 ms**; their latency geomean is **0.02636 ms**, about **8.0%
  below** the prior published N=512 aggregate latency of 0.028662 ms.
- N=512 `GEMM/BAG` ratios: IEEE column/row **0.871x/0.929x**, TF32 column/row
  **0.783x/0.727x**. Ordinary-GEMM timing varied across runs, so selection is
  based primarily on BAG latency while retaining every ratio.
- N>=1024 uses the prior implementation unchanged. Large TF32 remains above
  ordinary GEMM at N=2048 column-major (1.020x), N=4096 (1.054x/1.056x), and
  N=8192 (1.020x/1.023x).
- Outcome: **accept as the new provisional best** and continue with the
  independent N=1024 pointwise/IEEE-GEMM candidates before final publication.
- Trajectory: `trajectory/iter-36-20260831-170515`.

### Iteration 37 — widen N=1024 grouped pointwise passes

- Change: retain BLOCK=256/four warps through `half<=256`, specialize
  `half<=512` (full N=1024) prepare and decode to BLOCK=1024/four warps, and
  retain BLOCK=2048/eight warps above that boundary.
- Hypothesis: N=1024 has enough elements to amortize wider vector programs;
  reducing both pointwise grids from 1024 to 256 programs should lower launch
  scheduling overhead without changing the seven-product GEMM.
- Correctness: `CORRECT=True` on all 20 formal benchmark cases.
- N=1024 BAG latencies: IEEE column/row **0.1280/0.1270 ms**, TF32 column/row
  **0.0973/0.0973 ms**. Their geomean is **0.11138 ms**, **1.70% lower** than
  iteration 36's 0.11331 ms N=1024 geomean.
- Focused pre-screen independently showed the same direction: IEEE column
  `0.1321 -> 0.1309 ms` and TF32 column/row `0.0983/0.0984 ->
  0.0973/0.0975 ms`, with N=512 unchanged at benchmark resolution.
- Full-suite BAG geomean is **0.820715 ms** and full-suite `GEMM/BAG` geomean
  is **0.913709x**. The 0.22% full-geomean regression versus iteration 36 is
  smaller than the independent large-shape run variation; N>=2048 executes
  byte-identical kernels, so it is not attributed to this specialization.
- Outcome: **accept for the small-matrix objective**. The targeted N=1024
  signal improves consistently in focused and formal measurements, while all
  other buckets retain their prior implementation.
- Trajectory: `trajectory/iter-37-20260831-170655`.

### Iteration 38 — precision-specific N=512 K tile

- Change tested: retain the iteration-36 `32x64` spatial tile and use BK=32
  for IEEE while retaining BK=16 for TF32; four warps, three stages, and
  `GROUP_M=8` remain unchanged.
- Hypothesis: IEEE tensor-core lowering may amortize a wider K tile better,
  reducing loop/pipeline overhead without reducing the 224-CTA work supply.
- Correctness: `CORRECT=True` on all 20 formal benchmark cases.
- Focused pre-screen suggested a small IEEE column improvement (`0.0317 ->
  0.0307 ms`) with IEEE row unchanged, but the formal replay measured exactly
  **0.0317/0.0287 ms** for IEEE column/row, identical to iteration 37 at the
  benchmark's resolution. TF32 was intentionally unchanged.
- Full-suite BAG geomean is **0.823547 ms**, **0.35% slower** than iteration
  37's 0.820715 ms; full-suite `GEMM/BAG` geomean is **0.912918x**.
- Outcome: **reject**. The single-kernel signal did not translate into a
  repeatable end-to-end improvement. Restore BK=16 for both precision modes
  in the N=512 bucket and retain iteration 37 as the selected implementation.
- Trajectory: `trajectory/iter-38-20260831-170816`.

### Iteration 39 — last-completer decode for small TF32 buckets

- Change: specialize full N=512 and N=1024 TF32 grouped BAG so the existing
  prepare launch clears 32/64 per-spatial-tile counters in the otherwise
  unused `grouped_left[4]` plane.  After each product CTA stores its tile, it
  publishes one GPU-scope acquire/release atomic arrival; the seventh CTA
  decodes all seven products directly into the four output quadrants in
  1024-element chunks.  This keeps the product-major seven-GEMM launch while
  removing the independent decode launch.  IEEE and N>=2048 retain the
  iteration-37 path byte-for-byte at the specialization level.
- Hypothesis: at half sizes 256 and 512, the 32/64 last-completer tails can be
  overlapped with the remaining grouped GEMM work, while only 224/448 scalar
  counter atomics are required.  A 1024-element decode chunk should amortize
  address/loop overhead without spilling on sm86.
- Focused same-process A/B before the formal run measured N=512 TF32
  row/column `23.304/23.191 -> 22.763/22.667 us` (**1.0238x/1.0231x**) and
  N=1024 TF32 `96.944/97.081 -> 92.436/92.623 us`
  (**1.0488x/1.0481x**).  Three-seed row/column replay and thousands of timed
  calls found no stale-counter or visibility failure.  The fused grouped
  specialization uses 80 registers/thread, 12/16 KiB shared memory for
  N=512/N=1024, and zero spills.
- Correctness: `CORRECT=True` on all 20 formal benchmark cases.
- Formal N=512 TF32 BAG latency: column/row **0.0225/0.0225 ms**, versus
  iteration 37's **0.0236/0.0225 ms**.  The two-layout geomean falls from
  about **0.02304 ms to 0.02250 ms** (**2.36% lower**); the row result is on
  the same CUDA-event quantization step while the focused continuous A/B
  still measured a repeatable row improvement.
- Formal N=1024 TF32 BAG latency: column/row **0.0922/0.0923 ms**, versus
  iteration 37's **0.0973/0.0973 ms**.  The two-layout geomean is about
  **0.09225 ms**, **5.19% lower**.
- Full-suite BAG geometric-mean latency: **0.812858 ms**, **0.96% lower**
  than iteration 37's 0.820715 ms and the best formal result so far.
  Full-suite geometric-mean `GEMM/BAG` speedup: **0.925748x**, up from
  0.913709x in iteration 37.
- Large TF32 remains faster than ordinary GEMM at N=4096
  (**1.056x/1.057x**) and N=8192 (**1.013x/1.029x**) for column/row-major.
- Rejected companion experiments: tile-major last-completer ordering;
  workspace-free 3+4 and 5+2 two-wave BAG; N=512 two-wave fusion; and 29
  additional N=512 grouped tile/warp/stage configurations.  They either lost
  to the product-major last-completer or regressed at least one layout.
- Outcome: **accept as the new winner**.  It improves both targeted TF32
  small-matrix buckets, preserves seven genuine half-size GEMMs, creates no
  streams or events in the measured call, and improves the complete suite.
- Trajectory: `trajectory/iter-39-20260831-175410`.

### Final small-matrix last-completer verification and publication

- Published the complete iteration-36/37/39 winner from `solution/` to
  `src/`; the two `bag.py` files are byte-for-byte identical.
- Independent final benchmark: **0.812665 ms** BAG latency geometric mean and
  **0.921138x** geometric-mean `GEMM/BAG` speedup, with `CORRECT=True` on all
  20 cases.  The latency is 0.024% below iteration 39's 0.812858 ms, confirming
  the winner at effectively identical full-suite performance.
- Final N=512 BAG latency is IEEE column/row **0.0317/0.0287 ms** and TF32
  column/row **0.0225/0.0225 ms**.  Final N=1024 is IEEE
  **0.1280/0.1280 ms** and TF32 **0.0922/0.0922 ms** for column/row-major.
- Full regression before publication passed all **28** BAG numerical/interface
  cases and all **26** primitive cases.  A source-root replay then passed all
  **18** quick numerical cases plus grouped, hybrid, multistream, and
  seven-stream interface checks.
- Refreshed `benchmark_results/bag_vs_gemm_rtx3090.csv` and the Markdown report
  from the independent final trajectory.  The report retains the explicit
  ratio definition `ordinary GEMM median / BAG median`; values above one mean
  BAG is faster.
- The timed grouped path creates no streams or events.  The winning small-TF32
  specialization uses only the existing product grid, 32/64 workspace-backed
  scalar counters, and one acquire/release arrival per product CTA.
- Trajectory: `trajectory/final-20260831-175851`.

### Iteration 40 — four-tile product microchunks

- Change: retained the existing two-dimensional product-major launch for every
  non-fused bucket, but flattened the N=512/1024 TF32 last-completer grid into
  four-spatial-tile microchunks.  Within each chunk the linear CTA ids assign
  M1--M7 before advancing to the next four tiles; algebra, tile sizes, workspace,
  counter protocol, launch count, and numerical order are unchanged.
- Hypothesis: bounding the launch distance between the seven products for one
  spatial region would keep the product workspace hotter for the final CTA's
  fused decode while retaining some same-product input reuse across four
  neighboring tiles.
- Correctness: all 20 formal layout/precision/size cases reported
  `CORRECT=True` with unchanged IEEE and TF32 error levels.
- BAG geometric-mean latency: **0.823267 ms**, 1.26% above this session's
  independent **0.813021 ms** baseline.
- Targeted regression: N=512 TF32 rose from **0.0225 ms** to **0.0246 ms** in
  both layouts (about 9.3% slower); N=1024 TF32 rose from **0.0922 ms** to
  **0.0983 ms** in both layouts (about 6.6% slower).
- Analysis: at G=4 the loss of long same-product runs and their operand-cache
  reuse outweighs any product-workspace locality gained by bringing the seven
  arrivals closer together.  This agrees with the previously rejected fully
  tile-major ordering and makes smaller chunks unattractive.
- Outcome: **reject**.  Retain the implementation mechanism long enough to
  test a larger G=16 endpoint; do not publish G=4.
- Trajectory: `trajectory/iter-40-20260831-220905`.

### Iteration 41 — sixteen-tile product microchunks

- Change: enlarged only the N=512/1024 TF32 last-completer microchunk from
  four to sixteen spatial tiles.  The current product-major implementation
  remains unchanged for IEEE and every larger size.
- Hypothesis: longer same-product runs might recover most operand-cache reuse
  while still bounding the distance between a spatial tile's first and last
  product more tightly than the full product-major grid.
- Correctness: all 20 formal cases reported `CORRECT=True` with unchanged
  numerical error.
- BAG geometric-mean latency: **0.820826 ms**, 0.96% above the independent
  **0.813021 ms** product-major baseline.
- Targeted result: N=512 TF32 was **0.0236 ms** in both layouts versus the
  **0.0225 ms** baseline (about 4.9% slower).  N=1024 TF32 was **0.0932 ms**
  versus **0.0922 ms** (about 1.1% slower).
- Analysis: increasing G moves performance monotonically back toward the
  product-major endpoint but does not surpass it.  Together with iteration 40
  and the prior tile-major rejection, this shows product-input locality is more
  valuable than shortening product-workspace reuse distance for these grids.
- Outcome: **reject and close this scheduling axis**.  Restore the exact
  product-major launch before implementing interleaved preparation.
- Trajectory: `trajectory/iter-41-20260831-221028`.

### Iteration 42 — GROUP_M-aligned eight-tile microchunks

- Change: set the small-TF32 microchunk to eight spatial tiles, exactly one
  `GROUP_M=8` swizzle stripe.  This was the most structurally favorable middle
  point because it does not split the existing eight-CTA right-operand reuse
  group.  CUDA is not assumed to execute or complete CTAs in numeric order;
  correctness remains entirely protected by the existing GPU-scope acq_rel
  arrival counter.
- Hypothesis: an intact swizzle stripe could preserve the product-input reuse
  lost by G=4 while retaining more product-workspace locality than G=16 or the
  full product-major launch.
- Correctness: all 20 formal cases reported `CORRECT=True` with unchanged
  numerical error.
- BAG geometric-mean latency: **0.822996 ms**, 1.23% above the independent
  **0.813021 ms** product-major baseline.
- Targeted result: N=512 TF32 was **0.0246 ms** in both layouts (about 9.3%
  slower than **0.0225 ms**); N=1024 TF32 was **0.0973 ms** (about 5.5%
  slower than **0.0922 ms**).
- Analysis: even when the microchunk matches the GEMM swizzle, reducing the
  full same-product run loses more operand/cache efficiency than the fused
  decoder gains.  G=4, G=8, G=16, and the previously measured tile-major
  endpoint all agree.
- Outcome: **reject**.  Three consecutive scheduling candidates have not
  improved the best result; perform the required stall reassessment, restore
  the exact product-major kernel, and change direction to interleaved prepare.
- Trajectory: `trajectory/iter-42-20260831-221133`.

### Stall reassessment after iterations 40--42

- Profiling status: Nsight Compute hardware counters remain unavailable under
  the previously observed `ERR_NVGPUCTRPERM`, so the reassessment uses CUDA
  event distributions, the recorded zero-spill resource data for iteration 39,
  and the complete iteration log rather than retrying the blocked profiler.
- Runtime evidence: tile-major and microchunks G=4/8/16 all regress the target
  TF32 buckets; the larger chunks approach but never beat product-major.
- Literature check: SubCuber and FalconGEMM both make dataflow placement,
  selective pre-sum fusion, and resource-aware fallback first-class choices;
  neither implies that disturbing a cache-efficient product run should help.
- Decision: close the CTA-order axis and move to a distinct dataflow direction:
  fuse the independent prepare work into the two raw-quadrant products while
  preserving product-major ordering inside each phase and one GEMM accumulator
  per CTA.

### Iteration 43 — edge-owner raw2 interleaved preparation

- Change: restored product-major ordering.  For N=512/1024 TF32, split the
  grouped launch into M4/M5 followed by M1/M2/M3/M6/M7.  In the first launch,
  M4 CTAs with `program_n==0` materialize the four canonical-left transformed
  planes inside their K loop, while M5 CTAs with `program_m==0` materialize the
  four canonical-right planes.  M4 writes each spatial counter to two; the
  second launch performs five acq_rel arrivals and the existing last-completer
  decode.  IEEE and larger buckets retain the original path.
- Hypothesis: the two-launch schedule has the same 3/6 ideal wave counts as the
  original small kernels, removes the standalone prepare grid, reuses one raw
  fragment on each side, and never holds more than one GEMM accumulator.
- Result: the formal benchmark **failed correctness** at its first enabled
  case, N=512 column-major TF32.  Relative L2 error was **1.907166e+00** and
  maximum absolute error was **2.306713e+01**.  All ten preceding IEEE cases,
  which retain the original path, passed with their normal errors and latency.
- Analysis: the order-one error is a deterministic column-major transformed
  operand/store mapping defect, not a tolerance issue or marginal numerical
  drift.  No performance conclusion can be drawn from this iteration.
- Outcome: **implementation failure; diagnose and correct the column-major
  canonical-left/right slot mapping before timing the direction**.
- Trajectory: benchmark aborted before the trajectory snapshot step; console
  output is preserved in `_bench_output.txt` for this commit.

### Iteration 44 — correct grouped-destination strides for raw2 prepare

- Change: retained iteration 43's two-phase dataflow and fixed all eight
  transformed stores to use the compact grouped-plane strides.  The failed
  prototype had formed destination addresses with the original N-leading
  quadrant strides instead of the N/2-leading workspace strides.
- Correctness: all 20 formal cases now report `CORRECT=True`; IEEE and TF32
  errors match the established baseline.  A direct workspace diagnostic also
  confirmed the M4/M5 products and final counter state were already correct;
  only the transformed-plane addresses were defective in iteration 43.
- BAG geometric-mean latency: **0.857485 ms**, 5.47% above the independent
  **0.813021 ms** product-major baseline.
- Targeted result: N=512 TF32 rose from **0.0225 ms** to **0.0328 ms** in both
  layouts (about 45.8% slower); N=1024 rose from **0.0922 ms** to
  **0.1116 ms** (about 21.0% slower).
- Analysis: correctness validates the two-launch phase mapping, counter=2
  protocol, and row/column operand slots.  Performance rejects the fixed edge
  owners: only `Pm+Pn` CTAs perform all input transforms, so the first launch
  ends with a long tail even though its aggregate CTA wave count is unchanged.
- Outcome: **reject the edge-owner schedule, not yet the raw2 dataflow**.  Test
  a rotating unique owner per K-step so the same prepare work and fragment
  reuse are distributed across all raw-product CTAs.
- Trajectory: `trajectory/iter-44-20260831-222310`.

### Iteration 45 — rotate raw2 prepare owners by K step

- Change: kept iteration 44's algebra, two launches, transformed-plane traffic,
  and counter protocol, but selected the unique canonical-left owner as
  `program_n = k_step mod Pn` and the unique canonical-right owner as
  `program_m = k_step mod Pm`.  `Pm/Pn` are compile-time constants.  This
  distributes prepare across every M4/M5 CTA while preserving one writer per
  transformed tile and reuse of the current raw dot fragment.
- Correctness: all 20 formal cases report `CORRECT=True` with baseline IEEE
  and TF32 error levels.
- BAG geometric-mean latency: **0.820979 ms**, 0.98% above the independent
  **0.813021 ms** product-major baseline, but 4.26% below the fixed-edge
  iteration 44.
- Targeted result: N=512 TF32 improved from iteration 44's **0.0328 ms** to
  **0.0256 ms**, yet remains about 13.8% slower than the **0.0225 ms** winner.
  N=1024 improved from **0.1116 ms** to **0.0932 ms**, one approximately
  1.024-us event bin above the **0.0922 ms** winner.
- Analysis: balancing confirms that edge-owner tail skew caused most of the
  prior regression.  N=512 still pays for a 64-CTA raw launch that underfills
  82 SMs and for the phase boundary; N=1024 is effectively near the timer's
  resolution and needs component timing before any further specialization.
- Outcome: **reject raw2 for N=512; retain N=1024 only as an experimental
  candidate pending phase timing/resource inspection**.  Do not replace the
  published small-TF32 path yet.
- Trajectory: `trajectory/iter-45-20260831-222455`.

### Iteration 46 — reduce N=1024 raw2 pipeline depth

- Change: disabled raw2 for N=512 after its confirmed underfill regression and
  changed only the N=1024 raw two-product phase from three pipeline stages to
  two.  The five-product/last-completer phase retained the established
  three-stage configuration.
- Correctness: all 20 formal cases report `CORRECT=True` with the expected IEEE
  and TF32 error levels.
- BAG geometric-mean latency: **0.825452 ms**, 1.53% above the independent
  **0.813021 ms** product-major baseline.
- Targeted result: N=1024 TF32 remained **0.0932 ms** in both layouts, exactly
  the same measured event bin as iteration 45 and still above the **0.0922 ms**
  winner.  N=512 TF32 measured **0.0266/0.0256 ms** for column/row major even
  though raw2 was disabled, rather than returning to the published
  **0.0225 ms**.
- Analysis: reducing raw-phase pipeline depth does not improve end-to-end
  latency.  Moreover, merely gating off raw2 is insufficient to recover the
  small baseline while the generalized kernel signature and phase machinery
  remain in the source; that experimental infrastructure must not remain in a
  rejected final implementation.
- Outcome: **reject the two-stage raw2 variant**.  Inspect its compiled resource
  count once to determine whether the result is occupancy- or instruction-path
  limited, then close raw2 and restore the exact iteration-39 implementation
  before testing a genuinely different direction.
- Trajectory: `trajectory/iter-46-20260831-223005`.

### Iteration 47 — restore the exact product-major winner

- Change: removed the raw2 phase machinery, extra raw-quadrant arguments,
  interleaved K-loop preparation, and split host dispatch.  The resulting
  `solution/bag_triton/bag.py` is byte-for-byte identical to the iteration-39
  published implementation before benchmarking.
- Resource conclusion from the rejected branch: changing the N=1024 raw phase
  from three to two stages lowered its compiled allocation from **138** to
  **128 registers/thread**, with zero stack/local spill in both cases, but did
  not change the **0.0932 ms** end-to-end event bin.  Extra operand work,
  transformed stores, and the phase boundary—not register occupancy alone—set
  the remaining raw2 cost.
- Correctness: all 20 formal cases report `CORRECT=True`.
- BAG geometric-mean latency: **0.813207 ms**, within 0.03% of the independent
  **0.813021 ms** iteration-39 replay.
- Targeted result: N=512 TF32 returned to **0.0225 ms** and N=1024 TF32 to
  **0.0922 ms** in both layouts.  This confirms that the gated-off generalized
  raw2 kernel itself caused the small-bucket regression seen in iteration 46.
- Outcome: **accept as the clean experimental base, not as a new winner**.
  Close the raw2 direction and require future candidates to preserve the
  untouched seven-product specialization as an explicit fallback.
- Trajectory: `trajectory/iter-47-20260831-223302`.

### Iteration 48 — stage last-completer product loads

- Change: kept the product GEMM, counter protocol, decode vector width, and
  exact left-to-right BAG arithmetic unchanged, but moved the seven product
  plane loads next to their first output use.  The source-level peak falls
  from seven simultaneously named product vectors to about five, with the aim
  of shortening decoder live ranges.
- Correctness: all 20 formal cases report `CORRECT=True` with unchanged error
  levels.
- Targeted result: N=512 TF32 remained **0.0225 ms** for both layouts.  N=1024
  TF32 measured **0.0932/0.0942 ms** for column/row major, versus the clean
  baseline's **0.0922/0.0922 ms**.  The candidate therefore does not improve
  either target bucket and regresses one by two approximately 1.024-us event
  bins.
- Full-suite BAG geometric mean was **0.811781 ms**, nominally below the
  **0.813207 ms** adjacent baseline, but the apparent change comes from
  N=4096/8192 specializations whose generated code is unaffected and whose
  absolute latency varies between runs.  It is not evidence for this decoder
  edit.
- Analysis: source ordering alone either does not reduce compiled live ranges
  or perturbs memory issue scheduling without helping occupancy.  Inspect the
  generated resource count once; do not select a candidate that loses its
  directly affected buckets on the strength of unrelated large-shape noise.
- Outcome: **reject** and restore the exact iteration-47 source before the next
  last-completer/dataflow candidate.
- Trajectory: `trajectory/iter-48-20260831-223726`.

### Iteration 49 — shallower N=1024 IEEE grouped pipeline

- Change: restored the exact iteration-47 last-completer decoder, then changed
  only full N=1024 IEEE (`half=512`, `BK=32`) from three pipeline stages to
  two.  The other 19 formal specializations remain source-identical to the
  clean baseline.
- Motivation/resource evidence: the baseline N=1024 IEEE product kernel uses
  **96 registers/thread** and about **32 KiB shared memory/CTA**, allowing only
  roughly three CTAs per sm86 SM.  This was the strongest remaining occupancy
  anomaly.  Separately, iteration 48's staged decoder compiled to **64**
  registers at N=512 but remained **80** at N=1024; despite the N=512 reduction
  from the baseline's 80, its measured latency did not improve.
- Correctness: all 20 formal cases report `CORRECT=True`.
- Targeted result: N=1024 IEEE measured **0.1300/0.1290 ms** for column/row
  major, versus the adjacent clean baseline's **0.1290/0.1270 ms**.  N=512
  IEEE and all four small TF32 cases retained their baseline event bins.
- Full-suite BAG geometric mean was **0.813653 ms**, 0.05% above iteration 47;
  selection is based on the directly affected IEEE bucket, which consistently
  moved in the wrong direction.
- Analysis: the third pipeline stage hides enough operand latency to outweigh
  its lower static occupancy.  Occupancy is a constraint rather than a direct
  objective, and this bucket is not improved by reducing it mechanically.
- Outcome: **reject**; restore the three-stage N=1024 IEEE specialization.
- Trajectory: `trajectory/iter-49-20260831-223954`.

### Iteration 50 — ordered 6+1 M7 write-once epilogue

- Change: restored the three-stage N=1024 IEEE grouped kernel.  For that bucket
  only, the unchanged product kernel launches six product planes M1--M6, then
  a dedicated `32x32x32` four-warp kernel computes the seventh genuine
  half-size GEMM (M7) and directly writes all four output quadrants from its
  FP32 accumulator plus M1--M6.  It neither stores nor reloads M7 and uses no
  counter or atomic.  TF32 and every other size retain their prior paths.
- Dataflow/measurement boundary: the candidate still has three timed launches,
  exactly like IEEE's prepare + seven-product + decode baseline; all seven BAG
  products, the phase boundary, and the write-once epilogue are included.
- Correctness: all 20 formal cases report `CORRECT=True`; IEEE errors match the
  clean baseline, confirming both row-major and transposed column-major M7
  operand mappings.
- Targeted result: N=1024 IEEE measured **0.1352/0.1331 ms** for column/row
  major, versus iteration 47's **0.1290/0.1270 ms**.  Both layouts regress by
  about 4.8%, while the untouched N=512 IEEE and all small TF32 buckets retain
  their baseline event bins.
- Full-suite BAG geometric mean was **0.817252 ms**, 0.50% above the clean
  **0.813207 ms** replay.
- Analysis: eliminating one product-plane write/read and the standalone
  pointwise epilogue cannot offset serializing the final GEMM behind M1--M6.
  The 256-CTA final phase has adequate work supply, so the loss is the ordered
  phase boundary and reduced cross-product scheduling freedom rather than the
  original 64-CTA underfill concern.
- Outcome: **reject**.  Preserve this as evidence for the papers' AB/ABC
  crossover: write-once fusion is not automatically beneficial when it turns
  a batched GPU GEMM into ordered phases.
- Trajectory: `trajectory/iter-50-20260831-224844`.

### Iteration 51 — widen the ordered-final M7 tile

- Change: kept iteration 50's IEEE-only ordered 6+1 dataflow and widened only
  the final M7/write-once kernel from `32x32x32` to `32x64x32`.  Its grid falls
  from 256 to 128 CTAs, still above the RTX 3090's 82 SMs, to increase operand
  reuse and reduce CTA/address overhead.
- Resource motivation: the rejected `32x32` final kernel compiled to only
  **54 registers/thread**, with zero stack/local spill, while the unchanged
  six-product phase remained at **96 registers/thread**.  The final tile was
  therefore widened rather than assuming register pressure caused iteration
  50's loss.
- Correctness: all 20 formal cases report `CORRECT=True`.
- Targeted result: N=1024 IEEE measured **0.1372/0.1301 ms** for column/row
  major.  Widening improved row-major from iteration 50's 0.1331 ms but still
  loses to the clean 0.1270 ms baseline; column-major became slower than both
  iteration 50's 0.1352 ms and the 0.1290 ms baseline.
- Full-suite BAG geometric mean was **0.814828 ms**; the lower number than
  iteration 50 comes from unchanged large-shape variation and does not alter
  the target-bucket rejection.
- Analysis: final-phase tile reuse and CTA count trade differently by layout,
  but neither point approaches the batched product-major baseline.  A further
  `64x64` point would reduce the final grid below one CTA per SM and cannot
  address the demonstrated ordered-phase serialization.
- Outcome: **reject and close ordered 6+1**.  Restore the exact iteration-47
  product-major path rather than retaining a layout-specific regression.
- Trajectory: `trajectory/iter-51-20260831-225016`.

### Iteration 52 — integer last-completer tickets

- Change: restored the exact product-major/last-completer implementation and
  reinterpreted the otherwise-unused FP32 counter plane as aligned `int32`.
  Prepare writes integer zero; each small-TF32 product CTA performs the same
  GPU-scope acq_rel atomic add of integer one; the unique `old==6` CTA decodes.
  Allocation, counter count, memory ordering, product traffic, and BAG algebra
  are unchanged.
- Correctness: all 20 formal cases report `CORRECT=True`, including both
  layouts and repeated timed reuse of the aliased workspace plane.
- Targeted result: all four affected TF32 cases are exactly tied with the clean
  baseline at the benchmark resolution: **0.0225 ms** for N=512 and
  **0.0922 ms** for N=1024 in both layouts.
- Full-suite BAG geometric mean was **0.809415 ms**, nominally below the clean
  replay, but IEEE and large-shape specializations do not contain the changed
  atomic and varied materially in this run.  That aggregate movement is not
  attributed to the candidate.
- Analysis: integer and FP32 tickets have indistinguishable end-to-end latency
  at the approximately 1.024-us CUDA-event resolution.  Inspect SASS and one
  adjacent focused A/B only to determine whether an unresolved sub-bin signal
  exists; absent a repeatable direct-bucket win, keep the FP32 implementation
  for source simplicity.
- Outcome: **formal tie / provisional reject**.
- Trajectory: `trajectory/iter-52-20260831-225557`.

### Stall reassessment after iterations 48--52

- Last-completer register scheduling lowered N=512 from 80 to 64 registers but
  did not improve latency; N=1024 remained at 80 and regressed.
- N=1024 IEEE resource reductions (`stages=2`, and a separate read-only BK16
  A/B) lowered shared-memory pressure but lost the K-loop latency hiding of the
  selected `BK32/stages3` configuration.
- The papers' ordered AB/ABC candidate was implemented as a real six-plus-one
  BAG schedule.  Its final M7 kernel was healthy (54 registers, zero spill),
  yet both `32x32` and `32x64` variants lost because the phase boundary removes
  batched cross-product scheduling freedom.
- Integer counter SASS uses the same strong GPU-scope global atomic and the
  same 80-register allocation, but Triton additionally emits lane-id, vote,
  find-first, popcount, shuffle, and integer-add instructions for warp
  aggregation.  It does not reduce the seven serialized arrivals per tile.
- Decision: close load-order, local occupancy, ordered-signature, and atomic
  representation axes.  The remaining proposed self-accumulator optimization
  has only a 0.28/1.12-us absolute DRAM upper bound for N=512/1024 and raises
  accumulator live ranges; after five non-winning iterations it does not meet
  the evidence threshold for more production complexity.

### Iteration 53 — final exact-winner restoration

- Change: removed the integer-counter reinterpretation and restored
  `solution/bag_triton/bag.py` byte-for-byte to the published iteration-39/47
  product-major implementation before timing.
- Correctness: all 20 formal cases report `CORRECT=True`.
- BAG geometric-mean latency: **0.812709 ms**, 0.06% below iteration 47's
  **0.813207 ms** and effectively identical to the published independent
  **0.812665 ms** result.
- Targeted result: N=512 TF32 is **0.0225 ms** and N=1024 TF32 is
  **0.0922 ms** in both layouts.  N=512 IEEE is **0.0317/0.0287 ms** and
  N=1024 IEEE is **0.1280/0.1280 ms** for column/row major.
- Outcome: **retain iteration 39 as the winner**.  The new implementation work
  supplied correctness-checked negative results and resource boundaries but
  did not produce a candidate that beats the directly affected small buckets.
- Trajectory: `trajectory/iter-53-20260831-225735`.

### Final verification after the paper-guided optimization phase

- Selected source: the exact iteration-39 product-major small-TF32
  last-completer implementation; `src/bag_triton/bag.py` and
  `solution/bag_triton/bag.py` are byte-for-byte identical.
- Full functional regression passed all **28** BAG numerical/interface cases
  and all **26** primitive cases.  Coverage includes IEEE/TF32, row-major,
  column-major, sliced positive strides, non-aligned sizes, non-default caller
  streams, repeated multistream and seven-stream schedules, grouped/hybrid
  paths, and concurrent primitive atomic accumulation.
- Independent final benchmark reports `CORRECT=True` for all 20 cases, BAG
  geometric-mean latency **0.815398 ms**, and geometric-mean `GEMM/BAG`
  **0.919152x**.
- The priority TF32 buckets remain stable at **0.0225 ms** for N=512 and
  **0.0922 ms** for N=1024 in both layouts.  The 0.33% aggregate difference
  from iteration 53's 0.812709 ms is dominated by unchanged IEEE/large-shape
  variability; source and generated target specializations are unchanged.
- Final decision: no paper-guided candidate from iterations 40--52 beats the
  directly affected small buckets.  Keep the already-published winner and the
  complete negative-result/resource log rather than publishing a noisy or
  layout-specific regression.
- Trajectory: `trajectory/final-20260831-225852`.

## 2026-10-17 — Standalone CUDA FP32 beta=1 Strassen cost baseline

- User objective: measure full-size cuBLAS, half-size leaf GEMMs and the existing
  fused primitives before modifying their implementation. No kernel edits.
- Added `scripts/benchmark_strassen_fp32.cu`, `run_strassen_fp32.sh`,
  `summarize_strassen_fp32.py`, and `STRASSEN_FP32_BASELINE.md`.
- Platform: g02, one allocated NVIDIA A100 80GB PCIe, CUDA 12.2.140,
  cuBLAS 12.2.5, column-major FP32, alpha=beta=1,
  CUBLAS_PEDANTIC_MATH and NVIDIA_TF32_OVERRIDE=0.
- Full run: 16 parent sizes from 512 to 20480, 11 cases per size,
  5 randomized timing rounds, 880 timing rows. Every case passed FP64
  checks after 1 and 3 accumulated calls (352 checks); maximum sampled
  relative L2 error 2.933386e-6. Small smoke outputs were checked exhaustively.
- At N=2048, full cuBLAS=0.977161 ms, parent-stride leaf=0.128811 ms,
  primitive GEMM plus atomic output=0.258245 ms, sum of seven isolated
  primitives=2.101285 ms. At N=8192, the corresponding costs are
  58.451199/7.527917/9.065007/73.152671 ms. At N=20480 they are
  950.978597/118.997670/121.071279/893.870758 ms.
- Seven-leaf and seven-primitive sums are serial cost proxies, not measured
  end-to-end speedups or hardware bounds. Medium-size primitive computation
  and atomic output need attention; large-size primitive costs approach leaves.
- Additional verification: memcheck=0 errors for representative small/medium/
  large/huge kernels. NCU sampled cuBLAS SGEMM tensor instructions=0;
  standalone SASS contains FFMA and no Tensor Core instruction mnemonics.
- The first compile exposed a benchmark error-macro name collision, corrected
  before all successful measurements. The main benchmark completed, then its
  wrapper reported an EOF parse error after being edited while running.
  Wrapper execution was hardened and a clean replay across four buckets
  passed with exit code 0, 44 timing rows and 88 checks. Compiled CUDA source
  is identical to the main run. Full evidence is in `execution_audit.json`.
- Artifacts: `benchmark_results/strassen_fp32_a100_20261017/` (raw rounds,
  correctness, source snapshots and hashes, build/environment logs, telemetry,
  report, CSV summaries, PNG/PDF figures, and instrumentation evidence).

## Strassen FP32 one-level — iter 1 (2026-10-17)

- **Direction:** controlled `primitive_large` epilogue (same compute, atomic vs beta=1 vector add); SIMT double-buffer/prefetch versus synchronous loop; BK=8/16, CTA=64x32/64x64/128x64/128x128 and per-thread 4x4/8x4/8x8 blocking. Original medium/huge are measured controls. Complete seven-product API is included for subsequent end-to-end testing.
- **Profile evidence:** original 1024 leaf, 256 CTAs x64 threads, 113 registers/thread, theoretical occupancy 25%, achieved 6.60%, waves/SM 0.30. Grid size as well as register limit matters. Nsight reports 54% excessive global sectors and 39% excessive shared wavefronts.
- **Command:** `srun --jobid=108127 --overlap bash scripts/bench_strassen_onelevel.sh iter-1 benchmark_results/strassen_onelevel_20261017/iter1_ablation --mode ablation --sizes 2048,8192,20480 --rounds 3 --target-ms 100`.
- **Correctness:** PASS, 240 FP64 checks (one and three accumulated calls); maximum sampled relative L2 2.933386e-6, unchanged 3e-5/1e-4 thresholds. 360 raw timing rows (40 cases x3 sizes x3 rounds). Execution exit 0; source and environment archived.
- **N=2048 result:** cuBLAS full 0.976608 ms, leaf approximately 0.1288 ms. Same-body atomic GEMM 0.258395 ms vs normal 0.253569 ms (1.9%); M7 0.327541 vs 0.324752 ms (0.9%). Best candidate `pipe64x32_k16_r4` atomic: GEMM 0.170362 ms, M7 0.214696 ms (1.52x/1.53x vs original-large). Pipeline by itself (`pipe64_k8_r8`) regresses; reducing register tile and changing CTA shape matters more than removing atomics.
- **Large regression:** at N=8192 huge GEMM/M7 7.911/8.444 ms beats legacy 8.778/10.710 ms; small winner 10.246/13.401 ms regresses. At N=20480 original huge remains best, GEMM/M7 119.883/130.526 ms versus small winner 165.764/200.717 ms. No universal replacement is justified.
- **Decision:** keep ablations and separate size-dependent choices. Next measure genuine seven-product execution, including complete joins and nonzero-C accumulation. Do not infer end-to-end speedup from seven isolated leaves.
- **Artifacts:** `benchmark_results/strassen_onelevel_20261017/iter1_ablation/results.{rounds,correctness,summary}.csv`, source SHA256 manifest, build log, telemetry; baseline Nsight CSV in the parent directory.

## Strassen FP32 one-level — iter 2 (2026-10-17)

- **Direction:** retain the complete serial/7-stream/dependency schedules and add four-launch paired execution `(M1,M2),(M3,M4),(M5,M6),M7`. Generic pairs support ordinary beta=1 addition because active outputs within a pair are disjoint; original huge pairs retain atomic updates. Also reject incompatible unmasked-kernel shapes at the API boundary.
- **Command:** `srun --jobid=108127 --overlap bash scripts/bench_strassen_onelevel.sh iter-2 benchmark_results/strassen_onelevel_20261017/iter2_endtoend --mode endtoend --sizes 2048,8192,20480 --backends legacy,huge,pipe64x32_k16_r4 --rounds 3 --target-ms 100`.
- **Correctness:** PASS, all 72 complete-result FP64 checks (12 cases x3 sizes x1/3 accumulated calls), maximum sampled relative L2 3.1688436e-6. All seven products, transforms, output updates and required stream joins are timed. Source and environment archived; 108 timing rows; exit 0.
- **N=2048:** cuBLAS 0.981458 ms; original selector serial/seven-stream 2.219587/1.248484 ms; new generic seven-stream 1.154094 ms (0.8504x vs cuBLAS). Generic dependency/paired-atomic/paired-normal 1.316212/1.331828/1.365006 ms. Four launches are not a win over full concurrency. Huge paired 1.655909 ms versus seven-stream 1.172516 ms. Small isolated gains have not crossed the end-to-end cuBLAS baseline.
- **N=8192:** cuBLAS 57.513301 ms; huge seven-stream 56.465408 ms (1.0186x), paired 56.478378 ms, serial 58.501801 ms; original selector seven-stream 70.859777 ms. Using the huge tile at leaf 4096 is useful here.
- **N=20480:** cuBLAS 958.263265 ms; huge seven-stream 882.381836 ms (1.0860x), paired 883.701090 ms, serial 887.830892 ms. Small generic winner regresses to 1328.032796 ms. Preserve size-dependent choices rather than replacing every primitive.
- **Decision fixed before final replay:** N=2048 `Pipe64x32/SevenAtomic`; N=8192 and 20480 `Huge/SevenAtomic`. Final verdict will use five rounds, a 200 ms target and same-run cuBLAS, plus memcheck on partial tiles/huge pairs, racecheck on the shared-memory pipeline, Tensor instruction audit and a candidate Nsight profile. Paired schedules remain documented experimental controls, not recommended defaults.
- **Reproducibility:** add immutable-build replay, verification and plotting helpers; document the public API, shape/alignment limits, timing and numerical contract. The runner's clean-checkout header fallback was checked byte-for-byte against current local primitive headers. This changes no compiled reference data. Original first-round sources and all negative controls remain archived.

## Strassen FP32 one-level — final validation (2026-10-17)

- **Code selected:** iteration 2 (`077b63b`), which retains the fast original huge path and the measured generic/paired controls. No arithmetic source changes after that build. All four current CUDA headers match its archived source byte for byte (`current_source_match.json`).
- **Final protocol:** replay the immutable iteration-2 executable; five randomized rounds, 200 ms target, minimum three timed calls, same-run cuBLAS. Small phase N=2048 includes all 12 cases; large phase N=8192/20480 includes cuBLAS and all original-selector/huge schedules (six cases), after the generic candidate's large regressions were already retained in both signal phases. Final 120 raw timing rows and 48 FP64 checks, all pass; maximum sampled relative L2 3.16920876e-6.
- **Preselected-plan verdict:** N=2048 generic seven-stream 1.194171 ms vs cuBLAS 0.978687 ms (0.8196x); N=8192 huge seven-stream 56.723457 vs 57.929214 ms (1.0213x); N=20480 huge seven-stream 886.204102 vs 960.525635 ms (1.0839x). End-to-end gain materializes at the two larger measured points, not at N=2048.
- **Selection uncertainty retained:** final N=2048 huge seven-stream is 1.173235 ms (0.8342x), faster than the preselected generic candidate, reversing the signal ranking. Final paired-huge medians 56.527103/885.076986 ms are slightly better at large sizes (1.0248x/1.0852x); they are descriptive post-replay minima, not the preselected result. Seven-stream huge remains a stable baseline; no claim that the small generic candidate or a sub-percent schedule difference is universally best.
- **Profile:** generic GEMM leaf registers 56 vs original-large 113, achieved occupancy 23.47% vs 6.60%, theoretical 56.25% vs 25%, waves/SM 0.53 vs 0.30. Reported excessive global sectors 6% vs 54%, shared wavefronts 3% vs 39%. Single-kernel diagnostic metrics are not multi-stream occupancy measurements or primary timing data.
- **Independent checks:** memcheck generic paired-normal N=120/2056 and huge paired-atomic N=256, plus shared-memory racecheck N=120, all exit 0 and pass FP64 validation. N=120 is exhaustive and exercises partial tiles/K tails. Custom SASS has 37536 static FFMA instructions, zero Tensor opcode bases; dynamic cuBLAS Tensor instructions are zero. Profile, Tensor measurement and cuobjdump commands also exit 0.
- **Audit correction:** original aggregation exited 1 after all six instrument commands passed because a broad regex counted the `.MMA` modifier in `HFMA2.MMA` as a Tensor opcode. Fixed the parser to inspect instruction base opcodes, separated it into `audit_strassen_onelevel.py`, and re-audited the unchanged logs successfully. Original failure, original/fixed verification scripts, per-command zero statuses, corrected audit source/status/output and final summary are retained. No performance/correctness run was rewritten to hide this failure.
- **Delivery:** `benchmark_results/strassen_onelevel_20261017/REPORT.md`, raw records, CSV comparisons, Nsight and compiler resources, PNG/PDF figures, standalone API/runner/replay/verification documentation. No exact crossover claim, A800/V100 claim, two-level result or hardware-floor claim. A100 allocation 108127 released after all GPU work; original user .gitignore change remains untouched.

## Strassen FP32 batching — iter 3 (2026-10-17)

- **Direction:** paper section 3.3 four-variant batching: three grid.z=2 pairs plus M7, submitted concurrently on four nonblocking streams with caller readiness and completion joins. Existing ordered-pair and seven-stream controls are unchanged.
- **Command:** `srun --jobid=108128 --overlap bash scripts/bench_strassen_onelevel.sh iter-3 benchmark_results/strassen_batching_20261017/iter3_signal --mode endtoend --sizes 2048,8192,20480 --backends huge,pipe64x32_k16_r4 --rounds 3 --target-ms 100`.
- **Platform:** A100 80GB PCIe on g03, GPU UUID GPU-4db2fb47-d819-bfee-f2f8-e6c84299d435. This differs from the prior g02 allocation; ratios use this run's cuBLAS.
- **Correctness:** PASS, 72 FP64 checks (all 36 size/case combinations after 1 and 3 accumulated calls), max relative L2 3.1725278e-6. 108 timing rows; build and benchmark exit 0.
- **N=2048:** cuBLAS 0.976947 ms; generic four-batched 1.142053 ms (0.8554x), generic seven-stream 1.164345 ms; Huge four-batched/seven-stream 1.160418/1.167183 ms. Small improvements do not beat cuBLAS. Seven-stream CV about 3%; independent replay required before selection.
- **N=8192:** cuBLAS 57.324203 ms; Huge seven/paired/four-batched 55.831212/55.899134/55.920639 ms (about 1.0267/1.0255/1.0251x). No practically large batching advantage.
- **N=20480:** cuBLAS 958.117188 ms; Huge seven/paired/four-batched 881.360514/881.421305/881.474528 ms (about 1.087x). Generic paths remain about 1340 ms and cannot replace Huge at large sizes.
- **Decision:** retain Huge seven-stream as the stable large baseline, keep four-batched as an experimental option, extend the one-level curve, and test the paper's workspace-based hybrid two-level structure. No universal batching win or crossover claim.
- **Artifacts:** `benchmark_results/strassen_batching_20261017/iter3_signal/`, immutable source manifest, build/environment/telemetry logs, raw timing and correctness CSVs.

## Strassen FP32 hybrid — iter 4 (2026-10-17)

- **Direction:** workspace-based outer Strassen and fused one-level inner calls, with three reusable compact half-size scratch matrices. Transform inputs and reset the inner output in a preparation kernel; compute seven inner Strassen products; merge each outer product into the original nonzero C on the caller stream. Every initialization, transform, product, merge and stream dependency remains inside a complete call. No ordinary GEMM fallback.
- **Benchmark additions:** optional schedule filter, independent input seed offset (matching device initialization and FP64 reference), and cudaProfilerStart/Stop capture of selected complete calls. Default baseline seed values and tolerances unchanged.
- **Command:** `srun --jobid=108128 --overlap bash scripts/bench_strassen_onelevel.sh iter-4 benchmark_results/strassen_batching_20261017/iter4_hybrid_smoke --mode endtoend --sizes 120,128,256,512 --backends huge,pipe64x32_k16_r4 --schedules seven_atomic,four_batched_atomic --hybrid --validate-only`.
- **Correctness:** PASS, 48 checks across all legal one-level/hybrid configurations after 1 and 3 calls; max relative L2 8.4914467e-7. Full output checks at N=120/128 include masked partial tiles; Huge hybrid becomes legal at N=512 (quarter leaf 128). Build and benchmark exit 0. No performance claim from this smoke run.
- **Decision:** implementation passes initial correctness. Reuse this immutable build for the full A100 curve, hybrid large-size signal, independently seeded replay, full-call Nsight Systems capture and sanitizer checks; then freeze candidates for A800.
- **Artifacts:** `benchmark_results/strassen_batching_20261017/iter4_hybrid_smoke/`.

## Strassen FP32 batching/hybrid — final A100 validation (2026-10-17)

- **Frozen arithmetic:** iteration 4 (`64e2695`); current five CUDA headers match its archived build byte-for-byte. No arithmetic changes after the smoke build. Full-call comparisons retain Huge seven-stream and four-batched one-level/hybrid candidates, with no dense GEMM fallback.
- **Measurements:** two complete 12-size curves, input seed offsets 0 and 1009, five randomized rounds, 200 ms target, minimum three timed calls; 11 additional dense crossover sizes at each seed; six small sizes comparing Huge and generic seven/four schedules. Including iteration 3 and the iteration-4 smoke, 1368 raw timing rows and 624 regular FP64 checks, all pass. Maximum sampled relative L2 4.974704238e-6; 3e-5/1e-4 limits unchanged.
- **N=20480:** seed 0 cuBLAS/one-level/hybrid = 960.639974/884.775960/803.946533 ms, hybrid 1.194905x; seed 1009 = 957.859538/883.803141/802.924235 ms, hybrid 1.192964x. Hybrid reduces one-level latency by 9.14%/9.15%. Both use the preselected seven-stream schedule, not a post-hoc schedule minimum.
- **Other large sizes:** hybrid at N=10240/12288/16384 also exceeds the 3% improvement-over-one-level investigation threshold in both curves. At N=8192 the hybrid gain over one level is smaller and varies by run; report actual per-size values rather than a universal threshold.
- **Crossover:** dense one-level seven-stream medians at N=4096 are 1.0028x/0.9999x; N=4352 1.0150x/1.0165x; N=5632 only 1.0030x/1.0029x. Hybrid at N=6144 is 0.9981x/0.9990x and N=6656 1.1016x/1.0998x. Finite measured points are non-monotonic; these are not guarantees for every larger or unmeasured dimension.
- **Small sizes and scheduling:** N=2048 still loses to cuBLAS. Paper four-variant batching does not consistently beat seven-stream Huge across sizes. Retain explicit choices and all negative controls; do not translate isolated primitive wins into complete-call gains.
- **Correctness/instrumentation:** masked batching memcheck N=120/2056, generic hybrid memcheck N=128/256 with seed offset 1009, Huge hybrid memcheck N=512 and shared-memory racecheck N=128 pass (12 additional FP64 checks). Five full-call Nsight Systems captures pass with exact expected launch counts: 20/140/80/80 kernels for 20 cuBLAS/seven/batched/generic-batched calls; 189 kernels for three hybrid calls. Static SASS has 37536 FFMA and zero Tensor opcodes; cuBLAS dynamic Tensor instructions =0.
- **Timeline decision:** N=2048 seven-stream GPU busy union 1.153485 ms and internal idle 0.008176 ms/call, versus captured cuBLAS busy 0.977215 ms. This does not support prioritizing Graph for the roughly 0.185 ms complete-call deficit. Profiler durations are excluded from primary speedups and are not physical bounds. Hybrid N=20480 diagnostic preparation/reset 12.385588 ms, product interval union 770.318959 ms, merge 9.120130 ms; captured envelope 791.901839 ms differs from ordinary event timing and is used only for cost attribution.
- **Tool failure and recovery:** the initial followup wrapper exited 9 at NCU because /tmp/nsight-compute-lock belongs to another user. All prior timing, validation, sanitizer and five timeline stages had completed. Preserve the original failure in `verification/tensor_cublas_initial/` and `a100_followup.initial.status.json`. After checking no other NCU collection was active, use a private TMPDIR as documented by NVIDIA, rerun only Tensor audit, export SASS, and analyze successfully. Original performance/correctness rows were not rerun or rewritten. Fixed future verification source and actual repair commands are archived.
- **Memory/API:** hybrid reuses three compact half-size matrices, 3*N*N bytes (1.171875 GiB at N=20480), plus original A/B/C. All required scratch reset, transform, product, merge and stream joins are timed; workspaces require exclusive caller ownership until completion.
- **A800:** partition GPUs are occupied and other jobs are queued. Job 108131 is pending in persistent tmux session `strassen-a800-20261017`; allocation payload automatically runs the same fixed build, two input seeds and 12-size curves, refreshes analysis and releases resources. No A800 performance result is claimed before completion. Logs, payload snapshot, current queue and job metadata are archived.
- **Delivery/resource state:** report, curves and timeline PNG/PDF, algorithm/crossover CSVs, immutable sources/hashes, raw rounds/checks/environment/telemetry, API/runner instructions and successful instrument evidence under `benchmark_results/strassen_batching_20261017/` and `scripts/STRASSEN_BATCHING.md`. A100 allocation 108128 released after all GPU work. Preexisting user `.gitignore` modification remains unstaged and untouched.

## Crossover acceptance correction and bounded configuration audit — 2026-10-17

- **User feedback:** the crossover has not reached the paper's reported value. Mark the one-level N=1536 target unmet; the large hybrid speedup near 1.19x does not fulfill it. Persist this distinction in HINTS.md, the report generator, the generated report and runner documentation.
- **Paper audit:** the local paper abstract reports one level N=1536 and hybrid two levels N=7680 on V100 SXM2 / CUDA 9.1. The fixed Huge seven-stream A100 dense measurements first clearly win at N=4352 and N=6656, respectively, across both input seeds. These are measured points with candidate max below same-run cuBLAS min, not monotonic thresholds or same-hardware reproduction. The earlier commentary saying both crossovers were later was incorrect; the hybrid measured positive point is earlier.
- **Selection gap:** the paper searches strategies/block sizes; the existing main curve fixes Huge. Keep the fixed main curve identifiable and do not equate its crossover with the best over all possible configurations.
- **Experiment:** unchanged immutable arithmetic build `64e2695`, all 11 existing backends with seven streams plus Huge/Pipe64x32 four-batched schedules, N=1024/1280/1536/1792/2048/2560/3072. Five randomized rounds at 200 ms, pure FP32 C<-AB+C, alpha=beta=1, TF32 disabled, same-run PEDANTIC cuBLAS, unchanged errors and one/three accumulated-call checks. No solution modification and therefore no new optimization iteration number.
- **Selection/replay:** select each size's lowest-latency configuration only on seed 0; report the identical case on seed 1009. At N=1536/2048 this is `pipe64_k16_r4_seven_atomic`. Seed 1009 cuBLAS/candidate: 0.439561/0.518131 ms at N=1536 (0.8484x; another 15.16% candidate latency reduction to tie), 0.987207/1.158947 ms at N=2048 (0.8518x; 14.82% to tie). Every existing candidate still loses at both target sizes in both seeds. N=2560/3072 select Huge four-batched but also remain slower than cuBLAS.
- **Validation:** 980 timing rows and 392 FP64 checks all pass; max relative L2 1.54791789328e-6. Replay verifies immutable source SHA256. Analysis asserts round counts, per-case median agreement, input seed, precision and error status. Bash/Python syntax and git whitespace checks pass. A100 allocation 108132 completes and releases automatically; A800 108131 remains queued; unrelated H100 allocation unchanged.
- **Outcome/next work:** one-level crossover remains unmet after existing-configuration selection. Target the roughly 15% complete-call gap at N=1536/2048 through variant-specific primitive profiling (leaf=768/1024, registers, CTA concurrency, memory and atomic output updates), then implementation changes and complete seven-product validation. Earlier N=2048 internal kernel gaps ~0.008 ms do not explain the ~0.185 ms call deficit; neither Graph nor hardware differences are established explanations or physical bounds.
- **Artifacts:** `benchmark_results/strassen_crossover_20261017/REPORT.md`, raw independent-seed data, source verification, environment, telemetry, allocation/accounting and selection CSV; updated paper comparison in `benchmark_results/strassen_batching_20261017/REPORT.md`. Preexisting root `.gitignore` modification remains untouched and unstaged.

## Strassen iteration 5 — aligned XOR-swizzled B tiles (2026-10-17)

- **Direction:** use NCU evidence from the four operand/output variants at leaf=768/1024 to replace padded B shared-memory rows with an aligned XOR swizzle. Preserve original backends, add 64x64/64x32/32x64 r4 candidates; keep all atomic semantics and boundary masks.
- **Benchmark:** allocation 108133, endtoend N=1280/1536/1792/2048, 5 randomized rounds x 200 ms, same-run pure-FP32 PEDANTIC cuBLAS. Full archive: `benchmark_results/strassen_primitive_20261017/iter5_swizzle`.
- **Status:** {'phase': 'benchmark', 'exit_code': 0}
- **N=1280 cublas_full:** 0.259921170 ms; same-run cuBLAS/latency=1.000000x.
- **N=1280 pipe64_k16_r4_seven_atomic:** 0.305269735 ms; same-run cuBLAS/latency=0.851448x.
- **N=1280 swizzle32x64_k16_r4_seven_atomic:** 0.304778938 ms; same-run cuBLAS/latency=0.852819x.
- **N=1280 swizzle64_k16_r4_seven_atomic:** 0.299318710 ms; same-run cuBLAS/latency=0.868376x.
- **N=1280 swizzle64x32_k16_r4_seven_atomic:** 0.302737965 ms; same-run cuBLAS/latency=0.858568x.
- **N=1536 cublas_full:** 0.435571112 ms; same-run cuBLAS/latency=1.000000x.
- **N=1536 pipe64_k16_r4_seven_atomic:** 0.512263108 ms; same-run cuBLAS/latency=0.850288x.
- **N=1536 swizzle32x64_k16_r4_seven_atomic:** 0.519409417 ms; same-run cuBLAS/latency=0.838589x.
- **N=1536 swizzle64_k16_r4_seven_atomic:** 0.499919336 ms; same-run cuBLAS/latency=0.871283x.
- **N=1536 swizzle64x32_k16_r4_seven_atomic:** 0.514966715 ms; same-run cuBLAS/latency=0.845824x.
- **N=1792 cublas_full:** 0.694382482 ms; same-run cuBLAS/latency=1.000000x.
- **N=1792 pipe64_k16_r4_seven_atomic:** 0.783158350 ms; same-run cuBLAS/latency=0.886644x.
- **N=1792 swizzle32x64_k16_r4_seven_atomic:** 0.819375649 ms; same-run cuBLAS/latency=0.847453x.
- **N=1792 swizzle64_k16_r4_seven_atomic:** 0.763863981 ms; same-run cuBLAS/latency=0.909039x.
- **N=1792 swizzle64x32_k16_r4_seven_atomic:** 0.804066179 ms; same-run cuBLAS/latency=0.863589x.
- **N=2048 cublas_full:** 0.993680924 ms; same-run cuBLAS/latency=1.000000x.
- **N=2048 pipe64_k16_r4_seven_atomic:** 1.148364456 ms; same-run cuBLAS/latency=0.865301x.
- **N=2048 swizzle32x64_k16_r4_seven_atomic:** 1.213144938 ms; same-run cuBLAS/latency=0.819095x.
- **N=2048 swizzle64_k16_r4_seven_atomic:** 1.124632263 ms; same-run cuBLAS/latency=0.883561x.
- **N=2048 swizzle64x32_k16_r4_seven_atomic:** 1.180678065 ms; same-run cuBLAS/latency=0.841619x.
- **Correctness:** 56 FP64 checks; all pass=True.
- **Acceptance:** N=1536 crossover remains subject to the full-call numbers above; isolated NCU durations are diagnostic and kernel replay serializes the seven launches. No claim of a hardware floor or completion.
