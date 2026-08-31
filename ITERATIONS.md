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
- Analysis: even idealized leaves do not reach 8/7 at this finite size. Seven
  half-size cuBLAS calls scale to about 40.5 ms rather than exactly 7/8 of the
  full GEMM, and unavoidable input materialization/direct decoding adds about
  2.8 ms. The hybrid path establishes a measured practical ceiling near
  1.02x for IEEE N=8192 on this GPU.
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
