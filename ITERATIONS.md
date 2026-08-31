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
