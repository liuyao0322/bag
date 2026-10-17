# Optimization Directives

Source: user request, 2026-10-17 (CUDA alternative-basis multiplication).

- Implement the alternative-basis matrix multiplication in BAG.md using the
  tuned CUDA primitive. Preserve the documented seven-product algebra and
  basis/inverse transformations at each recursive level.
- Reuse the validated FP32 primitive body; verify the complete operation,
  including nonzero initial C, repeated accumulation and caller-stream order.
- Keep all transforms, scratch resets, products, inverse decoding and stream
  dependencies inside complete-call timing; compare same-run pure FP32 cuBLAS.

Source: active user goal, 2026-10-17 (primitive profiling and implementation).

- Profile the four fused primitive variants at leaf=768/1024: registers,
  CTA concurrency, memory access and atomic output updates. Change the kernel
  according to measured bottlenecks, retaining the N=1536 cuBLAS target.
- Judge changes by complete seven-product calls at N=1280/1536/1792/2048;
  isolated primitive improvement does not satisfy the performance goal.

Source: user correction, 2026-10-17 (paper crossover not reached).

- Treat the paper's one-level crossover at N=1536 as an unmet performance
  target. A100 large-size hybrid speedup near 1.19x does not establish that
  the small-size crossover has been reproduced.
- Report one-level and hybrid crossover separately, including hardware,
  candidate-selection policy, measured points, and non-monotonic behavior.
- Prioritize the one-level N=1536/2048 region; compare existing blocking
  strategies before attributing the gap to hardware or launch overhead.
- Keep complete-call timing, pure FP32, numerical limits, and same-run cuBLAS.

Source: user approval, 2026-10-17 (next-phase plan, "同意").

- Implement the paper's four-variant, four-stream atomic batching schedule;
  compare complete calls with existing seven-stream and ordered-pair schedules.
- Extend the A100 one-level size curve, measure complete-call timelines, and
  freeze candidates before independent final replay. Keep all slow cases.
- Then implement workspace-based outer Strassen with fused one-level inner
  products (hybrid two-level); include transformations, scratch initialization,
  all products, output merge and required synchronization in timed calls.
- Retain pure FP32 C <- AB+C and existing tolerances. Add independent input
  seeds and repeated-call checks. Validate fixed candidates on A800.
- Treat a reproducible 3% improvement over the stable one-level implementation
  as a decision threshold for further hybrid tuning, not a promised speedup.

Source: user goal, 2026-10-17 (primitive_large analysis and one-level Strassen).

- Analyze loop, blocking and output-update policies at N=2048 (leaf=1024).
- Preserve large-size regressions and integrate a complete seven-product
  one-level Strassen path to measure actual end-to-end gains.
- Keep pure FP32, alpha=beta=1, unchanged numerical tolerances, and all
  required stream dependencies inside the timed operation. Non-atomic
  updates are allowed only with an ordering that prevents output races.

Source: user goal, 2026-10-17 (Strassen performance baseline).

- Build a standalone CUDA baseline on A100/A800 with pure FP32 and beta=1
  (`C <- A*B + C`). Measure full-size cuBLAS, half-size leaf GEMM, and the
  existing fused primitive costs before changing the kernel implementation.
- Keep compact leaf storage and parent-leading-dimension leaf storage
  distinguishable. Record raw rounds, numerical checks, environment, and
  the exact primitive source snapshot for every measurement.

Source: user prompt, 2026-08-31.

- Optimize the Triton BAG primitive's GEMM body, not the benchmark.
- Preserve `P = (X + delta*Y) @ (V + epsilon*W)` and both output updates.
- Preserve all coefficient modes, arbitrary positive strides, boundary masks,
  IEEE/TF32 selection, and concurrent atomic accumulation correctness.
- Rank steady-state fused latency on the RTX 3090 signal set; run the full
  fused-vs-unfused suite before selecting the final implementation.
- Do not include JIT compilation or allocation time in steady-state results.
- Do not weaken correctness tolerances or remove hard shapes to claim speedup.

Source: user prompt, 2026-08-31 (512-bucket follow-up).

- Optimize the square-512 launch bucket specifically, across row-major and
  column-major layouts and both IEEE and TF32 precision modes.
- Retain the previously selected large-shape configuration and require a full
  correctness/performance regression before publishing the 512-bucket winner.

Source: user prompt, 2026-08-31 (end-to-end BAG implementation).

- Implement the depth-1 BAG matrix-multiplication algorithm with the Triton
  primitive, including the forward combinations, seven products, and inverse
  basis transform.
- Compare end-to-end BAG latency against ordinary `torch.mm`/cuBLAS GEMM with
  preallocated outputs and workspace; exclude allocation, JIT compilation,
  random-data generation, and library initialization from timed regions.
- Preserve exact BAG algebra and report IEEE and TF32 numerical error honestly;
  do not weaken correctness thresholds or omit slow buckets to claim speedup.

Source: user prompt, 2026-08-31 (BAG launch scheduling follow-up).

- Use multi-stream and dependency-batched scheduling for the seven BAG
  primitive products where it improves latency.
- Preserve caller-stream semantics, synchronize batches with CUDA events, and
  keep overlapping output updates race-free; do not count stream creation or
  graph construction in steady-state timings.

Source: user prompt, 2026-08-31 (continue toward the 8/7 bound).

- Continue optimizing the end-to-end BAG implementation because the measured
  speedup has not reached the arithmetic-only theoretical bound of 8/7.
- Evaluate a genuine seven-stream schedule with all seven primitive products
  eligible to run concurrently; pre-create streams and all workspace outside
  the timed region.
- Keep transform, output initialization, atomic accumulation, merge/inverse
  transform, and required synchronization inside end-to-end timing.
- If full concurrency is limited by GEMM resource contention, evaluate hybrid
  and grouped/batched leaf-GEMM execution without changing BAG algebra or
  hiding unfavorable buckets.

Source: user correction, 2026-08-31 (prior measured upper bound).

- Reconcile new finite-shape upper-bound claims with the previously observed
  approximately 1.14x upper bound before drawing conclusions about what BAG
  can reach.
- Report the exact shape, precision, numerator/denominator, and timing boundary
  for both measurements; do not generalize a result from N<=8192 to the
  asymptotic or previously measured regime.

Source: user prompt, 2026-08-31 (continue optimization after upper-bound audit).

- Continue active optimization toward the 8/7 target; changes may improve
  either the Triton primitive GEMM body or the end-to-end BAG algorithm and
  scheduling implementation.
- Prioritize genuine end-to-end latency reduction, including candidates that
  reduce intermediate product traffic, while retaining the existing full
  correctness and measurement contract.

Source: user prompt, 2026-08-31 (small-matrix BAG follow-up).

- Prioritize end-to-end BAG latency for N=512 and N=1024 across row-major and
  column-major layouts and both IEEE and TF32 precision modes; use N=2048 as a
  boundary-regression signal before the final all-shape replay.
- Preserve a genuine seven-product BAG execution path. Do not substitute an
  ordinary GEMM fallback, weaken correctness, or omit unfavorable cases to
  improve the reported small-matrix ratios.
- Target fixed launch cost, intermediate-product traffic, and small-batch CTA
  utilization while retaining the established workspace and timing contract.
