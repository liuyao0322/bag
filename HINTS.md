# Optimization Directives

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
