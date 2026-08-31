# Optimization Directives

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
