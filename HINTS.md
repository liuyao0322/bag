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
