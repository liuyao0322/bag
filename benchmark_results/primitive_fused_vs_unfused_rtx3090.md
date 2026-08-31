# Optimized Triton BAG Primitive: Fused vs. Unfused

## Benchmark setup

- Date: 2026-08-31
- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.10.0+cu128
- Triton: 3.6.0
- Operation: `P = (X - Y) @ (V - W); D += P; E += P`
- Fused path: one Triton BAG primitive kernel with atomic updates to `D` and `E`
- Unfused path: two `torch.sub`, one `torch.mm`, and two `Tensor.add_` GPU operators
- All unfused intermediate tensors are preallocated; allocation time is excluded
- Each shape passes a fused-vs-unfused correctness check before timing
- Timing: `triton.testing.do_bench`, 50 ms warmup and 200 ms sampling, with an L2 flush before each sample
- Reported latency: median GPU time; speedup is `unfused / fused`, so values below 1 mean the fused kernel is slower
- Optimized implementation: commit `f13b00b` trajectory, published from the
  iteration-14 best kernel

## Primary result: IEEE FP32, column-major

| Shape `(M,N,K)` | Fused (ms) | Unfused (ms) | Speedup | Fused effective TFLOP/s |
|---|---:|---:|---:|---:|
| `(256,256,256)` | 0.0205 | 0.0225 | 1.100x | 1.65 |
| `(512,512,512)` | 0.0799 | 0.0440 | 0.551x | 3.37 |
| `(1024,1024,1024)` | 0.1864 | 0.1761 | 0.945x | 11.55 |
| `(2048,2048,2048)` | 1.3517 | 1.0221 | 0.756x | 12.72 |
| `(4096,4096,4096)` | 8.6252 | 6.6652 | 0.773x | 15.94 |
| `(4096,1024,1024)` tall | 0.7521 | 0.6380 | 0.848x | 11.44 |
| `(1024,4096,1024)` wide | 0.7626 | 0.6298 | 0.826x | 11.28 |
| `(1024,1024,4096)` large-K | 0.6236 | 0.5612 | 0.900x | 13.79 |

## Aggregate comparison

| Precision | Layout | Initial speedup | Optimized speedup | Fused latency improvement | Best | Worst | Wins |
|---|---|---:|---:|---:|---:|---:|---:|
| IEEE FP32 | column-major | 0.437x | 0.823x | 1.887x | 1.100x | 0.551x | 1 / 8 |
| IEEE FP32 | row-major | 0.669x | 0.828x | 1.222x | 1.111x | 0.551x | 1 / 8 |
| TF32 | column-major | 0.706x | 0.858x | 1.196x | 1.250x | 0.493x | 2 / 8 |
| TF32 | row-major | 0.704x | 0.858x | 1.200x | 1.250x | 0.493x | 2 / 8 |

Across all 32 cases, the fused latency geometric mean falls from 0.514727 ms
to 0.381549 ms, a **1.349x kernel speedup (25.9% lower latency)**. The
fused-vs-unfused geometric mean improves from 0.617x to **0.842x**. On the
six-shape IEEE column-major optimization signal set (1024 through 4096,
tall/wide, and large-K), the final independent rerun is **2.269x faster than
the original Triton kernel**, reducing fused latency from 2.185636 ms to
0.963284 ms.

The largest improvement comes from column-major IEEE inputs: their fused
geometric-mean latency improves by 1.887x. The optimized wrapper recognizes
conventional column-major views and computes the equivalent transposed GEMM,
`C^T = B^T @ A^T`, by swapping operands, dimensions, coefficients, and
strides without materializing a transpose.

The GEMM body now uses shape-specific fused configurations. Problems with a
short edge of at least 1024, and large tall/wide problems with a short edge of
at least 512, use a `128x128x16` output/reduction tile with 16 warps and three
pipeline stages. This raises input reuse and exposes enough work inside the
single shared-memory-heavy CTA. Medium fused problems use `128x64x32`, eight
warps, and two stages. The full optimization trajectory and rejected
alternatives are recorded in `ITERATIONS.md`.

The optimized fused path wins at `256^3` in both precisions/layouts and at
TF32 large-K. It remains slower overall than the strong preallocated unfused
sequence: 0.842x means about 18.8% higher geometric-mean latency. The
comparison is deliberately demanding because the unfused GEMM uses cuBLAS,
while the primitive must perform two atomic output accumulations to preserve
safe concurrent BAG updates. The `512^3` case is the largest remaining
performance valley; at that size cuBLAS is already efficient while launch,
atomic, and limited-grid costs remain a large fraction of the fused kernel.

For representative large cases, IEEE column-major `4096^3` reaches 15.94
effective TFLOP/s fused versus 20.63 TFLOP/s unfused (0.773x), and TF32
column-major large-K reaches 19.18 effective TFLOP/s fused versus 18.59
TFLOP/s unfused (1.032x).

An Nsight Compute profile was attempted once with the SpeedOfLight and
MemoryWorkloadAnalysis sections, but the cluster returned
`ERR_NVGPUCTRPERM`: this user does not have permission to access GPU hardware
performance counters on the RTX 3090 node. No NCU metrics are therefore used
in the conclusions above.

## Reproduction

```bash
PYTHONPATH=src python scripts/benchmark_triton_primitive.py \
  --layout column-major row-major \
  --precision ieee tf32 \
  --warmup-ms 50 \
  --rep-ms 200 \
  --output benchmark_results/primitive_fused_vs_unfused_rtx3090_optimized.csv
```

The optimized CSV contains p20, median, and p80 latency for every case. The
original `primitive_fused_vs_unfused_rtx3090_all.csv` is retained so the
initial-to-optimized comparison can be reproduced.
