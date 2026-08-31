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
- The standalone suite passes 26 cases, including arbitrary positive strides,
  non-aligned 512-bucket boundaries, and concurrent two-stream atomic updates
- Timing: `triton.testing.do_bench`, 50 ms warmup and 200 ms sampling, with an L2 flush before each sample
- Reported latency: median GPU time; speedup is `unfused / fused`, so values below 1 mean the fused kernel is slower
- Optimized implementation: branch HEAD, combining the iteration-14
  large-shape winner with the iteration-16 square-512 bucket (final trajectory
  commit `4fcfafa`)

## Primary result: IEEE FP32, column-major

| Shape `(M,N,K)` | Fused (ms) | Unfused (ms) | Speedup | Fused effective TFLOP/s |
|---|---:|---:|---:|---:|
| `(256,256,256)` | 0.0205 | 0.0225 | 1.100x | 1.65 |
| `(512,512,512)` | 0.0512 | 0.0440 | 0.860x | 5.26 |
| `(1024,1024,1024)` | 0.1833 | 0.1761 | 0.961x | 11.74 |
| `(2048,2048,2048)` | 1.3189 | 1.0045 | 0.762x | 13.04 |
| `(4096,4096,4096)` | 8.6968 | 6.8270 | 0.785x | 15.81 |
| `(4096,1024,1024)` tall | 0.7363 | 0.6205 | 0.843x | 11.69 |
| `(1024,4096,1024)` wide | 0.7465 | 0.6205 | 0.831x | 11.53 |
| `(1024,1024,4096)` large-K | 0.6031 | 0.5540 | 0.919x | 14.26 |

## Aggregate comparison

| Precision | Layout | Initial speedup | Optimized speedup | Fused latency improvement | Best | Worst | Wins |
|---|---|---:|---:|---:|---:|---:|---:|
| IEEE FP32 | column-major | 0.437x | 0.877x | 2.022x | 1.100x | 0.762x | 1 / 8 |
| IEEE FP32 | row-major | 0.669x | 0.879x | 1.310x | 1.111x | 0.762x | 1 / 8 |
| TF32 | column-major | 0.706x | 0.922x | 1.296x | 1.250x | 0.757x | 2 / 8 |
| TF32 | row-major | 0.704x | 0.923x | 1.299x | 1.250x | 0.768x | 2 / 8 |

Across all 32 cases, the fused latency geometric mean falls from 0.514727 ms
to 0.354190 ms, a **1.453x kernel speedup (31.2% lower latency)**. The
fused-vs-unfused geometric mean improves from 0.617x to **0.900x**. On the
six-shape IEEE column-major optimization signal set (1024 through 4096,
tall/wide, and large-K), the final independent rerun is **2.269x faster than
the original Triton kernel**, reducing fused latency from 2.185636 ms to
0.963284 ms.

The largest improvement comes from column-major IEEE inputs: their fused
geometric-mean latency improves by 2.022x. The optimized wrapper recognizes
conventional column-major views and computes the equivalent transposed GEMM,
`C^T = B^T @ A^T`, by swapping operands, dimensions, coefficients, and
strides without materializing a transpose.

The GEMM body now uses shape-specific fused configurations. Problems with a
short edge of at least 1024, and large tall/wide problems with a short edge of
at least 512, use a `128x128x16` output/reduction tile with 16 warps and three
pipeline stages. This raises input reuse and exposes enough work inside the
single shared-memory-heavy CTA. Near-square fused problems between 257 and 512
use a `32x32x16` tile with four warps and three stages, producing 256 CTAs at
512 rather than the old bucket's 32. Other medium fused problems use
`128x64x32`, eight warps, and two stages. The full optimization trajectory
and rejected alternatives are recorded in `ITERATIONS.md`.

The square-512 follow-up removes the earlier performance cliff. In the formal
full-suite run, IEEE fused latency falls from 0.079872 to 0.051200 ms
(**1.560x faster**) for both layouts, while TF32 falls from 0.070656 to
0.040960 ms (**1.725x faster**). Relative-to-unfused performance improves from
0.551x to 0.860x for IEEE and from 0.493x to 0.850x for TF32. The independent
four-mode selection run measured a 1.641x fused-latency improvement over the
512 phase baseline.

The optimized fused path wins at `256^3` in both precisions/layouts and at
TF32 large-K. It remains slower overall than the strong preallocated unfused
sequence: 0.900x means about 11.1% higher geometric-mean latency. The
comparison is deliberately demanding because the unfused GEMM uses cuBLAS,
while the primitive must perform two atomic output accumulations to preserve
safe concurrent BAG updates. After fixing the 512 grid cliff, the remaining
worst cases are the large square GEMMs at about 0.76–0.79x, where the custom
IEEE/TF32 GEMM body still trails cuBLAS throughput and must also execute the
atomic epilogue.

For representative large cases, IEEE column-major `4096^3` reaches 15.81
effective TFLOP/s fused at 0.785x of the unfused path, and TF32 column-major
large-K reaches 19.58 effective TFLOP/s fused and wins by 1.042x.

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
