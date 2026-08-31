# Triton BAG Primitive: Fused vs. Unfused

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

## Primary result: IEEE FP32, column-major

| Shape `(M,N,K)` | Fused (ms) | Unfused (ms) | Speedup | Fused effective TFLOP/s |
|---|---:|---:|---:|---:|
| `(256,256,256)` | 0.0266 | 0.0236 | 0.885x | 1.27 |
| `(512,512,512)` | 0.0850 | 0.0440 | 0.518x | 3.17 |
| `(1024,1024,1024)` | 0.5140 | 0.1782 | 0.347x | 4.19 |
| `(2048,2048,2048)` | 2.0736 | 1.0117 | 0.488x | 8.29 |
| `(4096,4096,4096)` | 13.0949 | 6.6796 | 0.510x | 10.50 |
| `(4096,1024,1024)` tall | 1.9190 | 0.6241 | 0.325x | 4.48 |
| `(1024,4096,1024)` wide | 1.9866 | 0.6226 | 0.313x | 4.33 |
| `(1024,1024,4096)` large-K | 1.6977 | 0.5560 | 0.328x | 5.07 |

## Aggregate comparison

| Precision | Layout | Geometric-mean speedup | Best | Worst | Wins |
|---|---|---:|---:|---:|---:|
| IEEE FP32 | column-major | 0.437x | 0.885x | 0.313x | 0 / 8 |
| IEEE FP32 | row-major | 0.669x | 1.111x | 0.455x | 1 / 8 |
| TF32 | column-major | 0.706x | 1.250x | 0.550x | 1 / 8 |
| TF32 | row-major | 0.704x | 1.250x | 0.574x | 1 / 8 |

The only wins occur at `256^3`, where launch and intermediate-memory overhead
are a large fraction of total execution time. The `256^3` measurements also
have the largest percentile spread, so their exact speedup is less stable than
the results at larger sizes.

For `512^3` and larger problems, eliminating four launches and the intermediate
global-memory traffic does not compensate for the throughput gap between the
current Triton GEMM body and cuBLAS. For example, at IEEE row-major `4096^3`,
the fused kernel reaches 15.64 effective TFLOP/s while the unfused path reaches
20.63 effective TFLOP/s. At TF32 column-major `4096^3`, the corresponding
figures are 21.09 and 28.12 effective TFLOP/s.

Column-major IEEE inputs are especially unfavorable to the current Triton tile
layout: its geometric-mean speedup is 0.437x, compared with 0.669x for
row-major. TF32 reduces the gap, but the fused implementation still loses on
all tested shapes except `256^3`.

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
  --output benchmark_results/primitive_fused_vs_unfused_rtx3090_all.csv
```

The CSV contains p20, median, and p80 latency for every case.
