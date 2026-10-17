# GPU matrix multiplication: Strassen and BAG

This repository contains CUDA Strassen and Triton BAG implementations, with
complete-call benchmarks and archived numerical/performance evidence.

## Strassen on CUDA

- [Algorithm, API and reproduction](solution/strassen_cuda/README.md)
- [One-level seven-product implementation and scheduling](solution/strassen_cuda/engine.cuh)
- [Fused FP32 primitive](solution/strassen_cuda/simt.cuh)
- [Hybrid two-level implementation](solution/strassen_cuda/hybrid.cuh)
- [A100 small-size performance and validation](benchmark_results/strassen_primitive_20261017/REPORT.md)
- [Large-size and hybrid results](benchmark_results/strassen_batching_20261017/REPORT.md)

The operation is column-major FP32 `C <- A*B+C`. The validated small-size
configuration uses `ColumnTile8` with `FourBatchedAtomic`. Two independent
seven-round A100 runs at N=1536 measured 1.0199x and 1.0156x over same-run
cuBLAS. Size-specific results and large-size regressions are retained in the
reports; the small-size configuration is not a universal choice.

## Alternative-basis multiplication on CUDA

- [CUDA BAG algorithm, recursive API and reproduction](solution/bag_cuda/README.md)
- [Implementation using the tuned primitive](solution/bag_cuda/bag.cuh)

CUDA BAG implements the seven products and basis transformations described
in BAG.md. It supports recursive levels, minimal/full basis storage and
serial, seven-stream or batched schedules, preserving FP32 `C <- A*B+C`.

## BAG on Triton

- [BAG algorithm and project notes](BAG.md)
- [BAG implementation](solution/bag_triton/bag.py)
- [Triton fused primitive](solution/bag_triton/primitive.py)

Builds and measurements require a CUDA GPU. On the cluster, run them on an
allocated compute node; see the Strassen reproduction instructions above.
