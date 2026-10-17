# Alternative-basis multiplication on the tuned CUDA primitive

`bag.cuh` implements the BAG algebra in [BAG.md](../../BAG.md), including
forward basis transformation, seven products and inverse transformation at
every recursive level. The operation is square, column-major FP32
`C <- A*B+C`, with separate A/B/C allocations and leading dimension N.

The bottom level calls the existing tuned CUDA primitive. The default backend
is `strassen_cuda::Backend::ColumnTile8`: 64x64 CTA tiles, K steps of16,
4x8 register output tiles, shared input double buffering and coalesced atomic
outputs. The primitive body is reused without modification. No cuBLAS GEMM
replaces a BAG product; cuBLAS is the benchmark reference.

## API and recursion

```cpp
#include "bag.cuh"

// Create once, outside repeated or timed calls.
bag_cuda::Workspace workspace(n, /*depth=*/1, bag_cuda::Basis::Minimal);
bag_cuda::matmul(a, b, c, n, caller_stream, workspace,
                bag_cuda::Backend::ColumnTile8, bag_cuda::Schedule::Batched);
// Synchronize caller_stream before consuming/freeing C or destroying workspace.
```

The caller stream orders inputs, forward transformations, all worker-stream
products, inverse transformation and output accumulation. A workspace is not
reentrant; independent concurrent callers need distinct workspaces. A and B
remain unchanged. Nonzero initial C is accumulated only after decoding the
new product, so it is never interpreted in the transformed basis.

Depth is positive. At depth1 there are seven primitive products; depth2
recursively computes seven BAG subproblems, giving49 primitive products.
Every level performs its own forward/inverse transformation as described in
the algorithm. Outer products reuse compact child A/B/product buffers in
caller-stream order. The child's inverse transformation overwrites its
temporary product before the parent merges it, preventing stale accumulation
across repeated calls.

Each BAG level requires its parent dimension divisible by8, so root N must
be divisible by `8*2^(depth-1)`. The selected primitive masks partial tiles.
Other backends also impose leaf tile constraints; `Huge`, for example,
requires `N/2^depth` divisible by128. Invalid workspace sizes, depths and
backend leaf shapes are rejected.

## Basis layouts

`Basis::Minimal` materializes only:

```text
SA = A12-A21+A22
SB = B12-B21+B22
```

Both occupy adjacent row blocks of one `N x (N/2)` buffer, preserving the
primitive's common leading dimension N. The seven input pairs are:

| Product | Left | Right | Transformed-output update |
|---|---|---|---|
| M1 | A12+A22 | B12+B22 | T22 += M1 |
| M2 | A22-A21 | B22-B21 | T21 += M2 |
| M3 | SA | SB | T12 += M3 |
| M4 | A11 | B11 | T11 += M4 |
| M5 | A12 | B21 | T11 += M5; T12 += M5 |
| M6 | SA-A11 | B12 | T12 -= M6; T22 -= M6 |
| M7 | A21 | SB-B11 | T12 += M7; T21 += M7 |

All remaining combinations are fused into primitive input loads/packing.
M5 simplifies algebraically from `(Apsi12-Apsi21)(Bphi22-Bphi12)` to A12*B21.
The final decoder accumulates T11, T12-T21, -T12+T22 and -T12+T21+T22 into
C11, C12, C21 and C22, respectively.

`Basis::Full` stores complete transformed A/B and executes the original
seven formulas directly. Both modes zero an independent transformed output T.
One-level scratch uses `6*N*N` bytes in minimal mode and `12*N*N` bytes in
full mode. Deeper levels additionally reuse three child matrices and one
recursive child workspace; `workspace.bytes()` reports total allocation.

## Scheduling and measurement

- `Schedule::Serial`: all seven products on the caller stream.
- `Schedule::SevenStream`: all seven products eligible to run concurrently.
- `Schedule::Batched`: full mode batches M1/M2, M3/M4 and M5/M7 plus M6,
  using four worker streams; minimal mode batches M3/M4 plus five single
  products, using six worker streams. Concurrent outputs use atomic adds.

Transform, T initialization, all products, recursive child preparation/merges,
inverse decoding and stream dependencies are inside complete-call timing.
Workspaces, streams, library initialization and benchmark input generation
are outside. Primary comparisons use same-run pure FP32 cuBLAS, alpha=beta=1,
PEDANTIC math and `NVIDIA_TF32_OVERRIDE=0`.

On the cluster, build on an allocated compute node:

```bash
srun --jobid=<allocation> bash scripts/bench_bag_cuda.sh iter-N \
  benchmark_results/my_bag/build --sizes 1280,1536,1792,2048 \
  --depths 1 --rounds 5 --target-ms 200

srun --jobid=<allocation> bash scripts/replay_bag_cuda.sh \
  benchmark_results/my_bag/build benchmark_results/my_bag/recursive_checks \
  --sizes 128,256,512 --depths 1,2,3 --seed-offset 1009 --validate-only
```

Both scripts archive source hashes, exact commands, hardware and raw results.
Numerical checks compare against independent FP64 matrix products, including
nonzero initial C and one/three accumulated calls: exhaustive for N<=128 and
deterministic samples including corners otherwise. Limits remain relative
L2 <=3e-5 and scaled maximum error <=1e-4.

## Verified results

The [A100 report](../../benchmark_results/bag_cuda_20261017/REPORT.md) and
[machine-readable verification](../../benchmark_results/bag_cuda_20261017/verification.json)
retain complete-call timings, all layouts/schedules and large-size regressions.
Two independent input sets passed 608 ordinary FP64 records, depth1/2/3 and
memcheck/racecheck. The frozen small-size choice is minimal basis with batched
scheduling. N=1536 measures 0.450--0.453 ms against same-run cuBLAS 0.432--0.436 ms,
so this BAG implementation has not matched the earlier Strassen crossover.
N=1792 is about 1.6% faster than same-run cuBLAS. Timing includes all BAG
transforms and output decoding; neither result generalizes to all sizes.
