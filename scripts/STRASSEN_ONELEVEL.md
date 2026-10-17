# A100 FP32 one-level Strassen experiment

This experiment isolates loop structure, blocking and beta=1 output updates at
parent N=2048 (leaf 1024), retains N=8192/20480 as regressions, and measures an
actual seven-product Strassen operation. Results and figures are under
`benchmark_results/strassen_onelevel_20261017/REPORT.md`.

The subsequent paper four-variant batching, hybrid two-level implementation,
independent-seed curves and A800 queue workflow are documented in
[STRASSEN_BATCHING.md](STRASSEN_BATCHING.md).

## Build and measure

Use an existing A100 allocation, CUDA 12.2 and a GPU environment established by
`srun`. GPU builds and measurements run on the allocated compute node. The
runner checks the allocation, captures all sources and hashes, and records
build commands, raw rounds, FP64 checks, environment and GPU telemetry.

```bash
srun --jobid=<allocation> bash scripts/bench_strassen_onelevel.sh iter-1 \
  benchmark_results/my_strassen/ablation \
  --mode ablation --sizes 2048,8192,20480 --rounds 3 --target-ms 100

srun --jobid=<allocation> bash scripts/replay_strassen_onelevel.sh \
  benchmark_results/my_strassen/ablation benchmark_results/my_strassen/endtoend \
  --mode endtoend --sizes 2048,8192,20480 \
  --backends legacy,huge,pipe64x32_k16_r4 --rounds 5 --target-ms 200

srun --jobid=<allocation> bash scripts/verify_strassen_onelevel.sh \
  benchmark_results/my_strassen/ablation benchmark_results/my_strassen/verification
```

The build runner snapshots local primitive headers when present. A clean
checkout uses the tracked original baseline header archive instead. A replay
verifies the original source manifest and records the executable hash. If the
local executable has been removed, rebuild from that archive's recorded build
command; the archived benchmark and header sources are sufficient.

## Operation and timing

- Square, column-major FP32 matrices, separate A/B/C allocations, `C <- AB+C`.
  Alpha=beta=1, `CUBLAS_PEDANTIC_MATH`, cuBLAS atomics disabled, and
  `NVIDIA_TF32_OVERRIDE=0`. Custom kernels use FP32 FMA instructions.
- Inputs have fixed deterministic seeds A=11, B=23, C=37. Nonzero C is reset
  before each independent trial, outside timing, for both cuBLAS and Strassen.
  The algorithm never resets C. It fuses transforms and output updates into
  the seven primitive GEMMs without global intermediate matrices.
- CUDA events enclose repeated complete calls on the caller stream. Every
  worker stream waits for the caller's input readiness and rejoins it before
  the stop event. Stream/event creation is outside the interval. All required
  execution dependencies, arithmetic and output updates are inside it.
- Warmup targets 20 ms. Each randomized round probes three calls and targets
  the requested batch duration, with at least three timed calls. Final runs
  use five rounds and a 200 ms target. A100 clocks are not manually locked;
  compare cuBLAS from the same run and inspect raw ranges/CV.
- FP64 checks use exhaustive output verification for sizes <=128 and 128
  requested positions including corners (deduplicated) otherwise. Validate one and three
  consecutive accumulated calls. Unchanged limits: relative L2 <=3e-5 and
  maximum absolute error/(1+maximum reference magnitude) <=1e-4. Primitive
  checks also verify inactive C quadrants remain unchanged.

## Calling the implementation

Include `solution/strassen_cuda/engine.cuh` and pass both its directory and
the original primitive header archive to nvcc's include search path. Link CUDA
runtime; cuBLAS is only needed by the benchmark.

```cpp
strassen_cuda::Workspace workspace; // construct before repeated calls
strassen_cuda::one_level(
    strassen_cuda::Backend::Pipe64x32,
    strassen_cuda::Schedule::SevenAtomic,
    a, b, c, n, caller_stream, workspace);
// Work is ordered on caller_stream; synchronize it before reading/freeing C.
```

N must be positive and divisible by 8 for float4 alignment. Original unmasked
backends additionally require exact tiles; `valid_shape(backend,n/2)` expresses
their restrictions, and `one_level` rejects incompatible sizes. The generic
SIMT backends mask partial tiles. The high-level API uses leading dimension N.
Use a separate workspace for independent concurrent callers; a workspace owns
reusable streams/events and should remain alive until its calls finish.

The schedules are serial atomic, serial ordinary addition, seven-stream
atomic, two-stream addition with output dependencies, and paired products.
Ordinary addition requires a backend with that epilogue. Paired execution is
implemented for `Huge` (atomic only) and `Pipe64x32` (atomic or ordinary).

The four-launch paired schedule computes `(M1,M2)`, `(M3,M4)`, `(M5,M6)`, then
M7. Each pair has disjoint active outputs. The second pair remaps M3's active
output to its first output argument; it preserves its positive coefficient.
Serial ordering between pairs makes ordinary updates race-free. All schedules
perform the same seven products; there is no ordinary-GEMM fallback.

The benchmark measures cuBLAS full/leaf costs and isolated GEMM/M7 primitive
controls in `ablation` mode. `endtoend` mode validates the complete dense
result and measures all seven products and their joins. The sum of isolated
leaf times is a diagnostic proxy, not an end-to-end latency prediction.
