# Paper batching and hybrid two-level FP32 Strassen

The immutable arithmetic build is archived under
`benchmark_results/strassen_batching_20261017/iter4_hybrid_smoke`.
The report and full curves are in the parent directory's `REPORT.md`.
CUDA 12.2, sm_80, cuBLAS PEDANTIC, TF32 disabled; operation is column-major
FP32 `C <- AB+C`, alpha=beta=1. No ordinary GEMM fallback in Strassen cases.

## Unmet crossover target

The paper reports one-level crossover N=1536 on V100 SXM2 / CUDA 9.1.
The fixed Huge seven-stream A100 curve first clearly wins at the measured
dense point N=4352 across both seeds; this does not establish the best
crossover across all blocking strategies. The small-size target is unmet.
Hybrid first clearly wins at measured N=6656, compared with the paper's
reported 7680. Neither comparison is a same-hardware reproduction.
Large-size hybrid speedup near 1.19x does not satisfy the one-level target.

The bounded existing-configuration audit uses the unchanged immutable build:

```bash
salloc --partition=a100 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
  --time=2-00:00:00 --job-name=strassen-crossover \
  bash scripts/run_strassen_crossover_a100.sh \
  benchmark_results/strassen_batching_20261017/iter4_hybrid_smoke \
  benchmark_results/my_crossover_audit
/share/home/liuyao/miniconda3/envs/kernelagent/bin/python -s \
  scripts/analyze_strassen_crossover.py benchmark_results/my_crossover_audit
```

This compares all 11 existing backends with seven streams and the two
supported four-batched backends at N=1024/1280/1536/1792/2048/2560/3072,
five rounds of 200 ms at seed offsets 0 and 1009. Select each size's candidate
on seed 0 and report the same configuration on seed 1009. Keep the fixed main
curve and the configuration audit separately identifiable.

## Implementations

- `Schedule::FourBatchedAtomic`: four concurrent streams, three grid.z=2
  batches and M7. Same coefficients share a specialization; M3's sole output
  is remapped to share M4's specialization. Original ordered pairs remain a
  separate control. Supported backends: Huge and Pipe64x32.
- `hybrid_two_level`: workspace-based outer Strassen and fused one-level inner
  products, following the structure of paper section 3.4. Three compact
  half-size scratch matrices are reused across seven outer products. Each
  preparation kernel transforms A/B and clears the new intermediate output;
  the caller waits for each inner call before merging it into original C.
  Both seven-stream and four-batched inner schedules are supported.
- Extra scratch is `3*N*N` bytes (1.171875 GiB at N=20480). A/B/C take an
  additional `12*N*N` bytes. Allocation is outside the steady-state interval.
  Every required transform, scratch reset, product, merge and join is inside.

## API

```cpp
#include "hybrid.cuh"
strassen_cuda::Workspace workers;
strassen_cuda::HybridWorkspace scratch(n);
strassen_cuda::hybrid_two_level(
    strassen_cuda::Backend::Huge,
    strassen_cuda::Schedule::SevenAtomic,
    a, b, c, n, caller_stream, workers, scratch);
```

One level requires N divisible by 8; hybrid requires N divisible by 16 and a
valid quarter-size leaf. Huge requires its leaf dimension divisible by 128,
so Huge one level requires N divisible by 256 and hybrid N divisible by 512.
Generic Pipe64x32 supports masked partial tiles, with float4 alignment still
required. A/B/C are separate allocations with leading dimension N. Scratch
uses compact leading dimension N/2. Workspaces are not reentrant; use separate
workers and scratch for independent concurrent callers. Synchronize the caller
before reading/freeing outputs or destroying workspaces.

## Reproduce

Use salloc first and run inside the allocated GPU environment. GPU builds run
on compute nodes. Existing runner/replay contracts are in STRASSEN_ONELEVEL.md.

```bash
srun --jobid=<allocation> bash scripts/bench_strassen_onelevel.sh iter-4 \
  benchmark_results/my_batching/build --mode endtoend \
  --sizes 128,256,512 --backends huge,pipe64x32_k16_r4 \
  --schedules seven_atomic,four_batched_atomic --hybrid --validate-only

srun --jobid=<allocation> bash scripts/replay_strassen_onelevel.sh \
  benchmark_results/my_batching/build benchmark_results/my_batching/curve \
  --mode endtoend \
  --sizes 512,1024,1536,2048,3072,4096,6144,8192,10240,12288,16384,20480 \
  --backends huge --schedules seven_atomic,four_batched_atomic \
  --hybrid --rounds 5 --target-ms 200
```

Repeat with a new output directory and `--seed-offset 1009` for independent
inputs. Device initialization and FP64 reference both use seeds A=11+offset,
B=23+offset, C=37+offset. The original cost benchmark's defaults are unchanged.

Numerical checks cover one and three accumulated calls. Exhaustive output
checks apply at N<=128; other cases request 128 positions including corners.
Relative L2 <=3e-5 and scaled maximum error <=1e-4. Five randomized timing
rounds, 200 ms target, minimum three calls, same-run cuBLAS denominator.

`--schedules` filters schedule names, and `--hybrid` adds legal two-level cases.
`--only` selects an exact case. `--profile-calls K` replaces ordinary timing
with K calls between cudaProfilerStart/Stop after validation/warmup; select
exactly one case. Profiler intervals never enter the primary timing table.

```bash
srun --jobid=<allocation> bash scripts/verify_strassen_batching.sh \
  benchmark_results/my_batching/build benchmark_results/my_batching/verification
/share/home/liuyao/miniconda3/envs/kernelagent/bin/python -s \
  scripts/analyze_strassen_batching.py benchmark_results/my_batching
```

Verification includes masked batching and two-level memcheck, shared-memory
racecheck, complete-call Nsight Systems traces, dynamic cuBLAS Tensor counters
and static custom SASS opcode audit. Global output race safety additionally
relies on atomic updates for concurrent workers and caller-stream ordering
between outer products; shared-memory racecheck alone does not prove this.

## A800 queue

`scripts/run_strassen_batching_a800.sh <immutable-build> <suite-output>` is an
allocation payload that uses srun to run the same twelve sizes at seed offsets
0 and 1009, and exits to release its allocation. It does not download anything.
The current queued request is held by tmux session `strassen-a800-20261017`;
job ID and state are recorded in the suite's `a800_queue.txt` and allocation log.
Do not remove the immutable executable/archive while that request is pending.

Refresh the report after A800 finishes using the analysis command above.
An A800 result is accepted only after both replay status files have exit 0;
queued or partially generated data is not a performance result.
