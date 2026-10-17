# Standalone Strassen FP32 cost baseline

This benchmark measures the costs needed before attempting the GPU Strassen
performance reproduction: full-size cuBLAS SGEMM, compact and parent-stride
half-size cuBLAS leaves, and the existing CUDA primitive implementations.
It preserves the existing kernels and does not use the incomplete `kernels.cuh`
umbrella header or depend on PyTorch, Triton, or CUTLASS.

The main operation is `C <- A*B + C`, with FP32 inputs, accumulators and outputs.
cuBLAS uses `CUBLAS_PEDANTIC_MATH` and `CUBLAS_ATOMICS_NOT_ALLOWED`; the wrapper
also sets `NVIDIA_TF32_OVERRIDE=0`. All matrices are column-major. The binary is
compiled for Ampere `sm_80`; it is intended for the cluster A100/A800 partitions.

## Run

First obtain an allocation on the login node. The current a100/a800 partitions
allow two days; check live partition configuration before a new allocation.
Keep the allocation shell alive while running the benchmark, for example in tmux.

```bash
scontrol show partition a100
salloc --partition=a100 --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=4 --mem=32G --time=2-00:00:00 \
    --job-name=strassen-fp32-baseline bash
```

In another login shell, replace `<jobid>` with the granted allocation:

```bash
cd /share/home/liuyao/workspace/bag
srun --jobid=<jobid> --ntasks=1 bash -lc \
    'cd /share/home/liuyao/workspace/bag && bash scripts/run_strassen_fp32.sh benchmark_results/strassen_fp32_a100_<timestamp> --rounds 5 --target-ms 200'
```

The wrapper builds on the assigned compute node after loading CUDA 12.2. It
snapshots all included primitive headers and the benchmark source before building.
Use a new output directory for each run. Every run has a 45-minute hard timeout.

An initial numerical check covering every square-kernel bucket:

```bash
# Run with srun --jobid=<jobid> on the assigned node, as above.
bash scripts/run_strassen_fp32.sh benchmark_results/strassen_fp32_smoke_<timestamp> \
    --sizes 256,512,1024,2048,4096,10240 --validate-only
```

Options: `--sizes` is comma-separated; `--rounds`, `--target-ms`, and `--samples`
control measurement; `--only <case>` selects one case for instrumentation;
`--validate-only` runs numerical checks without performance sampling.
Unsupported dimensions are rejected: existing primitive kernels have unmasked
tiles. This harness does not add fringe support or claim it was verified.

After successful completion:

```bash
python scripts/summarize_strassen_fp32.py benchmark_results/strassen_fp32_a100_<timestamp>
# Optional scientific figures, using an environment that already has matplotlib:
/share/home/liuyao/miniconda3/envs/kernelagent/bin/python -s \
    scripts/summarize_strassen_fp32.py benchmark_results/strassen_fp32_a100_<timestamp> --plot
```

The summarizer rejects missing or duplicate rounds, missing cases, invalid
latencies, and incomplete or failed accumulation checks. It produces a report,
per-case distribution CSV, and a cost comparison CSV. Raw rounds remain intact.

## Cases and interpretation

- `cublas_full`: N x N x N, ld=N.
- `cublas_leaf_compact`: (N/2)^3, ld=N/2.
- `cublas_leaf_parent_ld`: (N/2)^3, ld=N, operating on parent quadrants.
- `primitive_gemm`: existing primitive with delta=epsilon=0 and one atomic output.
- `primitive_m1` through `primitive_m7`: the seven coefficient patterns in
  `1_level_strassen.cuh`, measured in isolation. X/Y and V/W are representative
  top-left/bottom-right input quadrants, D/E are disjoint output quadrants.
  These cases measure costs rather than execute the complete Strassen formula.

The leaf and primitive sums are serial cost proxies. They are not measured
Strassen speedups or physical upper bounds. The leaf proxy excludes transforms
and multi-output accumulation; both omit actual operand/cache interactions and
concurrent scheduling. `primitive_gemm` still includes atomic output operations,
so it is not a pure FMA-body measurement.

## Timing and accuracy

Every case is warmed for approximately 20 ms. Each round randomizes case order
and measures an adaptive batch of 3 to 10,000 calls, targeting 200 ms by
default (the cap shortens the fastest cases). CUDA events enclose all work on one non-default stream. Allocation,
input/output-seed initialization, correctness checking, and library setup are
outside timing. Inside a batch, repeated calls accumulate into C with beta=1;
the output seed is restored before each batch. No CUDA Graphs are used. Reported
per-call latency is batch time divided by repetitions, so it includes any GPU
idle time caused by C++ host submission.

Correctness uses reproducible exactly representable uniform FP32 inputs and
nonzero C seeds, compared to CPU FP64 dot products after one and three calls.
Outputs of side <=128 are checked exhaustively; larger outputs use deterministic
samples including corners, plus untouched-quadrant sentinels. Inactive-output
and sentinel samples must match the original seed exactly. Acceptance thresholds
are relative L2 <=3e-5 and max-absolute-error/(1+max-absolute-reference) <=1e-4.
Large-matrix correctness is sampled. GPU clocks are not locked; telemetry is
recorded to distinguish frequency variation from algorithmic effects.
