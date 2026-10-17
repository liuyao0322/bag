# Pure FP32, beta=1 CUDA cost baseline

GPU: **NVIDIA A100 80GB PCIe**. CUDA runtime 12020; cuBLAS 120205.
`cublasSgemm`, `CUBLAS_PEDANTIC_MATH`, `NVIDIA_TF32_OVERRIDE=0`; column-major, alpha=beta=1.

## Measurement

Each case has 5 CUDA-event batches targeting 200 ms each. Each batch contains 3 to 10,000 calls; latency is batch time divided by calls. The cap can make the shortest cases run for less than the target duration. Twenty ms of per-case warmup precedes randomized case order in each round. Inputs and nonzero output seeds are initialized outside timing. Calls inside each batch repeatedly accumulate into C. Allocation, validation, event creation and library initialization are excluded. A single non-default CUDA stream covers all timed work; CUDA Graphs are not used.

Compact leaves have ld=N/2. Parent-view leaves and primitives have ld=N. Primitive inputs X/Y and V/W use the top-left/bottom-right quadrants of initialized parent matrices; D/E use separate top-left/bottom-right output quadrants. Each of M1..M7 reproduces the coefficient pattern in the existing one-level implementation, using these representative operands. These isolated measurements do not execute the complete Strassen algebra or concurrent scheduling.

## Correctness

All 352 case/accumulation checks passed against CPU FP64 dot products after 1 and 3 calls. Maximum sampled relative L2 error: 2.93339e-06. The acceptance limits are relative L2 <=3e-5 and max absolute error/(1+max absolute reference) <=1e-4. Untouched output samples must match the initial seed exactly. Output matrices with side <=128 are checked exhaustively; larger outputs use up to 128 distinct deterministic coordinates per output, including corners, plus sentinel samples in untouched quadrants. This is sampled numerical validation, not an exhaustive correctness proof for large matrices.

## Costs

All times below are milliseconds; per-case values are medians of batch averages.

| N | Full cuBLAS | Half compact | Half parent ld | Primitive GEMM | Sum of 7 primitives | Full / (7 half GEMMs) | Full / sum primitives |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.028272 | 0.012330 | 0.012384 | 0.011504 | 0.101048 | 0.326 | 0.280 |
| 768 | 0.063537 | 0.015770 | 0.015767 | 0.024927 | 0.208101 | 0.576 | 0.305 |
| 1024 | 0.128613 | 0.028290 | 0.027899 | 0.039043 | 0.315739 | 0.659 | 0.407 |
| 1280 | 0.256984 | 0.038324 | 0.038287 | 0.070118 | 0.554725 | 0.959 | 0.463 |
| 1536 | 0.434062 | 0.063574 | 0.063573 | 0.088158 | 0.711108 | 0.975 | 0.610 |
| 1792 | 0.695737 | 0.094912 | 0.094902 | 0.129677 | 1.056535 | 1.047 | 0.659 |
| 2048 | 0.977161 | 0.128786 | 0.128811 | 0.258245 | 2.101285 | 1.084 | 0.465 |
| 3072 | 3.206592 | 0.438534 | 0.440020 | 0.663957 | 5.080352 | 1.041 | 0.631 |
| 4096 | 7.447040 | 0.984683 | 0.983055 | 1.209188 | 10.532750 | 1.082 | 0.707 |
| 6144 | 25.401983 | 3.238499 | 3.237211 | 3.767451 | 30.322420 | 1.121 | 0.838 |
| 7680 | 48.596786 | 6.267680 | 6.266560 | 7.132379 | 59.187527 | 1.108 | 0.821 |
| 8192 | 58.451199 | 7.507930 | 7.527917 | 9.065007 | 73.152671 | 1.109 | 0.799 |
| 10240 | 118.783661 | 14.500206 | 14.507739 | 15.172242 | 111.932810 | 1.170 | 1.061 |
| 12288 | 204.808187 | 25.539200 | 25.562113 | 26.551552 | 195.356417 | 1.145 | 1.048 |
| 16384 | 484.027751 | 58.561535 | 58.561790 | 61.291264 | 453.602123 | 1.181 | 1.067 |
| 20480 | 950.978597 | 118.993235 | 118.997670 | 121.071279 | 893.870758 | 1.142 | 1.064 |

The two ratios are **serial cost proxies**, not measured end-to-end speedups or hardware bounds. They omit actual cross-product cache interaction, dispatch ordering, and concurrency. The cuBLAS leaf proxy additionally omits Strassen input combinations and multi-output accumulation. M1..M7 costs are summed from isolated medians.

Largest per-case coefficient of variation across rounds: 6.42%. `summary.csv` retains p20/p80, min/max and CV for every case; `results.rounds.csv` retains every timing batch. GPU clocks are not locked; `telemetry.csv` records clocks, utilization, power and temperature. Other GPUs on the node may be in use; the allocated GPU is recorded in `environment.txt`.

## Files

- `results.metadata.json`: runtime, precision and shape configuration.
- `results.correctness.csv`: numerical checks and counts.
- `results.rounds.csv`: raw timing rounds.
- `summary.csv`: per-case timing distributions.
- `cost_summary.csv`: full/leaf/primitive costs and serial proxies.
- `source/`, `source.sha256`, `build-command.txt`, `build.log`: exact compiled source and build evidence.
- `environment.txt`, `run-command.txt`, `run.log`, `telemetry.csv`: execution evidence.

## Additional verification and execution audit

Compute Sanitizer memcheck reported **0 errors** for primitive_m7 at N=512,1024,2048,8448, covering small/medium/large/huge kernels. This instrumentation checks representative cases, not every measured shape.

Nsight Compute sampled `ampere_sgemm_128x64_nn` at N=2048: `sm__inst_executed_pipe_tensor.sum = 0`, `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum = 8594128896`. Static disassembly of the standalone binary has 18,720 FFMA instructions and no Tensor Core instruction mnemonics. The cuBLAS sample is a representative dynamic check; FP32 configuration is enforced for all benchmark cases.

The measurement binary printed COMPLETE after all 880 timing batches. The original wrapper then exited with code 2 because its file was edited while the shell was executing it. The revised wrapper parses its body before starting work and records its exit status. A separate replay covering four kernel buckets completed with exit code 0, 44 timing rows and 88 numerical checks; the compiled CUDA source was byte-for-byte identical. Original measurement rows were retained. See `execution_audit.json` and `verification/entry_*`.

The summary completeness gate was tested with missing rounds, duplicate rounds, a failed numerical check and NaN latency; all four corrupt inputs were rejected. See `summary_gate_tests.json`.
