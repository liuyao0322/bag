# CUDA alternative-basis matrix multiplication

Implemented BAG.md using the previously tuned CUDA primitive, unchanged.
All seven products, forward/inverse transforms and recursive levels are included.
Operation: square column-major FP32 C <- AB+C; same-run cuBLAS PEDANTIC, TF32 disabled.

## Frozen configuration and independent replay

Selected before independent replay: `bag_column_minimal_d1_batched`; see selection.json.
Each independent input uses seven shuffled rounds, target300ms per round.
These are complete calls including transforms, scratch resets, products, decoding and stream joins.
Allocation, library/stream creation and input generation are outside timing.

| Seed | N | BAG ms | cuBLAS ms | Strassen ms | cuBLAS/BAG |
|---:|---:|---:|---:|---:|---:|
| 1009 | 1280 | 0.274927 | 0.257603 | 0.253180 | 0.9370x |
| 1009 | 1536 | 0.450311 | 0.431820 | 0.426007 | 0.9589x |
| 1009 | 1792 | 0.678217 | 0.689315 | 0.648007 | 1.0164x |
| 1009 | 2048 | 0.992140 | 0.984703 | 0.954365 | 0.9925x |
| 2027 | 1280 | 0.275268 | 0.258502 | 0.254839 | 0.9391x |
| 2027 | 1536 | 0.453347 | 0.435993 | 0.428813 | 0.9617x |
| 2027 | 1792 | 0.684262 | 0.695335 | 0.653726 | 1.0162x |
| 2027 | 2048 | 1.000247 | 0.985656 | 0.970229 | 0.9854x |

N=1536 BAG does not exceed same-run cuBLAS in this implementation.
The earlier Strassen N=1536 acceptance does not imply the BAG transforms are free.
No ordinary-GEMM replacement, reduced tolerance or omitted slower layout/schedule is used.

## Correctness and recursion

Depth1/2/3, both basis layouts and all three schedules passed two independent input sets.
Sizes include8/24/32/64/120/128/256; incompatible recursion shapes are explicitly skipped.
FP64 reference: exhaustive at N<=128, otherwise deterministic128 positions including corners.
Checks cover nonzero initial C and one/three consecutive accumulated calls.
Unchanged limits: relative L2<=3e-5, scaled maximum error<=1e-4.
Ordinary-run FP64 records: 608; all pass.
Memcheck: N24/128/2056, depth1/2 where legal, both layouts/all schedules, zero errors.
Racecheck: N24/128, depth1/2 where legal, both layouts/all schedules, zero shared hazards.
Global output race safety also relies on atomics for overlapping worker updates and caller ordering.

## Storage and schedules

Minimal mode materializes SA=A12-A21+A22 and SB=B12-B21+B22; all remaining sums are fused.
SA/SB share an N x (N/2) buffer with leading dimension N; independent transformed output T is zeroed.
One-level scratch:6*N*N bytes minimal,12*N*N full. Workspaces are preallocated and not reentrant.
Minimal batched uses6 product launches; full batched uses4. Each also includes prepare/decode.
Serial, seven-stream and both basis variants remain available. All negative results are retained.

## Recursive complete-call performance

| N | Configuration | BAG ms | Same-run cuBLAS ms | Speedup |
|---:|---|---:|---:|---:|
| 512 | bag_column_minimal_d1_batched | 0.049697 | 0.028259 | 0.5686x |
| 512 | bag_column_minimal_d2_batched | 0.292181 | 0.028259 | 0.0967x |
| 1536 | bag_column_minimal_d1_batched | 0.454435 | 0.435133 | 0.9575x |
| 1536 | bag_column_minimal_d2_batched | 0.900457 | 0.435133 | 0.4832x |

## Large-size regression, all configurations

| N | Configuration | BAG ms | Same-run cuBLAS ms | Speedup |
|---:|---|---:|---:|---:|
| 4096 | bag_column_full_d1_batched | 7.702528 | 7.437532 | 0.9656x |
| 4096 | bag_column_full_d1_serial | 8.058328 | 7.437532 | 0.9230x |
| 4096 | bag_column_full_d1_seven | 7.596544 | 7.437532 | 0.9791x |
| 4096 | bag_column_minimal_d1_batched | 7.558290 | 7.437532 | 0.9840x |
| 4096 | bag_column_minimal_d1_serial | 7.946555 | 7.437532 | 0.9359x |
| 4096 | bag_column_minimal_d1_seven | 7.501458 | 7.437532 | 0.9915x |
| 8192 | bag_column_full_d1_batched | 61.350911 | 57.549825 | 0.9380x |
| 8192 | bag_column_full_d1_serial | 61.284353 | 57.549825 | 0.9391x |
| 8192 | bag_column_full_d1_seven | 62.246231 | 57.549825 | 0.9246x |
| 8192 | bag_column_minimal_d1_batched | 61.374807 | 57.549825 | 0.9377x |
| 8192 | bag_column_minimal_d1_serial | 61.250559 | 57.549825 | 0.9396x |
| 8192 | bag_column_minimal_d1_seven | 61.617493 | 57.549825 | 0.9340x |
| 4096 | bag_huge_full_d1_batched | 7.556242 | 7.392549 | 0.9783x |
| 4096 | bag_huge_full_d1_serial | 9.345210 | 7.392549 | 0.7911x |
| 4096 | bag_huge_full_d1_seven | 7.513161 | 7.392549 | 0.9839x |
| 4096 | bag_huge_minimal_d1_batched | 7.401545 | 7.392549 | 0.9988x |
| 4096 | bag_huge_minimal_d1_serial | 9.257611 | 7.392549 | 0.7985x |
| 4096 | bag_huge_minimal_d1_seven | 7.405056 | 7.392549 | 0.9983x |
| 8192 | bag_huge_full_d1_batched | 56.947713 | 58.020182 | 1.0188x |
| 8192 | bag_huge_full_d1_serial | 59.181737 | 58.020182 | 0.9804x |
| 8192 | bag_huge_full_d1_seven | 57.039530 | 58.020182 | 1.0172x |
| 8192 | bag_huge_minimal_d1_batched | 56.602285 | 58.020182 | 1.0251x |
| 8192 | bag_huge_minimal_d1_serial | 58.490880 | 58.020182 | 0.9920x |
| 8192 | bag_huge_minimal_d1_seven | 56.337748 | 58.020182 | 1.0299x |

## Evidence and initial runner issue

Source archives/hashes, exact build/run commands, environment, raw rounds, workspace sizes and FP64 errors are preserved per run.
iter17_initial failed before compilation because a non-login shell lacked the module function.
iter18_initial completed all checks and rounds with program status0, while its login-shell srun returned1.
Runners now initialize /etc/profile.d/modules.sh explicitly in a normal shell; independent suite status is0.
The frozen arithmetic binary was not changed during replay. See verification.json and inventory.json.
A100 evidence does not establish A800 performance. Earlier queued Strassen A800 jobs remain separate.

## Isolated reused primitive profile

One serial M1 primitive at leaf768 was profiled with NCU kernel replay.
This is an isolated resource/memory diagnostic, not concurrent occupancy or complete-call timing.
Registers/thread: 119; threads/CTA: 128; CTA grid: 144.
Static shared bytes: 16384; theoretical occupancy: 25%.
Shared load/store bank conflicts: 0/0.
The profiled kernel is the reused strassen_cuda::simt_primitive; raw counters and exact command are in profile/.
