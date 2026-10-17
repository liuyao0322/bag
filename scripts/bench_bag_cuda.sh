#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LABEL="${1:?iteration label required}"; shift
OUT="${1:?output directory required}"; shift
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
TRAJ_DIR="$ROOT/trajectory/bag_cuda/${LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TRAJ_DIR" "$OUT/source/bag_cuda" "$OUT/source/strassen_cuda" "$OUT/source/primitive_headers"
cp -r solution/* "$TRAJ_DIR/"
cp solution/bag_cuda/*.cuh "$OUT/source/bag_cuda/"
cp solution/strassen_cuda/*.cuh "$OUT/source/strassen_cuda/"
cp benchmark_results/strassen_fp32_a100_20261017/source/primitive_headers/*.cuh "$OUT/source/primitive_headers/"
cp scripts/benchmark_bag_cuda.cu scripts/benchmark_strassen_fp32.cu scripts/bench_bag_cuda.sh "$OUT/source/"
(cd "$OUT" && sha256sum source/bag_cuda/* source/strassen_cuda/* source/primitive_headers/* source/*.*) > "$OUT/source.sha256"
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module load cuda/12.2
export NVIDIA_TF32_OVERRIDE=0
{
    date --iso-8601=seconds
    hostname
    echo "SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
    nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit --format=csv
    nvcc --version
    git rev-parse HEAD
} > "$OUT/environment.txt"
BUILD=(nvcc -O3 -std=c++17 -lineinfo -Xptxas=-v -gencode arch=compute_80,code=sm_80
       -I "$OUT/source/primitive_headers" -I "$OUT/source/strassen_cuda" -I "$OUT/source/bag_cuda"
       "$OUT/source/benchmark_bag_cuda.cu" -lcublas -o "$OUT/benchmark_bag_cuda")
printf '%q ' "${BUILD[@]}" > "$OUT/build-command.txt"; printf '\n' >> "$OUT/build-command.txt"
if "${BUILD[@]}" > "$OUT/build.log" 2>&1; then
    echo 'BUILD PASS'
else
    status=$?
    printf '{"phase":"build","exit_code":%d}\n' "$status" > "$OUT/execution_status.json"
    tail -40 "$OUT/build.log"; exit "$status"
fi
CMD=("$OUT/benchmark_bag_cuda" --output "$OUT/results" "$@")
printf '%q ' "${CMD[@]}" > "$OUT/run-command.txt"; printf '\n' >> "$OUT/run-command.txt"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw --format=csv -l 1 > "$OUT/telemetry.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
status=0
timeout --signal=TERM --kill-after=30s 45m "${CMD[@]}" 2>&1 | tee "$OUT/run.log" || status=$?
printf '{"phase":"benchmark","exit_code":%d}\n' "$status" > "$OUT/execution_status.json"
exit "$status"
