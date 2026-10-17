#!/bin/bash
# Run inside an assigned Ampere GPU allocation (salloc, then srun --jobid=...).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo 'Run on an allocated GPU with srun --jobid=<id> bash scripts/run_strassen_fp32.sh <output-dir> [benchmark flags]' >&2
    exit 2
fi
OUT="${1:?output directory required}"
shift
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$ROOT"
module load cuda/12.2
export NVIDIA_TF32_OVERRIDE=0
{
    date --iso-8601=seconds
    hostname
    echo "SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
    nvcc --version
    nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit,compute_mode --format=csv
    git rev-parse HEAD
    git status --short
} > "$OUT/environment.txt"
mkdir -p "$OUT/source/primitive_headers"
cp "$ROOT"/kernel/bag/include/prim_*.cuh "$OUT/source/primitive_headers/"
cp "$ROOT/kernel/bag/include/base_components.cuh" "$OUT/source/primitive_headers/"
cp "$ROOT/scripts/benchmark_strassen_fp32.cu" "$ROOT/scripts/run_strassen_fp32.sh" "$OUT/source/"
sha256sum "$OUT"/source/primitive_headers/*.cuh "$OUT"/source/*.* > "$OUT/source.sha256"
BUILD=(nvcc -O3 -std=c++17 -lineinfo -Xptxas=-v -gencode arch=compute_80,code=sm_80
    -I "$OUT/source/primitive_headers" "$OUT/source/benchmark_strassen_fp32.cu"
    -lcublas -o "$OUT/benchmark_strassen_fp32")
printf '%q ' "${BUILD[@]}" > "$OUT/build-command.txt"
printf '\n' >> "$OUT/build-command.txt"
"${BUILD[@]}" > "$OUT/build.log" 2>&1
printf '%q ' "$OUT/benchmark_strassen_fp32" --output "$OUT/results" "$@" > "$OUT/run-command.txt"
printf '\n' >> "$OUT/run-command.txt"
# GPU telemetry runs concurrently, outside the CUDA event intervals.
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 1 > "$OUT/telemetry.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
timeout --signal=TERM --kill-after=30s 45m "$OUT/benchmark_strassen_fp32" --output "$OUT/results" "$@" 2>&1 | tee "$OUT/run.log"
