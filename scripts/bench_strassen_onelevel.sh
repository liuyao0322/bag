#!/bin/bash
set -euo pipefail
main() {
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:?iteration label required}"; shift
OUT="${1:?output directory required}"; shift
if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo 'Use srun inside an existing GPU allocation.' >&2
    return 2
fi
cd "$ROOT"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TRAJ_DIR="$ROOT/trajectory/strassen_onelevel/${LABEL}_${TIMESTAMP}"
mkdir -p "$TRAJ_DIR"
cp -r solution/* "$TRAJ_DIR/"
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
mkdir -p "$OUT/source/primitive_headers" "$OUT/source/strassen_cuda"
HEADER_DIR="$ROOT/kernel/bag/include"
if [[ ! -f "$HEADER_DIR/prim_inter.cuh" ]]; then
    HEADER_DIR="$ROOT/benchmark_results/strassen_fp32_a100_20261017/source/primitive_headers"
fi
cp "$HEADER_DIR"/prim_*.cuh "$HEADER_DIR/base_components.cuh" "$OUT/source/primitive_headers/"
cp solution/strassen_cuda/*.cuh "$OUT/source/strassen_cuda/"
cp scripts/benchmark_strassen_fp32.cu scripts/benchmark_strassen_onelevel.cu scripts/bench_strassen_onelevel.sh scripts/summarize_strassen_onelevel.py "$OUT/source/"
(cd "$OUT" && sha256sum source/primitive_headers/*.cuh source/strassen_cuda/*.cuh source/*.*) > "$OUT/source.sha256"
BUILD=(nvcc -O3 -std=c++17 -lineinfo -Xptxas=-v -gencode arch=compute_80,code=sm_80
    -I "$OUT/source/primitive_headers" -I "$OUT/source/strassen_cuda"
    "$OUT/source/benchmark_strassen_onelevel.cu" -lcublas -o "$OUT/benchmark_strassen_onelevel")
printf '%q ' "${BUILD[@]}" > "$OUT/build-command.txt"
printf '\n' >> "$OUT/build-command.txt"
if "${BUILD[@]}" > "$OUT/build.log" 2>&1; then
    echo 'BUILD PASS'
else
    local status=$?
    tail -70 "$OUT/build.log"
    printf '{"phase":"build","exit_code":%d}\n' "$status" > "$OUT/execution_status.json"
    return "$status"
fi
printf '%q ' "$OUT/benchmark_strassen_onelevel" --output "$OUT/results" "$@" > "$OUT/run-command.txt"
printf '\n' >> "$OUT/run-command.txt"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 1 > "$OUT/telemetry.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
if timeout --signal=TERM --kill-after=30s 45m "$OUT/benchmark_strassen_onelevel" --output "$OUT/results" "$@" 2>&1 | tee "$OUT/run.log"; then
    printf '{"phase":"benchmark","exit_code":0}\n' > "$OUT/execution_status.json"
else
    local status=$?
    printf '{"phase":"benchmark","exit_code":%d}\n' "$status" > "$OUT/execution_status.json"
    return "$status"
fi
/share/home/liuyao/miniconda3/envs/kernelagent/bin/python -s scripts/summarize_strassen_onelevel.py "$OUT/results"
}
main "$@"; exit "$?"
