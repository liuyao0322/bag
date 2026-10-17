#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="$(cd "${1:?immutable build directory}" && pwd)"; shift
OUT="${1:?output directory}"; shift
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
module load cuda/12.2
export NVIDIA_TF32_OVERRIDE=0
(cd "$ORIGIN" && sha256sum --check source.sha256) > "$OUT/source-verification.txt"
{
    echo "origin=$ORIGIN"
    sha256sum "$ORIGIN/benchmark_bag_cuda" "$ORIGIN/source.sha256"
} > "$OUT/build-origin.txt"
{
    date --iso-8601=seconds
    hostname
    echo "SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
    nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,power.limit --format=csv
    git rev-parse HEAD
} > "$OUT/environment.txt"
cp scripts/replay_bag_cuda.sh "$OUT/replay-source.sh"
CMD=("$ORIGIN/benchmark_bag_cuda" --output "$OUT/results" "$@")
printf '%q ' "${CMD[@]}" > "$OUT/run-command.txt"; printf '\n' >> "$OUT/run-command.txt"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw --format=csv -l 1 > "$OUT/telemetry.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
status=0
timeout --signal=TERM --kill-after=30s 45m "${CMD[@]}" 2>&1 | tee "$OUT/run.log" || status=$?
printf '{"exit_code":%d}\n' "$status" > "$OUT/execution_status.json"
exit "$status"
