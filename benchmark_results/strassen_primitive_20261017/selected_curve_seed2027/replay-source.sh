#!/bin/bash
# Replay an immutable build; source hash verification ties it to its archive.
set -euo pipefail
main() {
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="$(cd "${1:?original build directory required}" && pwd)"; shift
OUT="${1:?new output directory required}"; shift
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
module load cuda/12.2
export NVIDIA_TF32_OVERRIDE=0
(cd "$ORIGIN" && sha256sum --check source.sha256) > "$OUT/source-verification.txt"
{
    echo "origin=$ORIGIN"
    sha256sum "$ORIGIN/benchmark_strassen_onelevel" "$ORIGIN/source.sha256"
} > "$OUT/build-origin.txt"
{
    date --iso-8601=seconds
    hostname
    echo "SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
    nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit --format=csv
    git rev-parse HEAD
} > "$OUT/environment.txt"
printf '%q ' "$ORIGIN/benchmark_strassen_onelevel" --output "$OUT/results" "$@" > "$OUT/run-command.txt"
printf '\n' >> "$OUT/run-command.txt"
cp scripts/replay_strassen_onelevel.sh "$OUT/replay-source.sh"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 1 > "$OUT/telemetry.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
if timeout --signal=TERM --kill-after=30s 45m "$ORIGIN/benchmark_strassen_onelevel" --output "$OUT/results" "$@" 2>&1 | tee "$OUT/run.log"; then
    printf '{"exit_code":0}\n' > "$OUT/execution_status.json"
else
    local status=$?
    printf '{"exit_code":%d}\n' "$status" > "$OUT/execution_status.json"
    return "$status"
fi
/share/home/liuyao/miniconda3/envs/kernelagent/bin/python -s scripts/summarize_strassen_onelevel.py "$OUT/results"
}
main "$@"; exit "$?"
