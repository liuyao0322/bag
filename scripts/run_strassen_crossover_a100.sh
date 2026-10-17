#!/bin/bash
# Bounded configuration audit; reuse the immutable arithmetic executable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="${1:?immutable build required}"
OUT="${2:?audit directory required}"
mkdir -p "$OUT"
trap 'result=$?; printf "{\"exit_code\":%d,\"allocation\":\"%s\"}\n" "$result" "${SLURM_JOB_ID:-}" > "$OUT/suite.status.json"' EXIT
[[ -n "${SLURM_JOB_ID:-}" ]]
for seed in 0 1009; do
    srun --jobid="$SLURM_JOB_ID" bash scripts/replay_strassen_onelevel.sh \
        "$ORIGIN" "$OUT/seed$seed" --mode endtoend \
        --sizes 1024,1280,1536,1792,2048,2560,3072 \
        --schedules seven_atomic,four_batched_atomic \
        --seed-offset "$seed" --rounds 5 --target-ms 200 --samples 128
done
