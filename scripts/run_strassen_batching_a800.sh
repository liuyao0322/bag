#!/bin/bash
# Allocation payload: execute automatically once the requested A800 is granted.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="${1:?immutable build directory required}"
OUT="${2:?suite directory required}"
mkdir -p "$OUT"
trap 'result=$?; printf "{\"exit_code\":%d,\"allocation\":\"%s\"}\n" "$result" "${SLURM_JOB_ID:-}" > "$OUT/a800_suite.status.json"' EXIT
[[ -n "${SLURM_JOB_ID:-}" ]]
ARGS=(--mode endtoend --sizes 512,1024,1536,2048,3072,4096,6144,8192,10240,12288,16384,20480
      --backends huge --schedules seven_atomic,four_batched_atomic --hybrid --rounds 5 --target-ms 200)
for seed in 0 1009; do
    name=a800_curve
    if [[ "$seed" != 0 ]]; then name=a800_replay_seed1009; fi
    srun --jobid="$SLURM_JOB_ID" bash scripts/replay_strassen_onelevel.sh "$ORIGIN" "$OUT/$name" "${ARGS[@]}" --seed-offset "$seed"
done
/share/home/liuyao/miniconda3/envs/kernelagent/bin/python -s scripts/analyze_strassen_batching.py "$OUT" > "$OUT/a800_analysis.log" 2>&1
echo 'A800 SUITE COMPLETE'
