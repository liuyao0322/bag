#!/bin/bash
# Run inside the existing A100 allocation after the first curve completes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="${1:?immutable build directory required}"
OUT="${2:?suite directory required}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
trap 'result=$?; printf "{\"exit_code\":%d}\n" "$result" > "$OUT/a100_followup.status.json"' EXIT
while [[ ! -f "$OUT/a100_curve/execution_status.json" ]]; do sleep 5; done
/share/home/liuyao/miniconda3/envs/kernelagent/bin/python -s -c 'import json,sys; assert json.load(open(sys.argv[1]))["exit_code"]==0' "$OUT/a100_curve/execution_status.json"
bash scripts/replay_strassen_onelevel.sh "$ORIGIN" "$OUT/a100_replay_seed1009" \
    --mode endtoend --sizes 512,1024,1536,2048,3072,4096,6144,8192,10240,12288,16384,20480 \
    --backends huge --schedules seven_atomic,four_batched_atomic --hybrid --rounds 5 --target-ms 200 --seed-offset 1009
for seed in 0 1009; do
    bash scripts/replay_strassen_onelevel.sh "$ORIGIN" "$OUT/a100_crossover_seed$seed" \
        --mode endtoend --sizes 4096,4352,4608,4864,5120,5632,6144,6656,7168,7680,8192 \
        --backends huge --schedules seven_atomic,four_batched_atomic --hybrid --rounds 5 --target-ms 200 --seed-offset "$seed"
done
bash scripts/replay_strassen_onelevel.sh "$ORIGIN" "$OUT/a100_small_seed1009" \
    --mode endtoend --sizes 512,1024,1536,2048,3072,4096 --backends huge,pipe64x32_k16_r4 \
    --schedules seven_atomic,four_batched_atomic --rounds 5 --target-ms 200 --seed-offset 1009
bash scripts/verify_strassen_batching.sh "$ORIGIN" "$OUT/verification"
/share/home/liuyao/miniconda3/envs/kernelagent/bin/python -s scripts/analyze_strassen_batching.py "$OUT" > "$OUT/analysis.log" 2>&1
echo 'A100 FOLLOWUP COMPLETE'
