#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="${1:?immutable build directory}"
OUT="${2:?suite output directory}"
[[ -n "${SLURM_JOB_ID:-}" ]]
mkdir -p "$OUT"
trap 'result=$?; printf "{\"exit_code\":%d}\n" "$result" > "$OUT/validation.status.json"' EXIT
for seed in 1009 2027; do
    srun --jobid="$SLURM_JOB_ID" bash scripts/replay_bag_cuda.sh "$ORIGIN" "$OUT/holdout_seed$seed" \
        --sizes 1280,1536,1792,2048 --depths 1 --basis minimal --schedule batched \
        --seed-offset "$seed" --rounds 7 --target-ms 300
    srun --jobid="$SLURM_JOB_ID" bash scripts/replay_bag_cuda.sh "$ORIGIN" "$OUT/recursive_seed$seed" \
        --sizes 8,24,32,64,120,128,256 --depths 1,2,3 --basis both --schedule all \
        --seed-offset "$seed" --validate-only
done
srun --jobid="$SLURM_JOB_ID" bash scripts/replay_bag_cuda.sh "$ORIGIN" "$OUT/recursive_performance" \
    --sizes 512,1536 --depths 1,2 --basis minimal --schedule batched \
    --seed-offset 2027 --rounds 5 --target-ms 200
for backend in column huge; do
    srun --jobid="$SLURM_JOB_ID" bash scripts/replay_bag_cuda.sh "$ORIGIN" "$OUT/large_$backend" \
        --sizes 4096,8192 --depths 1 --basis both --schedule all --backend "$backend" \
        --seed-offset 2027 --rounds 3 --target-ms 100
done
srun --jobid="$SLURM_JOB_ID" bash scripts/verify_bag_cuda.sh "$ORIGIN" "$OUT/verification"
echo 'BAG CUDA VALIDATION COMPLETE'
