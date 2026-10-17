#!/bin/bash
# Aggregate concurrent complete calls using NCU range replay. Diagnostic only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="$(cd "${1:?immutable build}" && pwd)"
OUT="${2:?output directory}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
module load cuda/12.2
export NVIDIA_TF32_OVERRIDE=0
if pgrep -x ncu >/dev/null; then echo 'Another NCU collection is active.' >&2; exit 2; fi
export TMPDIR="/tmp/liuyao-strassen-ncu-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cp scripts/profile_strassen_stalls.sh "$OUT/profile-source.sh"
(cd "$ORIGIN" && sha256sum --check source.sha256) > "$OUT/source-verification.txt"
sha256sum "$ORIGIN/benchmark_strassen_onelevel" > "$OUT/binary.sha256"
{ ncu --version; hostname; nvidia-smi --query-gpu=uuid,name,driver_version --format=csv; } > "$OUT/environment.txt"
for n in 1536 2048; do
    for name in cublas_full deferred_exact_swizzle_coalesced64_k16_r4_seven_atomic deferred_exact_swizzle_coalesced64_k16_r4_four_batched_atomic; do
        stem="$OUT/n${n}_$name"
        CMD=(ncu --replay-mode range --launch-count 1
             --cache-control none --clock-control none
             --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__sass_thread_inst_executed_op_ffma_pred_on.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
             --csv --log-file "$stem.csv"
             "$ORIGIN/benchmark_strassen_onelevel" --mode endtoend
             --backends deferred_exact_swizzle_coalesced64_k16_r4 --schedules seven_atomic,four_batched_atomic
             --only "$name" --sizes "$n" --profile-calls 10 --output "$stem")
        printf '%q ' "${CMD[@]}" > "$stem.command.txt"; printf '\n' >> "$stem.command.txt"
        status=0
        timeout --signal=TERM --kill-after=30s 15m "${CMD[@]}" > "$stem.stdout" 2>&1 || status=$?
        printf '{"exit_code":%d}\n' "$status" > "$stem.status.json"
        if [[ "$status" != 0 ]]; then tail -30 "$stem.stdout"; exit "$status"; fi
        echo "N=$n $name RANGE PASS"
    done
done
