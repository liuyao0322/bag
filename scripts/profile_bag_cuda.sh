#!/bin/bash
# One isolated primitive profile; never used as the complete-call timing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="$(cd "${1:?immutable build directory}" && pwd)"
OUT="${2:?output directory}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module load cuda/12.2
export NVIDIA_TF32_OVERRIDE=0
(cd "$ORIGIN" && sha256sum --check source.sha256) > "$OUT/source-verification.txt"
sha256sum "$ORIGIN/benchmark_bag_cuda" > "$OUT/build-origin.txt"
CMD=(ncu --target-processes all --kernel-name-base function --kernel-name regex:simt_primitive
     --launch-count 1 --section LaunchStats --section Occupancy
     --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum
     --csv --log-file "$OUT/primitive.csv"
     "$ORIGIN/benchmark_bag_cuda" --sizes 1536 --depths 1 --basis minimal --schedule serial
     --only bag_column_minimal_d1_serial --seed-offset 1009 --validate-only --output "$OUT/results")
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
status=0
timeout --signal=TERM --kill-after=30s 10m "${CMD[@]}" > "$OUT/stdout.txt" 2>&1 || status=$?
printf '{"exit_code":%d}\n' "$status" > "$OUT/status.json"
exit "$status"
