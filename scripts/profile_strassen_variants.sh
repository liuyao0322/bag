#!/bin/bash
# Kernel replay serializes launches: use these counters for primitive diagnosis,
# and ordinary complete-call measurements for performance decisions.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="$(cd "${1:?immutable build}" && pwd)"
OUT="${2:?output directory}"
BACKEND="${3:-pipe64_k16_r4}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
module load cuda/12.2
export NVIDIA_TF32_OVERRIDE=0
if pgrep -x ncu >/dev/null; then echo 'Another NCU collection is active.' >&2; exit 2; fi
export TMPDIR="/tmp/liuyao-strassen-ncu-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
(cd "$ORIGIN" && sha256sum --check source.sha256) > "$OUT/source-verification.txt"
sha256sum "$ORIGIN/benchmark_strassen_onelevel" > "$OUT/binary.sha256"
cp scripts/profile_strassen_variants.sh "$OUT/profile-source.sh"
{ ncu --version; hostname; nvidia-smi --query-gpu=uuid,name,driver_version --format=csv; } > "$OUT/environment.txt"
for n in 1536 2048; do
    CMD=(ncu --profile-from-start off --launch-count 7
         --section LaunchStats --section Occupancy --section SchedulerStats
         --section WarpStateStats --section MemoryWorkloadAnalysis --section SpeedOfLight
         --section SourceCounters
         --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
         --csv --log-file "$OUT/n$n.csv"
         "$ORIGIN/benchmark_strassen_onelevel" --mode endtoend --backends "$BACKEND"
         --only "${BACKEND}_seven_atomic" --sizes "$n" --profile-calls 1 --output "$OUT/n$n")
    printf '%q ' "${CMD[@]}" > "$OUT/n$n.command.txt"; printf '\n' >> "$OUT/n$n.command.txt"
    status=0
    timeout --signal=TERM --kill-after=30s 15m "${CMD[@]}" > "$OUT/n$n.stdout" 2>&1 || status=$?
    printf '{"exit_code":%d}\n' "$status" > "$OUT/n$n.status.json"
    if [[ "$status" != 0 ]]; then tail -30 "$OUT/n$n.stdout"; exit "$status"; fi
    echo "N=$n $BACKEND PROFILE PASS"
done
