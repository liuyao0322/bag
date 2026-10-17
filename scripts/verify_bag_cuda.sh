#!/bin/bash
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
cp scripts/verify_bag_cuda.sh "$OUT/verification-source.sh"
for tool in memcheck racecheck; do
    sizes=24,128,2056
    if [[ "$tool" == racecheck ]]; then sizes=24,128; fi
    CMD=(compute-sanitizer --tool "$tool" --error-exitcode 42 --log-file "$OUT/$tool.log"
         "$ORIGIN/benchmark_bag_cuda" --sizes "$sizes" --depths 1,2
         --basis both --schedule all --backend column --seed-offset 1009
         --validate-only --output "$OUT/$tool")
    printf '%q ' "${CMD[@]}" > "$OUT/$tool.command.txt"; printf '\n' >> "$OUT/$tool.command.txt"
    status=0
    timeout --signal=TERM --kill-after=30s 15m "${CMD[@]}" > "$OUT/$tool.stdout" 2>&1 || status=$?
    printf '{"exit_code":%d}\n' "$status" > "$OUT/$tool.status.json"
    if [[ "$status" != 0 ]]; then tail -30 "$OUT/$tool.stdout"; exit "$status"; fi
    echo "$tool PASS"
done
