#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="$(cd "${1:?immutable build}" && pwd)"
OUT="${2:?output directory}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
module load cuda/12.2
export NVIDIA_TF32_OVERRIDE=0
(cd "$ORIGIN" && sha256sum --check source.sha256) > "$OUT/source-verification.txt"
cp scripts/verify_strassen_small_primitive.sh "$OUT/verification-source.sh"
for tool in memcheck racecheck; do
    sizes=120,128,1536,2056
    if [[ "$tool" == racecheck ]]; then sizes=120,128; fi
    name=deferred_exact_swizzle_coalesced64_k16_r4_seven_atomic
    CMD=(compute-sanitizer --tool "$tool" --error-exitcode 42 --log-file "$OUT/$tool.log"
         "$ORIGIN/benchmark_strassen_onelevel" --mode endtoend
         --backends deferred_exact_swizzle_coalesced64_k16_r4 --only "$name"
         --sizes "$sizes" --seed-offset 1009 --validate-only --output "$OUT/$tool")
    printf '%q ' "${CMD[@]}" > "$OUT/$tool.command.txt"; printf '\n' >> "$OUT/$tool.command.txt"
    status=0
    timeout --signal=TERM --kill-after=30s 15m "${CMD[@]}" > "$OUT/$tool.stdout" 2>&1 || status=$?
    printf '{"exit_code":%d}\n' "$status" > "$OUT/$tool.status.json"
    if [[ "$status" != 0 ]]; then tail -30 "$OUT/$tool.stdout"; exit "$status"; fi
    echo "$tool PASS"
done
