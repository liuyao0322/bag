#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="$(cd "${1:?immutable build directory required}" && pwd)"
OUT="${2:?verification directory required}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
module load cuda/12.2
export NVIDIA_TF32_OVERRIDE=0
BIN="$ORIGIN/benchmark_strassen_onelevel"
(cd "$ORIGIN" && sha256sum --check source.sha256) > "$OUT/source-verification.txt"
sha256sum "$BIN" "$ORIGIN/source.sha256" > "$OUT/build-origin.txt"
cp scripts/verify_strassen_batching.sh "$OUT/verification-source.sh"
{
    date --iso-8601=seconds
    hostname
    echo "SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    compute-sanitizer --version
    nsys --version
    ncu --version
    nvidia-smi --query-gpu=uuid,name,driver_version --format=csv
} > "$OUT/environment.txt"
run() {
    local name="$1"; shift
    printf '%q ' "$@" > "$OUT/$name.command.txt"; printf '\n' >> "$OUT/$name.command.txt"
    local status=0
    timeout --signal=TERM --kill-after=30s 15m "$@" > "$OUT/$name.stdout" 2>&1 || status=$?
    printf '{"exit_code":%d}\n' "$status" > "$OUT/$name.status.json"
    if [[ "$status" != 0 ]]; then tail -40 "$OUT/$name.stdout"; return "$status"; fi
    echo "$name PASS"
}
run memcheck_batched compute-sanitizer --tool memcheck --error-exitcode 42 --log-file "$OUT/memcheck_batched.log" \
    "$BIN" --mode endtoend --backends pipe64x32_k16_r4 --only pipe64x32_k16_r4_four_batched_atomic --sizes 120,2056 --validate-only --output "$OUT/memcheck_batched"
run memcheck_hybrid compute-sanitizer --tool memcheck --error-exitcode 42 --log-file "$OUT/memcheck_hybrid.log" \
    "$BIN" --mode endtoend --backends pipe64x32_k16_r4 --hybrid --only pipe64x32_k16_r4_hybrid2_four_batched_atomic --sizes 128,256 --validate-only --seed-offset 1009 --output "$OUT/memcheck_hybrid"
run memcheck_huge_hybrid compute-sanitizer --tool memcheck --error-exitcode 42 --log-file "$OUT/memcheck_huge_hybrid.log" \
    "$BIN" --mode endtoend --backends huge --hybrid --only huge_hybrid2_seven_atomic --sizes 512 --validate-only --output "$OUT/memcheck_huge_hybrid"
run racecheck_hybrid compute-sanitizer --tool racecheck --error-exitcode 42 --log-file "$OUT/racecheck_hybrid.log" \
    "$BIN" --mode endtoend --backends pipe64x32_k16_r4 --hybrid --only pipe64x32_k16_r4_hybrid2_four_batched_atomic --sizes 128 --validate-only --output "$OUT/racecheck_hybrid"
for name in cublas_full huge_seven_atomic huge_four_batched_atomic pipe64x32_k16_r4_four_batched_atomic; do
    run "timeline_$name" nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
        --capture-range-end=stop --export=sqlite --force-overwrite=true -o "$OUT/timeline_$name" \
        "$BIN" --mode endtoend --backends huge,pipe64x32_k16_r4 --only "$name" --sizes 2048 --profile-calls 20 --output "$OUT/timeline_$name"
done
run timeline_hybrid_large nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
    --capture-range-end=stop --export=sqlite --force-overwrite=true -o "$OUT/timeline_hybrid_large" \
    "$BIN" --mode endtoend --backends huge --hybrid --only huge_hybrid2_seven_atomic --sizes 20480 --profile-calls 3 --output "$OUT/timeline_hybrid_large"
# Older NCU uses a system-wide /tmp lock. Use a user-owned temporary directory
# only after checking that no other NCU collection is active on this node.
if pgrep -x ncu >/dev/null; then echo 'Another NCU collection is active; defer tensor audit.' >&2; exit 2; fi
NCU_TMP="/tmp/liuyao-strassen-ncu-${SLURM_JOB_ID}"
mkdir -p "$NCU_TMP"
run tensor_cublas env TMPDIR="$NCU_TMP" ncu --metrics sm__inst_executed_pipe_tensor.sum --kernel-name 'regex:.*sgemm.*' --launch-count 1 --csv --log-file "$OUT/tensor_cublas.csv" \
    "$BIN" --mode endtoend --backends huge --only cublas_full --sizes 2048 --validate-only --output "$OUT/tensor_cublas"
run cuobjdump cuobjdump --dump-sass "$BIN"
mv "$OUT/cuobjdump.stdout" "$OUT/sass.txt"
echo 'VERIFICATION COMMANDS COMPLETE'
