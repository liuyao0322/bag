#!/bin/bash
set -euo pipefail
main() {
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="$(cd "${1:?build directory required}" && pwd)"; shift
OUT="${1:?verification directory required}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
module load cuda/12.2
export NVIDIA_TF32_OVERRIDE=0
BIN="$ORIGIN/benchmark_strassen_onelevel"
PY=/share/home/liuyao/miniconda3/envs/kernelagent/bin/python
(cd "$ORIGIN" && sha256sum --check source.sha256) > "$OUT/source-verification.txt"
sha256sum "$BIN" "$ORIGIN/source.sha256" > "$OUT/build-origin.txt"
cp scripts/verify_strassen_onelevel.sh "$OUT/verification-source.sh"
{
    date --iso-8601=seconds
    hostname
    echo "SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
    compute-sanitizer --version
    ncu --version
    nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem --format=csv
} > "$OUT/environment.txt"
run() {
    local name="$1"; shift
    printf '%q ' "$@" > "$OUT/$name.command.txt"; printf '\n' >> "$OUT/$name.command.txt"
    if "$@" > "$OUT/$name.stdout" 2>&1; then
        printf '{"exit_code":0}\n' > "$OUT/$name.status.json"
        echo "$name PASS"
    else
        local status=$?
        printf '{"exit_code":%d}\n' "$status" > "$OUT/$name.status.json"
        tail -45 "$OUT/$name.stdout"
        return "$status"
    fi
}
run memcheck_mask compute-sanitizer --tool memcheck --error-exitcode 42 --log-file "$OUT/memcheck_mask.log" \
    "$BIN" --mode endtoend --backends pipe64x32_k16_r4 --only pipe64x32_k16_r4_paired_normal --sizes 120,2056 --validate-only --output "$OUT/memcheck_mask"
run memcheck_huge compute-sanitizer --tool memcheck --error-exitcode 42 --log-file "$OUT/memcheck_huge.log" \
    "$BIN" --mode endtoend --backends huge --only huge_paired_atomic --sizes 256 --validate-only --output "$OUT/memcheck_huge"
run racecheck compute-sanitizer --tool racecheck --error-exitcode 42 --log-file "$OUT/racecheck.log" \
    "$BIN" --mode endtoend --backends pipe64x32_k16_r4 --only pipe64x32_k16_r4_paired_normal --sizes 120 --validate-only --output "$OUT/racecheck"
run profile_small ncu --set full --kernel-name 'regex:.*simt_primitive.*' --launch-count 1 --csv --log-file "$OUT/profile_small.csv" \
    "$BIN" --mode ablation --backends pipe64x32_k16_r4 --only pipe64x32_k16_r4_gemm_atomic --sizes 2048 --validate-only --output "$OUT/profile_small"
run tensor_cublas ncu --metrics sm__inst_executed_pipe_tensor.sum --kernel-name 'regex:.*sgemm.*' --launch-count 1 --csv --log-file "$OUT/tensor_cublas.csv" \
    "$BIN" --mode endtoend --backends huge --only cublas_full --sizes 2048 --validate-only --output "$OUT/tensor_cublas"
run cuobjdump cuobjdump --dump-sass "$BIN"
mv "$OUT/cuobjdump.stdout" "$OUT/sass.txt"
"$PY" -s - "$OUT" <<'PY'
import csv, io, json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
sass = (out/'sass.txt').read_text()
summary = {'ffma_instructions': len(re.findall(r'\bFFMA\b',sass)),
           'tensor_instructions': len(re.findall(r'\b(?:HMMA|IMMA|DMMA|MMA|WGMMA|HGMMA)\b',sass))}
assert summary['ffma_instructions'] > 0 and summary['tensor_instructions'] == 0
for name in ('profile_small','tensor_cublas'):
    content=(out/f'{name}.csv').read_text()
    rows=list(csv.DictReader(io.StringIO(content[content.index('"ID"'):])) )
    metrics={r.get('Metric Name'):r.get('Metric Value') for r in rows if r.get('Metric Name')}
    if name=='profile_small':
        wanted=('Registers Per Thread','Theoretical Occupancy','Achieved Occupancy','Waves Per SM')
        summary['small_profile']={k:metrics[k] for k in wanted}
    else:
        value=metrics['sm__inst_executed_pipe_tensor.sum']
        assert float(value.replace(',','')) == 0
        summary['cublas_tensor_instructions']=value
for name in ('memcheck_mask','memcheck_huge','racecheck'):
    content=(out/f'{name}.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in content or 'RACECHECK SUMMARY: 0 hazards' in content
    checks=list(csv.DictReader(open(out/f'{name}.correctness.csv')))
    assert checks and all(r['pass']=='1' for r in checks)
    summary[f'{name}_fp64_checks']=len(checks)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY
}
main "$@"; exit "$?"
