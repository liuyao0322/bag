#!/bin/bash
# Replay the frozen A100-selected configuration without selecting on A800.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ORIGIN="${1:-$ROOT/benchmark_results/strassen_primitive_20261017/iter16_column_resources}"
OUT="${2:-$ROOT/benchmark_results/strassen_primitive_a800_20261017}"
mkdir -p "$OUT"
trap 'result=$?; printf "{\"exit_code\":%d,\"allocation\":\"%s\"}\n" "$result" "${SLURM_JOB_ID:-}" > "$OUT/a800_suite.status.json"' EXIT
[[ -n "${SLURM_JOB_ID:-}" ]]
cp scripts/run_strassen_column_a800.sh "$OUT/suite-source.sh"
for seed in 1009 2027; do
    srun --jobid="$SLURM_JOB_ID" bash scripts/replay_strassen_onelevel.sh "$ORIGIN" "$OUT/seed$seed" \
        --mode endtoend --sizes 1280,1536,1792,2048 --backends column_tile64_k16_r4x8 \
        --schedules seven_atomic,four_batched_atomic --seed-offset "$seed" --rounds 7 --target-ms 300
done
srun --jobid="$SLURM_JOB_ID" bash scripts/verify_strassen_small_primitive.sh "$ORIGIN" "$OUT/verification" \
    column_tile64_k16_r4x8 four_batched_atomic
/share/home/liuyao/miniconda3/envs/kernelagent/bin/python -s - "$OUT" <<'PY'
import csv, json, sys
from pathlib import Path
r = Path(sys.argv[1])
lines = ['# A800 frozen column candidate replay', '',
         'Fixed on A100 before this run: column_tile64_k16_r4x8 / four_batched_atomic.',
         'Pure FP32 C <- AB+C; complete seven products, all updates and joins timed.', '',
         '| Seed | N | Strassen ms | Same-run cuBLAS ms | Speedup |',
         '|---:|---:|---:|---:|---:|']
for seed in (1009, 2027):
    d = r/f'seed{seed}'
    assert json.loads((d/'execution_status.json').read_text())['exit_code'] == 0
    checks = list(csv.DictReader((d/'results.correctness.csv').open()))
    assert checks and all(x['pass'] == '1' for x in checks)
    for x in csv.DictReader((d/'results.summary.csv').open()):
        if x['case'] == 'column_tile64_k16_r4x8_four_batched_atomic':
            lines.append(f"| {seed} | {x['parent_n']} | {float(x['median_ms']):.6f} | {float(x['cublas_full_ms']):.6f} | {float(x['full_speedup']):.4f}x |")
for tool in ('memcheck', 'racecheck'):
    assert json.loads((r/'verification'/f'{tool}.status.json').read_text())['exit_code'] == 0
lines += ['', 'All archived FP64 checks and both sanitizer tools passed.',
          'Hardware, commands, rounds, immutable source SHA256 and binary origins are archived per seed.']
(r/'REPORT.md').write_text('\n'.join(lines)+'\n')
PY
echo 'A800 COLUMN REPLAY COMPLETE'
