"""Summarize complete CUDA BAG calls without discarding slower configurations."""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
project = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path, records):
    if records:
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


inventory, complete = [], []
for folder in sorted(root.iterdir()):
    if not folder.is_dir():
        continue
    status_path = folder / 'execution_status.json'
    if not status_path.exists():
        if (folder / 'source.sha256').exists():
            inventory.append(dict(run=folder.name, status='incomplete before device execution'))
        continue
    status = json.loads(status_path.read_text())
    assert status['exit_code'] == 0, (folder, status)
    checks = read_rows(folder / 'results.correctness.csv')
    assert checks and all(r['pass'] == '1' and float(r['relative_l2_error']) <= 3e-5
                          and float(r['max_scaled_error']) <= 1e-4 for r in checks), folder
    meta = json.loads((folder / 'results.metadata.json').read_text())
    assert meta['alpha'] == meta['beta'] == 1 and meta['math_mode'] == 'CUBLAS_PEDANTIC_MATH'
    inventory.append(dict(run=folder.name, status='passed', seed_offset=meta['seed_offset'],
                          fp64_records=len(checks),
                          max_relative_l2=max(float(r['relative_l2_error']) for r in checks),
                          max_scaled_error=max(float(r['max_scaled_error']) for r in checks)))
    raw = read_rows(folder / 'results.rounds.csv')
    summary = []
    for n in sorted({int(r['parent_n']) for r in raw}):
        lookup = {}
        for case in sorted({r['case'] for r in raw if int(r['parent_n']) == n}):
            points = [r for r in raw if int(r['parent_n']) == n and r['case'] == case]
            assert len(points) == meta['rounds']
            assert {int(r['round']) for r in points} == set(range(meta['rounds']))
            times = [float(r['latency_ms']) for r in points]
            lookup[case] = dict(parent_n=n, case=case, median_ms=statistics.median(times),
                                min_ms=min(times), max_ms=max(times), rounds=len(times))
        blas = lookup['cublas_full']['median_ms']
        strassen = next(r['median_ms'] for name, r in lookup.items() if name.startswith('strassen_'))
        for case, row in lookup.items():
            row['speedup'] = blas / row['median_ms']
            summary.append(row)
            complete.append(dict(run=folder.name, seed_offset=meta['seed_offset'], **row,
                                 same_run_cublas_ms=blas, same_run_strassen_ms=strassen))
    write_rows(folder / 'results.summary.csv', summary)
write_rows(root / 'complete_calls.csv', complete)
(root / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')

selection = json.loads((root / 'selection.json').read_text())
origin = root / selection['source_run']
assert hashlib.sha256((origin / 'benchmark_bag_cuda').read_bytes()).hexdigest() == selection['binary_sha256']
assert (origin / 'source/bag_cuda/bag.cuh').read_bytes() == (project / 'solution/bag_cuda/bag.cuh').read_bytes()
for path in (origin / 'source/strassen_cuda').glob('*.cuh'):
    assert path.read_bytes() == (project / 'solution/strassen_cuda' / path.name).read_bytes()
    tuned = project / 'benchmark_results/strassen_primitive_20261017/iter16_column_resources/source/strassen_cuda' / path.name
    assert path.read_bytes() == tuned.read_bytes(), 'tuned primitive changed'
for line in (origin / 'source.sha256').read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    assert hashlib.sha256((origin / name).read_bytes()).hexdigest() == digest
selected = [r for r in complete if r['run'].startswith('holdout_seed') and r['case'] == selection['case']]
assert len(selected) == 8 and {r['seed_offset'] for r in selected} == {1009, 2027}
for seed in (1009, 2027):
    folder = root / f'holdout_seed{seed}'
    assert selection['binary_sha256'] in (folder / 'build-origin.txt').read_text()
    checks = read_rows(root / f'recursive_seed{seed}' / 'results.correctness.csv')
    assert all(r['pass'] == '1' for r in checks)
    for n in (8, 24, 32, 64, 120, 128, 256):
        for depth in (1, 2, 3):
            size, legal = n, True
            for _ in range(depth):
                legal = legal and size > 0 and size % 8 == 0
                size //= 2
            if not legal:
                continue
            for basis in ('minimal', 'full'):
                for schedule in ('serial', 'seven', 'batched'):
                    name = f'bag_column_{basis}_d{depth}_{schedule}'
                    relevant = [r for r in checks if int(r['parent_n']) == n and r['case'] == name]
                    assert {int(r['repeats']) for r in relevant} == {1, 3}, (seed, n, name)
                    if n <= 128:
                        assert all(int(r['checked_values']) == n*n for r in relevant)

sanitizers = {}
for tool in ('memcheck', 'racecheck'):
    folder = root / 'verification'
    sanitizers[tool] = json.loads((folder / f'{tool}.status.json').read_text())
    assert sanitizers[tool]['exit_code'] == 0
    checks = read_rows(folder / f'{tool}.correctness.csv')
    assert checks and all(r['pass'] == '1' for r in checks)
    log = (folder / f'{tool}.log').read_text()
    assert ('ERROR SUMMARY: 0 errors' if tool == 'memcheck'
            else '0 hazards displayed (0 errors, 0 warnings)') in log
assert json.loads((root / 'validation.status.json').read_text())['exit_code'] == 0

profile = None
profile_dir = root / 'profile'
if (profile_dir / 'status.json').exists():
    import io
    profile_status = json.loads((profile_dir / 'status.json').read_text())
    profile = dict(exit_code=profile_status['exit_code'])
    if profile_status['exit_code'] == 0:
        checks = read_rows(profile_dir / 'results.correctness.csv')
        assert checks and all(r['pass'] == '1' for r in checks)
        content = (profile_dir / 'primitive.csv').read_text()
        data = list(csv.DictReader(io.StringIO(content[content.index('"ID"'):])))
        data = [r for r in data if r.get('ID', '').isdigit()]
        assert len({r['ID'] for r in data}) == 1
        assert all('simt_primitive' in r['Kernel Name'] for r in data)
        values = {r['Metric Name']: r['Metric Value'].replace(',', '') for r in data if r.get('Metric Name')}
        for name in ('Registers Per Thread', 'Block Size', 'Grid Size', 'Static Shared Memory Per Block',
                     'Theoretical Occupancy', 'Block Limit Registers',
                     'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
                     'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
                     'l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum'):
            profile[name] = float(values[name])

proof = dict(implementation_verified=True, scope='A100, pure FP32 C<-AB+C',
             source_matches_archive=True, primitive_body_unchanged=True,
             recursion_depths_verified=[1, 2, 3],
             independent_input_offsets=[1009, 2027],
             ordinary_fp64_records=sum(r.get('fp64_records', 0) for r in inventory),
             sanitizers=sanitizers, isolated_primitive_profile=profile, selected_complete_calls=selected,
             n1536_faster_than_cublas=all(r['speedup'] > 1 for r in selected if r['parent_n'] == 1536))
(root / 'verification.json').write_text(json.dumps(proof, indent=2) + '\n')

lines = ['# CUDA alternative-basis matrix multiplication', '',
         'Implemented BAG.md using the previously tuned CUDA primitive, unchanged.',
         'All seven products, forward/inverse transforms and recursive levels are included.',
         'Operation: square column-major FP32 C <- AB+C; same-run cuBLAS PEDANTIC, TF32 disabled.', '',
         '## Frozen configuration and independent replay', '',
         f"Selected before independent replay: `{selection['case']}`; see selection.json.",
         'Each independent input uses seven shuffled rounds, target300ms per round.',
         'These are complete calls including transforms, scratch resets, products, decoding and stream joins.',
         'Allocation, library/stream creation and input generation are outside timing.', '',
         '| Seed | N | BAG ms | cuBLAS ms | Strassen ms | cuBLAS/BAG |',
         '|---:|---:|---:|---:|---:|---:|']
for r in selected:
    lines.append(f"| {r['seed_offset']} | {r['parent_n']} | {r['median_ms']:.6f} | {r['same_run_cublas_ms']:.6f} | {r['same_run_strassen_ms']:.6f} | {r['speedup']:.4f}x |")
lines += ['', 'N=1536 BAG does not exceed same-run cuBLAS in this implementation.',
          'The earlier Strassen N=1536 acceptance does not imply the BAG transforms are free.',
          'No ordinary-GEMM replacement, reduced tolerance or omitted slower layout/schedule is used.', '',
          '## Correctness and recursion', '',
          'Depth1/2/3, both basis layouts and all three schedules passed two independent input sets.',
          'Sizes include8/24/32/64/120/128/256; incompatible recursion shapes are explicitly skipped.',
          'FP64 reference: exhaustive at N<=128, otherwise deterministic128 positions including corners.',
          'Checks cover nonzero initial C and one/three consecutive accumulated calls.',
          'Unchanged limits: relative L2<=3e-5, scaled maximum error<=1e-4.',
          f"Ordinary-run FP64 records: {proof['ordinary_fp64_records']}; all pass.",
          'Memcheck: N24/128/2056, depth1/2 where legal, both layouts/all schedules, zero errors.',
          'Racecheck: N24/128, depth1/2 where legal, both layouts/all schedules, zero shared hazards.',
          'Global output race safety also relies on atomics for overlapping worker updates and caller ordering.', '',
          '## Storage and schedules', '',
          'Minimal mode materializes SA=A12-A21+A22 and SB=B12-B21+B22; all remaining sums are fused.',
          'SA/SB share an N x (N/2) buffer with leading dimension N; independent transformed output T is zeroed.',
          'One-level scratch:6*N*N bytes minimal,12*N*N full. Workspaces are preallocated and not reentrant.',
          'Minimal batched uses6 product launches; full batched uses4. Each also includes prepare/decode.',
          'Serial, seven-stream and both basis variants remain available. All negative results are retained.', '',
          '## Recursive complete-call performance', '',
          '| N | Configuration | BAG ms | Same-run cuBLAS ms | Speedup |',
          '|---:|---|---:|---:|---:|']
for r in complete:
    if r['run'] == 'recursive_performance' and r['case'].startswith('bag_'):
        lines.append(f"| {r['parent_n']} | {r['case']} | {r['median_ms']:.6f} | {r['same_run_cublas_ms']:.6f} | {r['speedup']:.4f}x |")
lines += ['', '## Large-size regression, all configurations', '',
          '| N | Configuration | BAG ms | Same-run cuBLAS ms | Speedup |',
          '|---:|---|---:|---:|---:|']
for r in complete:
    if r['run'].startswith('large_') and r['case'].startswith('bag_'):
        lines.append(f"| {r['parent_n']} | {r['case']} | {r['median_ms']:.6f} | {r['same_run_cublas_ms']:.6f} | {r['speedup']:.4f}x |")
lines += ['', '## Evidence and initial runner issue', '',
          'Source archives/hashes, exact build/run commands, environment, raw rounds, workspace sizes and FP64 errors are preserved per run.',
          'iter17_initial failed before compilation because a non-login shell lacked the module function.',
          'iter18_initial completed all checks and rounds with program status0, while its login-shell srun returned1.',
          'Runners now initialize /etc/profile.d/modules.sh explicitly in a normal shell; independent suite status is0.',
          'The frozen arithmetic binary was not changed during replay. See verification.json and inventory.json.',
          'A100 evidence does not establish A800 performance. Earlier queued Strassen A800 jobs remain separate.']
if profile and profile['exit_code'] == 0:
    lines += ['', '## Isolated reused primitive profile', '',
              'One serial M1 primitive at leaf768 was profiled with NCU kernel replay.',
              'This is an isolated resource/memory diagnostic, not concurrent occupancy or complete-call timing.',
              f"Registers/thread: {profile['Registers Per Thread']:g}; threads/CTA: {profile['Block Size']:g}; CTA grid: {profile['Grid Size']:g}.",
              f"Static shared bytes: {profile['Static Shared Memory Per Block']:g}; theoretical occupancy: {profile['Theoretical Occupancy']:g}%.",
              f"Shared load/store bank conflicts: {profile['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum']:g}/{profile['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum']:g}.",
              'The profiled kernel is the reused strassen_cuda::simt_primitive; raw counters and exact command are in profile/.']
(root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({k:v for k,v in proof.items() if k != 'selected_complete_calls'}, indent=2))
