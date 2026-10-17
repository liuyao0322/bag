"""Build comparison tables and standalone figures from archived measurements."""
import csv
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(sys.argv[1])


def read(folder):
    return {(int(r['parent_n']), r['case']): r for r in
            csv.DictReader(open(ROOT / folder / 'results.summary.csv'))}


def write(name, rows):
    with open(ROOT / name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


ablation = read('iter1_ablation')
order = ['large_gemm_atomic', 'large_gemm_normal', 'sync64_k8_r8_gemm_atomic',
         'pipe64_k8_r8_gemm_atomic', 'pipe64_k16_r8_gemm_atomic',
         'pipe64_k16_r4_gemm_atomic', 'pipe64x32_k16_r4_gemm_atomic']
labels = ['Original large / atomic', 'Same body / normal add', 'Sync / 64x64 / K8 / r8',
          'Pipeline / 64x64 / K8 / r8', 'Pipeline / 64x64 / K16 / r8',
          'Pipeline / 64x64 / K16 / r4', 'Pipeline / 64x32 / K16 / r4']
table = []
for case in order:
    gemm = float(ablation[2048, case]['median_ms'])
    m7 = float(ablation[2048, case.replace('_gemm_', '_m7_')]['median_ms'])
    table.append(dict(case=case, gemm_ms=gemm, m7_ms=m7,
                      gemm_vs_original=float(ablation[2048, order[0]]['median_ms']) / gemm,
                      m7_vs_original=float(ablation[2048, 'large_m7_atomic']['median_ms']) / m7))
write('ablation_comparison.csv', table)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, key, title in zip(axes, ['gemm_ms', 'm7_ms'], ['GEMM primitive', 'M7: two input sums, two output updates']):
    values = [r[key] for r in table]
    ax.barh(labels, values, color=['#666666', '#999999', '#a1c1dc', '#73a2cb', '#578bb8', '#346f9d', '#d58435'])
    ax.set_xlabel('Median latency (ms; lower is better)')
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.18)
    for i, value in enumerate(values):
        ax.text(value + .004, i, f'{value:.4f}', va='center', fontsize=9)
axes[0].invert_yaxis()
fig.suptitle('A100 FP32 ablation: parent N=2048, leaf=1024, beta=1 (3 rounds)')
fig.tight_layout()
fig.savefig(ROOT / 'ablation.png', dpi=180)
fig.savefig(ROOT / 'ablation.pdf')
plt.close(fig)

small, large = read('final_small'), read('final_large')
final = {**small, **large}
rows = [dict(parent_n=n, **{k:v for k,v in row.items() if k!='parent_n'}) for (n,_),row in sorted(final.items())]
write('final_comparison.csv', rows)
recommended = []
for n in (2048, 8192, 20480):
    case = 'pipe64x32_k16_r4_seven_atomic' if n==2048 else 'huge_seven_atomic'
    row = final[n, case]
    recommended.append(dict(parent_n=n, case=case, median_ms=row['median_ms'],
                            cublas_ms=row['cublas_full_ms'], speedup=row['full_speedup'],
                            cv=row['cv'], min_ms=row['min_ms'], max_ms=row['max_ms']))
write('recommended.csv', recommended)
fig, ax = plt.subplots(figsize=(8.5, 4.7))
styles = [('legacy_serial_atomic', 'Original selector / serial', '#888888'),
          ('legacy_seven_atomic', 'Original selector / 7 streams', '#3978ab'),
          ('huge_seven_atomic', '128x128 original huge / 7 streams', '#b77835')]
for case, label, color in styles:
    ax.plot(range(3), [float(final[n, case]['full_speedup']) for n in (2048,8192,20480)],
            marker='o', label=label, color=color)
ax.plot(range(3), [float(r['speedup']) for r in recommended], marker='s',
        label='Plan selected before final replay', color='#227a55', linestyle='--')
ax.axhline(1, color='black', linewidth=.8)
ax.set_xticks(range(3), ['2048', '8192', '20480'])
ax.set_ylim(0, 1.2)
ax.set_xlabel('Parent N')
ax.set_ylabel('cuBLAS latency / complete Strassen latency')
ax.set_title('A100 FP32: actual seven-product execution, beta=1 (5 rounds)')
ax.grid(axis='y', alpha=.2)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT/'endtoend.png', dpi=180)
fig.savefig(ROOT/'endtoend.pdf')
plt.close(fig)

registers=[]
name = None
for line in (ROOT/'iter2_endtoend/build.log').read_text().splitlines():
    match=re.search(r"Function properties for (\S+)",line)
    if match:
        name=match[1]
    used=re.search(r'Used (\d+) registers, (\d+) bytes smem',line)
    if used and name:
        registers.append(dict(mangled=name,registers=int(used[1]),shared_bytes=int(used[2])))
demangled=subprocess.run(['c++filt'],input='\n'.join(r['mangled'] for r in registers),
                          text=True,capture_output=True,check=True).stdout.splitlines()
assert len(demangled)==len(registers)
for row, name in zip(registers, demangled):
    row['function']=name
write('compiler_resources.csv', registers)

profiles=[]
for name, path in [('original_large',ROOT/'profile_baseline.csv'),
                   ('pipe64x32_k16_r4',ROOT/'verification/profile_small.csv')]:
    content=path.read_text()
    metrics={r.get('Metric Name'):r.get('Metric Value') for r in csv.DictReader(io.StringIO(content[content.index('"ID"'):]))}
    profiles.append(dict(case=name,**{key:metrics[key] for key in
        ('Registers Per Thread','Theoretical Occupancy','Achieved Occupancy','Waves Per SM')}))
write('profile_comparison.csv',profiles)
print(json.dumps({'recommended':recommended,'profiles':profiles},indent=2))
