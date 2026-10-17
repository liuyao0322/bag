"""Plot fixed-candidate measurements; bounds are observed ranges, not CIs."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1]).resolve()
case = 'column_tile64_k16_r4x8_four_batched_atomic'
series = [
    ('selected_curve_seed2027', 'Fixed candidate / dense curve, seed 2027', '#334155', 'o'),
    ('holdout_batched_seed1009', 'Independent replay, seed 1009', '#d97706', '^'),
    ('holdout_batched_seed2027', 'Independent replay, seed 2027', '#2563eb', 's'),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for run, label, color, marker in series:
    rows = list(csv.DictReader((root/run/'results.summary.csv').open()))
    full = {int(r['parent_n']): r for r in rows if r['case'] == 'cublas_full'}
    selected = sorted((r for r in rows if r['case'] == case), key=lambda r: int(r['parent_n']))
    ns = [int(r['parent_n']) for r in selected]
    mid = [float(r['full_speedup']) for r in selected]
    lo = [float(full[n]['min_ms'])/float(r['max_ms']) for n, r in zip(ns, selected)]
    hi = [float(full[n]['max_ms'])/float(r['min_ms']) for n, r in zip(ns, selected)]
    for ax in axes:
        ax.errorbar(ns, mid, yerr=[[m-l for m,l in zip(mid,lo)], [h-m for h,m in zip(hi,mid)]],
                    color=color, marker=marker, markersize=4, linewidth=1, capsize=2,
                    label=label, linestyle='-' if 'curve' in run else 'none')
for ax in axes:
    ax.axhline(1, color='#dc2626', linestyle='--', linewidth=1)
    ax.axvline(1536, color='#94a3b8', linestyle=':', linewidth=1)
    ax.set_xlabel('Parent square dimension N')
    ax.set_ylabel('Speedup = same-run cuBLAS / full Strassen call')
    ax.grid(alpha=.18)
axes[0].set(xlim=(450, 2110), ylim=(.5, 1.115), title='All measured small sizes')
axes[1].set(xlim=(1200, 2110), ylim=(.98, 1.10), title='Crossover region (expanded scale)')
axes[0].legend(loc='lower right', fontsize=8)
fig.suptitle('A100 80GB PCIe: pure FP32, C = AB + C; all seven products and joins timed')
fig.supxlabel('Bars: conservative bounds from observed min/max rounds; not confidence intervals', fontsize=9)
for suffix in ('svg', 'png'):
    fig.savefig(root/f'selected_crossover.{suffix}', dpi=180)
plt.close(fig)
