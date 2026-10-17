"""Summarize new ablation/end-to-end experiments without changing raw data."""
import csv
import json
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path


def summarize(prefix):
    meta = json.loads(Path(str(prefix) + '.metadata.json').read_text())
    checks = list(csv.DictReader(open(str(prefix) + '.correctness.csv')))
    assert checks and all(r['pass'] in ('1', 'true') for r in checks), 'correctness failed'
    print(f"CORRECT=True checks={len(checks)} max_relative_l2={max(float(r['relative_l2_error']) for r in checks):.8g}")
    rows = list(csv.DictReader(open(str(prefix) + '.rounds.csv')))
    groups = defaultdict(list)
    for row in rows:
        groups[int(row['parent_n']), row['case']].append(float(row['latency_ms']))
    summaries = []
    for (n, name), times in sorted(groups.items()):
        assert len(times) == meta['rounds'], (n, name, len(times))
        relevant = [r for r in checks if int(r['parent_n']) == n and r['case'] == name]
        assert sorted(int(r['repeats']) for r in relevant) == [1, 3]
        median = stats.median(times)
        full = stats.median(groups[n, 'cublas_full'])
        leaf = stats.median(groups[n, 'cublas_leaf']) if (n, 'cublas_leaf') in groups else 0
        summaries.append(dict(parent_n=n, case=name, median_ms=median,
                              min_ms=min(times), max_ms=max(times),
                              cv=stats.pstdev(times) / stats.mean(times),
                              cublas_full_ms=full, cublas_leaf_ms=leaf,
                              full_speedup=full / median,
                              leaf_speedup=leaf / median if leaf else 0))
    if summaries:
        with open(str(prefix) + '.summary.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    for n in sorted({r['parent_n'] for r in summaries}):
        selected = [r for r in summaries if r['parent_n'] == n]
        print(f"N={n} mode={meta['mode']} cuBLAS={selected[0]['cublas_full_ms']:.6f} ms")
        if meta['mode'] == 'ablation':
            for kind in ('gemm', 'm7'):
                for row in sorted((r for r in selected if f'_{kind}_' in r['case']), key=lambda r: r['median_ms']):
                    print(f"  {row['case']:42s} {row['median_ms']:.6f} ms leaf_ratio={row['leaf_speedup']:.3f} cv={row['cv']:.3%}")
        else:
            for row in sorted((r for r in selected if r['case'] != 'cublas_full'), key=lambda r: r['median_ms']):
                print(f"  {row['case']:48s} {row['median_ms']:.6f} ms speedup={row['full_speedup']:.4f} cv={row['cv']:.3%}")
    return summaries


if __name__ == '__main__':
    summarize(Path(sys.argv[1]))
