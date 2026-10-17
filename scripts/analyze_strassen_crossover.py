"""Audit small-size configurations without replacing the fixed main curve."""
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
runs = {}
inventory = []
for seed in [0, 1009]:
    folder = root / f'seed{seed}'
    assert json.loads((folder / 'execution_status.json').read_text())['exit_code'] == 0
    meta = json.loads((folder / 'results.metadata.json').read_text())
    assert meta['seed_offset'] == seed and meta['rounds'] == 5
    assert meta['target_ms'] == 200 and meta['math_mode'] == 'CUBLAS_PEDANTIC_MATH'
    checks = list(csv.DictReader((folder / 'results.correctness.csv').open()))
    assert checks and all(r['pass'] == '1' for r in checks)
    assert {int(r['repeats']) for r in checks} == {1, 3}
    raw = list(csv.DictReader((folder / 'results.rounds.csv').open()))
    summary = {(int(r['parent_n']), r['case']): r
               for r in csv.DictReader((folder / 'results.summary.csv').open())}
    groups = {}
    for row in raw:
        groups.setdefault((int(row['parent_n']), row['case']), []).append(float(row['latency_ms']))
    for key, values in groups.items():
        assert len(values) == 5
        assert statistics.median(values) == float(summary[key]['median_ms'])
    assert set(groups) == set(summary)
    runs[seed] = summary
    inventory.append(dict(seed=seed, timing_rows=len(raw), fp64_checks=len(checks),
                          max_relative_l2=max(float(r['relative_l2_error']) for r in checks)))

rows = []
sizes = sorted({key[0] for key in runs[0]})
small_unmet = all(
    min(float(row['median_ms']) for (size, case), row in summary.items()
        if size == n and case != 'cublas_full') > float(summary[n, 'cublas_full']['median_ms'])
    for summary in runs.values() for n in [1536, 2048])
for n in sizes:
    # Select only on seed 0; apply exactly that case to independent seed 1009.
    candidates = [case for size, case in runs[0] if size == n and case != 'cublas_full']
    case = min(candidates, key=lambda name: float(runs[0][n, name]['median_ms']))
    for seed, summary in runs.items():
        ref = summary[n, 'cublas_full']
        selected = summary[n, case]
        full, latency = float(ref['median_ms']), float(selected['median_ms'])
        rows.append(dict(parent_n=n, seed=seed, selected_on_seed0=case,
                         cublas_ms=full, candidate_ms=latency, speedup=full/latency,
                         latency_reduction_needed_to_tie=max(0, 1-full/latency),
                         ranges_separated=float(selected['max_ms']) < float(ref['min_ms'])))

with (root / 'selected_candidates.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
lines = ['# A100 一层 Strassen 小尺寸 crossover 诊断', '',
         '论文一层 crossover N=1536 尚未复现。大尺寸混合两层约 1.19× 不替代该项验收。',
         '本次保持 `64e2695` 的算术代码，复用 iter4_hybrid_smoke 的不可变二进制。',
         '完整调用 FP32 C←AB+C，alpha=beta=1，cuBLAS PEDANTIC、TF32 禁用；',
         '五轮随机顺序、每轮目标 200 ms、两组输入；一次与三次累加均作 FP64 校验。', '',
         '扫描全部 11 个现有 backend 的七流调度，以及 Huge、Pipe64x32 的四类 batching。',
         '每个 N 在 seed 0 中选择耗时最低的配置，随后在 seed 1009 中报告同一配置。',
         '这是配置筛查；该独立输入复测未覆盖独立日期/设备，也不是所有可能 kernel 的性能下界。', '',
         '| N | seed 0 选定配置 | seed 0 加速比 | seed 1009 加速比 | seed 1009 cuBLAS ms | seed 1009 一层 ms | 持平还需降时 |',
         '|---:|---|---:|---:|---:|---:|---:|']
for n in sizes:
    a, b = [r for r in rows if r['parent_n'] == n]
    lines.append(f"| {n} | {a['selected_on_seed0']} | {a['speedup']:.4f}× | {b['speedup']:.4f}× | {b['cublas_ms']:.6f} | {b['candidate_ms']:.6f} | {100*b['latency_reduction_needed_to_tie']:.2f}% |")
lines += ['', '## 结论与下一轮目标', '',
          '不同尺寸需要不同分块；固定 Huge 的主曲线不能当作所有分块中的最优曲线。',
          '完整配置结果、每轮耗时、输入校验、环境与 telemetry 均保留在 seed0/、seed1009/。',
          '上述选择只属于本次诊断，不回填或替换此前固定候选的主表。',
          ('本轮 N=1536/2048 的全部现有候选仍落后，下一步需要改变融合 primitive 的执行效率，'
           if small_unmet else 'N=1536/2048 的目标状态以原始候选结果为准，继续按测量优化融合 primitive，'),
          '优先检查四类 operand/output 数量不同的变体在 leaf=768/1024 时的寄存器、',
          'CTA 并发、内存访问和输出原子更新；按测得瓶颈改实现，再验证完整七乘积调用。',
          '现有 N=2048 七流时间线内部空隙约 0.008 ms，不能解释约 0.185 ms 的调用差距；',
          'Graph 暂不作为首要优化方向，也不能仅凭这条时间线断言其完全无益。',
          '验收保留同批 cuBLAS、纯 FP32 和现有误差阈值；目标是在 N=1536 明确超过 cuBLAS，',
          '并同时检查 N=1280/1792/2048 的曲线，防止把局部非单调优势说成稳定 crossover。',
          'V100/CUDA 9.1 与 A100/CUDA 12.2 的差异需单独研究，当前不将其作为差距已解释的结论。', '',
          '## 校验记录', '', '```json', json.dumps(inventory, indent=2), '```']
(root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(rows, indent=2))
