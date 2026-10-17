"""Analyze archived complete-call measurements; profiler times are diagnostic."""
import csv
import io
import json
import re
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]).resolve()
PROJECT = Path(__file__).resolve().parents[1]
CASES = ['huge_seven_atomic', 'huge_four_batched_atomic',
         'huge_hybrid2_seven_atomic', 'huge_hybrid2_four_batched_atomic']
LABELS = ['One level / 7 streams', 'One level / 4 batched streams',
          'Hybrid 2 levels / 7 streams', 'Hybrid 2 levels / 4 batched streams']


def write_csv(name, rows):
    if not rows:
        return
    with (ROOT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def union_ns(intervals):
    total = 0
    end = None
    for lo, hi in sorted(intervals):
        if end is None or lo > end:
            total += hi - lo
        elif hi > end:
            total += hi - end
        end = max(end or 0, hi)
    return total


def timeline(path):
    name = path.stem
    meta = json.loads(path.with_suffix('.metadata.json').read_text())
    calls = meta['profile_calls']
    assert calls > 0
    db = sqlite3.connect(path)
    columns = {r[1] for r in db.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
    assert {'start', 'end', 'demangledName'} <= columns, columns
    rows = db.execute('select k.start,k.end,s.value,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k '
                      'join StringIds s on k.demangledName=s.id order by k.start').fetchall()
    assert rows
    intervals = [(r[0], r[1]) for r in rows]
    busy = union_ns(intervals)
    envelope = max(r[1] for r in rows) - min(r[0] for r in rows)
    groups = {}
    for start, end, kernel, stream in rows:
        category = 'prepare' if 'hybrid_prepare' in kernel else 'merge' if 'hybrid_merge' in kernel else 'products'
        groups.setdefault(category, []).append((start, end))
    output = dict(profile=name, calls=calls, kernels=len(rows), streams=len({r[3] for r in rows}),
                  kernel_envelope_ms_per_call=envelope/calls/1e6,
                  gpu_busy_union_ms_per_call=busy/calls/1e6,
                  internal_gpu_idle_ms_per_call=(envelope-busy)/calls/1e6,
                  summed_kernel_ms_per_call=sum(hi-lo for lo,hi in intervals)/calls/1e6)
    for category in ['prepare', 'products', 'merge']:
        output[category+'_busy_ms_per_call'] = union_ns(groups.get(category, []))/calls/1e6
    expected = 63 if 'hybrid' in name else 7 if name.endswith('huge_seven_atomic') else 4 if 'batched' in name else 1
    assert len(rows) == calls*expected, (name, len(rows), calls*expected)
    db.close()
    return output


runs = {}
inventory = []
curve_rows = []
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir() or not (folder / 'results.correctness.csv').exists():
        continue
    status_file = folder / 'execution_status.json'
    if not status_file.exists():
        continue  # An allocation may still be producing a new replay directory.
    meta = json.loads((folder / 'results.metadata.json').read_text())
    checks = list(csv.DictReader((folder / 'results.correctness.csv').open()))
    raw = list(csv.DictReader((folder / 'results.rounds.csv').open()))
    assert json.loads(status_file.read_text())['exit_code'] == 0, folder
    assert checks and all(r['pass'] == '1' for r in checks), folder
    inventory.append(dict(run=folder.name, gpu=meta['gpu'], seed_offset=meta.get('seed_offset', 0),
                          timing_rows=len(raw), fp64_checks=len(checks),
                          max_relative_l2=max(float(r['relative_l2_error']) for r in checks)))
    if not raw:
        continue
    summary = {(int(r['parent_n']), r['case']): r for r in csv.DictReader((folder/'results.summary.csv').open())}
    runs[folder.name] = summary
    groups = {}
    for r in raw:
        groups.setdefault((int(r['parent_n']), r['case']), []).append(float(r['latency_ms']))
    for key, values in groups.items():
        assert len(values) == meta['rounds']
        assert statistics.median(values) == float(summary[key]['median_ms'])
    if folder.name.startswith(('a100_curve', 'a100_replay', 'a800_curve', 'a800_replay', 'a100_crossover')):
        for n in sorted({key[0] for key in summary}):
            full = float(summary[n, 'cublas_full']['median_ms'])
            for case in CASES:
                if (n, case) not in summary:
                    continue
                r = summary[n, case]
                row = dict(run=folder.name, seed_offset=meta.get('seed_offset', 0), parent_n=n, case=case,
                           median_ms=float(r['median_ms']), cublas_ms=full,
                           speedup=full/float(r['median_ms']), cv=float(r['cv']),
                           scratch_bytes=3*n*n if 'hybrid2' in case else 0)
                curve_rows.append(row)
write_csv('run_inventory.csv', inventory)
write_csv('curves.csv', curve_rows)

comparisons = []
for run, summary in runs.items():
    for n in sorted({key[0] for key in summary}):
        if not all((n, case) in summary for case in CASES):
            continue
        vals = [float(summary[n, case]['median_ms']) for case in CASES]
        full = float(summary[n, 'cublas_full']['median_ms'])
        comparisons.append(dict(run=run, parent_n=n, cublas_ms=full, one_seven_ms=vals[0],
                                one_batched_ms=vals[1], hybrid_seven_ms=vals[2], hybrid_batched_ms=vals[3],
                                one_speedup=full/vals[0], hybrid_speedup=full/vals[2],
                                hybrid_latency_reduction=1-vals[2]/vals[0],
                                one_batch_latency_reduction=1-vals[1]/vals[0],
                                hybrid_batch_latency_reduction=1-vals[3]/vals[2]))
write_csv('algorithm_comparison.csv', comparisons)

# Dense points are descriptive observations, not a guarantee about unmeasured N.
crossover_rows = []
for run in ['a100_crossover_seed0', 'a100_crossover_seed1009']:
    if run not in runs:
        continue
    summary = runs[run]
    for n in sorted({key[0] for key in summary}):
        ref = summary[n, 'cublas_full']
        for case in CASES:
            if (n, case) not in summary:
                continue
            row = summary[n, case]
            crossover_rows.append(dict(run=run, parent_n=n, case=case,
                                       median_speedup=float(ref['median_ms'])/float(row['median_ms']),
                                       candidate_max_ms=float(row['max_ms']), cublas_min_ms=float(ref['min_ms']),
                                       observed_ranges_separated=float(row['max_ms']) < float(ref['min_ms'])))
write_csv('crossover.csv', crossover_rows)

plot_runs = [r for r in ['a100_curve', 'a100_replay_seed1009', 'a800_curve', 'a800_replay_seed1009'] if r in runs]
if plot_runs:
    fig, axes = plt.subplots(len(plot_runs), 1, figsize=(10, 4*len(plot_runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], plot_runs):
        for case, label in zip(CASES, LABELS):
            points = sorted([r for r in curve_rows if r['run'] == run and r['case'] == case], key=lambda r:r['parent_n'])
            ax.plot([r['parent_n'] for r in points], [r['speedup'] for r in points], marker='o', markersize=4, label=label)
        ax.axhline(1, color='black', lw=.8)
        ax.set_xscale('log', base=2)
        ax.set_xticks([512,1024,2048,4096,8192,16384,20480], ['512','1024','2048','4096','8192','16384','20480'])
        ax.set_ylim(0, 1.35)
        ax.set_xlabel('Parent N')
        ax.set_ylabel('Same-run cuBLAS / complete Strassen')
        ax.set_title(run + ': FP32, C <- AB+C, 5 rounds')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(ROOT / ('curves.'+ext), dpi=180)
    plt.close(fig)

verification = ROOT / 'verification'
audit = {}
if verification.exists():
    if (verification/'summary.json').exists():
        audit = json.loads((verification/'summary.json').read_text())
    for status in verification.glob('*.status.json'):
        assert json.loads(status.read_text())['exit_code'] == 0, status
    for log in verification.glob('*check*.log'):
        text = log.read_text()
        assert 'ERROR SUMMARY: 0 errors' in text or 'RACECHECK SUMMARY: 0 hazards' in text, log
        checks_path = log.with_suffix('.correctness.csv')
        if checks_path.exists():
            rows = list(csv.DictReader(checks_path.open()))
            assert rows and all(r['pass'] == '1' for r in rows), checks_path
            audit[log.stem+'_fp64_checks'] = len(rows)
    if (verification/'sass.txt').exists():
        pattern = r'^\s*/\*[0-9a-f]+\*/\s+(?:@!?[A-Z]+\d+\s+)?([A-Z][A-Z0-9]*(?:\.[A-Z0-9_]+)*)\b'
        opcodes = Counter(op.split('.')[0] for op in re.findall(pattern, (verification/'sass.txt').read_text(), re.M))
        audit['ffma_instructions'] = opcodes['FFMA']
        audit['tensor_instructions'] = sum(opcodes[k] for k in ['HMMA','IMMA','DMMA','BMMA','MMA','WGMMA','HGMMA'])
        assert audit['ffma_instructions'] > 0 and audit['tensor_instructions'] == 0
    if (verification/'tensor_cublas.csv').exists():
        content = (verification/'tensor_cublas.csv').read_text()
        metrics = list(csv.DictReader(io.StringIO(content[content.index('"ID"'):])))
        values = [float(r['Metric Value'].replace(',', '')) for r in metrics if r.get('Metric Name') == 'sm__inst_executed_pipe_tensor.sum']
        assert values and all(v == 0 for v in values)
        audit['cublas_tensor_instructions'] = values
    profiles = [timeline(path) for path in sorted(verification.glob('timeline_*.sqlite'))]
    if profiles:
        write_csv('timeline_comparison.csv', profiles)
        audit['timeline_profiles'] = len(profiles)
    elif (ROOT/'timeline_comparison.csv').exists():
        # Keep the saved extraction when large local profiler binaries are absent.
        audit['timeline_profiles'] = len(list(csv.DictReader((ROOT/'timeline_comparison.csv').open())))
        profiles = list(csv.DictReader((ROOT/'timeline_comparison.csv').open()))
    if profiles:
        small = [r for r in profiles if r['profile'] != 'timeline_hybrid_large']
        big = [r for r in profiles if r['profile'] == 'timeline_hybrid_large']
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        names = [r['profile'].replace('timeline_', '').replace('_atomic', '') for r in small]
        busy = [float(r['gpu_busy_union_ms_per_call']) for r in small]
        idle = [float(r['internal_gpu_idle_ms_per_call']) for r in small]
        axes[0].barh(names, busy, label='GPU busy (kernel interval union)')
        axes[0].barh(names, idle, left=busy, label='Internal GPU idle')
        axes[0].set_title('N=2048: diagnostic kernel envelope')
        axes[0].set_xlabel('ms per complete call (20 captured calls)')
        axes[0].legend(fontsize=7)
        if big:
            r = big[0]
            left = 0
            for key, label in [('prepare_busy_ms_per_call','Input transforms + scratch reset'),
                               ('products_busy_ms_per_call','49 fused products'),
                               ('merge_busy_ms_per_call','Outer output merge'),
                               ('internal_gpu_idle_ms_per_call','Internal GPU idle')]:
                val = float(r[key])
                axes[1].barh(['Hybrid 2 levels'], [val], left=[left], label=f'{label}: {val:.3f} ms')
                left += val
            axes[1].set_title('N=20480: diagnostic hybrid phase costs')
            axes[1].set_xlabel('ms per complete call (3 captured calls)')
            axes[1].legend(fontsize=7)
        fig.suptitle('Nsight Systems diagnostics; excluded from primary speedups')
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(ROOT/('timeline_diagnostics.'+ext), dpi=180)
        plt.close(fig)
    (verification/'summary.json').write_text(json.dumps(audit, indent=2)+'\n')

origin = ROOT / 'iter4_hybrid_smoke/source/strassen_cuda'
matches = {path.name: path.read_bytes() == (PROJECT/'solution/strassen_cuda'/path.name).read_bytes() for path in origin.glob('*.cuh')}
assert matches and all(matches.values())
(ROOT/'current_source_match.json').write_text(json.dumps(matches, indent=2)+'\n')

lines = ['# A100/A800 FP32 Strassen：论文 batching 与混合两层', '',
         '固定实现：`64e2695` 的算术代码；列主序 FP32 `C <- AB+C`，alpha=beta=1。',
         '主表为完整调用的 CUDA event 计时；所有变换、中间输出清零、乘积、合并和跨流依赖均计入。',
         '仅分配、输入生成、流/事件创建排除。禁用 TF32，cuBLAS PEDANTIC，误差阈值不变。', '',
         '## 固定候选与独立复测', '',
         '固定比较 Huge 七流、四类 batching，以及相应的混合两层。下表使用七流作为共同基线，',
         '不根据复测后的最低值重新选候选。四类 batching 的全部结果见 algorithm_comparison.csv。', '',
         '| 运行 | N | cuBLAS ms | 一层 ms | 两层 ms | 一层加速比 | 两层加速比 | 两层相对一层耗时下降 |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in comparisons:
    if r['run'] not in plot_runs or r['parent_n'] not in [2048,4096,6144,8192,10240,12288,16384,20480]:
        continue
    lines.append(f"| {r['run']} | {r['parent_n']} | {r['cublas_ms']:.6f} | {r['one_seven_ms']:.6f} | {r['hybrid_seven_ms']:.6f} | {r['one_speedup']:.4f}× | {r['hybrid_speedup']:.4f}× | {100*r['hybrid_latency_reduction']:.2f}% |")
lines += ['', '![完整性能曲线](curves.png)', '', '## 正确性与可复现性', '',
          '| 运行 | 输入 seed offset | 计时行 | FP64 校验记录 | 最大 relative L2 |',
          '|---|---:|---:|---:|---:|']
for r in inventory:
    lines.append(f"| {r['run']} | {r['seed_offset']} | {r['timing_rows']} | {r['fp64_checks']} | {r['max_relative_l2']:.8g} |")
lines += ['', '每个 size/case 检查一次与连续三次累加；N<=128 全量检查，其他尺寸检查含角点的 128 个请求位置。',
          '同一 seed 下的重复记录不代表独立输入。不同 seed offset 的输入及 FP64 参考同步变化。',
          '五轮随机顺序，目标 200 ms，至少三次调用。GPU 不手动锁频，分母只取同批 cuBLAS。',
          '每次构建和复测保留源文件 SHA256、命令、环境、GPU UUID、原始轮次、校验及 telemetry。',
          '当前 CUDA 头文件与固定构建逐字节一致，见 current_source_match.json。', '',
          '## 实现与显存', '',
          '论文四类 batching 使用三个双乘积 batch 和一个单乘积，由四条流并发执行。',
          '原配对对照则按阶段执行；两者不是同一调度。输出使用原子加，调用流等待全部 worker。',
          '混合两层将外层变换写入紧凑矩阵，七个外层产品依次调用融合一层，共 49 个四分之一乘积。',
          '复现的是论文的混合分层结构；外层变换和工作区调度由本仓库实现，不宣称与论文参考代码逐行一致。',
          '准备 kernel 同时清零每个新的中间输出，merge 在调用流执行，保留原始 C 和重复调用语义。',
          '复用三个半尺寸 scratch 矩阵，额外显存为 `3*N*N` 字节；N=20480 时约 1.172 GiB，',
          '不含原始 A/B/C。各算法阶段的诊断计时见 timeline_comparison.csv。', '',
          '## 分析边界', '',
          'Nsight Systems/Compute 的时长不用于主表。时间线的 busy 是 kernel 区间并集，',
          'internal idle 只覆盖首个 kernel 到最后一个 kernel 的内部空隙，不等同于全部调用开销。',
          '小尺寸调度收益、非单调 crossover 和不同 GPU 的结论分别报告；不以 8/7 或 64/49 算术量比作性能硬上界。',
          'A800 只有完整成功的 CSV 出现后才计入结果；排队状态和自动执行日志另存，未运行不代表通过。', '',
          '复现入口：`scripts/STRASSEN_BATCHING.md`。']
if crossover_rows:
    lines += ['', '## Crossover 加密测点', '',
              '下表给出每个输入种子下的中位数加速比；四类 batching 及观测耗时范围见 crossover.csv。',
              '范围分离表示本次五轮中 candidate 最慢值仍低于 cuBLAS 最快值，不是统计置信区间。',
              '这些结果只覆盖已测尺寸，不假设尺寸增大时性能单调。', '',
              '| N | 一层 seed 0 | 一层 seed 1009 | 两层 seed 0 | 两层 seed 1009 |',
              '|---:|---:|---:|---:|---:|']
    lookup = {(r['parent_n'], r['run'], r['case']): r for r in crossover_rows}
    for n in sorted({r['parent_n'] for r in crossover_rows}):
        cells = []
        for case in [CASES[0], CASES[2]]:
            for seed in [0, 1009]:
                row = lookup.get((n, f'a100_crossover_seed{seed}', case))
                cells.append(f"{row['median_speedup']:.4f}×" if row else '—')
        lines.append(f"| {n} | " + ' | '.join(cells) + ' |')
if audit:
    lines += ['', '## 仪器验证', '',
              '逐命令退出状态、输入校验、工具日志和源文件验证位于 verification/。',
              '共享内存 racecheck 不替代跨流全局输出的原子/依赖约束。', '',
              '```json', json.dumps(audit, indent=2), '```']
    if (ROOT/'timeline_comparison.csv').exists():
        extracted = list(csv.DictReader((ROOT/'timeline_comparison.csv').open()))
        lines += ['', '| Profile | kernel 数 | 每调用 GPU busy ms | 每调用内部 idle ms |',
                  '|---|---:|---:|---:|']
        for r in extracted:
            lines.append(f"| {r['profile']} | {r['kernels']} | {float(r['gpu_busy_union_ms_per_call']):.6f} | {float(r['internal_gpu_idle_ms_per_call']):.6f} |")
        lines += ['', '![时间线诊断](timeline_diagnostics.png)']
    if (verification/'tensor_cublas_initial/tensor_cublas.status.json').exists():
        lines += ['', '## 工具异常与恢复', '',
                  '首个验证流程在 NCU Tensor 计数阶段以 9 退出：无法写入其他用户拥有的 /tmp/nsight-compute-lock。',
                  '此前四个 sanitizer 命令和五个时间线采集均成功；全部性能/常规数值数据未重跑或改写。',
                  '确认节点没有其他 NCU 采集后，按 [NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq)',
                  '使用用户可写的独立 TMPDIR，仅重跑 Tensor 审计并补做 SASS 导出；两项均以 0 退出，汇总分析通过。',
                  '原失败证据在 verification/tensor_cublas_initial/ 和 a100_followup.initial.status.json，修复命令另存。']
selected = [r for r in comparisons if r['run'] in ['a100_curve','a100_replay_seed1009'] and r['parent_n']==20480]
if len(selected)==2:
    lines += ['', '## 结果解读与选择', '',
              f"N=20480 的两组输入中，七流混合两层相对同批 cuBLAS 为 {selected[0]['hybrid_speedup']:.4f}× / {selected[1]['hybrid_speedup']:.4f}×，",
              f"相对对应一层的耗时下降为 {100*selected[0]['hybrid_latency_reduction']:.2f}% / {100*selected[1]['hybrid_latency_reduction']:.2f}%。",
              'N=10240、12288、16384、20480 的两层收益均超过本轮 3% 投入门槛，可作为后续大尺寸工作基线。',
              '在较小尺寸应继续区分两层相对 cuBLAS 的收益与两层相对一层的额外收益；工作区开销也需纳入选择。',
              'N=2048 的现有候选仍未超过 cuBLAS。四类 batching 没有在所有尺寸稳定优于七流，因此保留显式选择。',
              '本轮不新增自动 GEMM 回退，不以 post-hoc 最低值替换预先固定的七流对照。',
              '一层在 N=4096 附近持平，N=4352 的两组加密测点中位数开始超过 cuBLAS；N=5632 仍接近持平。',
              '两层在加密测点 N=6144 附近持平，N=6656 的两组中位数明确超过 cuBLAS。这里只描述测点，不保证未测尺寸。',
              'N=2048 七流时间线内部 GPU 空隙约 0.008 ms，小于与 cuBLAS 约 0.185 ms 的完整调用差距；',
              '目前没有足够证据优先投入 Graph。时间线并非完整调用的性能下界，也不证明 Graph 在其他场景无用。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'runs': len(inventory), 'timing_rows': sum(r['timing_rows'] for r in inventory),
                  'fp64_checks': sum(r['fp64_checks'] for r in inventory), 'verification': audit}, indent=2))
