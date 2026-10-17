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

final_checks=[]
for phase in ('final_small','final_large'):
    final_checks.extend(csv.DictReader(open(ROOT/phase/'results.correctness.csv')))
assert len(final_checks)==48 and all(r['pass']=='1' for r in final_checks)
assert all(json.loads((ROOT/p/'execution_status.json').read_text())['exit_code']==0
           for p in ('iter1_ablation','endtoend_screen','iter2_endtoend','final_small','final_large'))
max_error=max(float(r['relative_l2_error']) for r in final_checks)
snapshot=ROOT/'iter2_endtoend/source/strassen_cuda'
workspace=ROOT.resolve().parents[1]
matches={p.name:p.read_bytes()==(workspace/'solution/strassen_cuda'/p.name).read_bytes()
         for p in snapshot.glob('*.cuh')}
assert matches and all(matches.values())
(ROOT/'current_source_match.json').write_text(json.dumps(matches,indent=2)+'\n')
verification=json.loads((ROOT/'verification/summary.json').read_text())
best=[]
for n in (2048,8192,20480):
    case,row=min(((case,row) for (size,case),row in final.items()
                  if size==n and case!='cublas_full'),key=lambda v:float(v[1]['median_ms']))
    best.append((n,case,row))
main_rows='\n'.join(f"| {n} | {n//2} | {float(row['cublas_full_ms']):.6f} | `{case}` | {float(row['median_ms']):.6f} | {float(row['full_speedup']):.4f}× |"
                    for n,case,row in best)
plan_rows='\n'.join(f"| {r['parent_n']} | `{r['case']}` | {float(r['median_ms']):.6f} | {float(r['speedup']):.4f}× | {float(r['cv']):.3%} |"
                    for r in recommended)
ablation_rows='\n'.join(f"| {label} | {r['gemm_ms']:.6f} | {r['m7_ms']:.6f} |" for label,r in zip(labels,table))
profile_rows='\n'.join('| '+' | '.join(str(r[k]) for k in profiles[0])+' |' for r in profiles)
stage_rows=[]
for phase in ('iter1_ablation','endtoend_screen','iter2_endtoend','final_small','final_large'):
    timings=list(csv.DictReader(open(ROOT/phase/'results.rounds.csv')))
    correctness=list(csv.DictReader(open(ROOT/phase/'results.correctness.csv')))
    stage_rows.append(f"| `{phase}` | {len(timings)} | {len(correctness)} | PASS |")
source_match=', '.join(matches)
proxy_rows=[]
for n in (2048,8192,20480):
    leaf=float(ablation[n,'cublas_leaf']['median_ms'])
    full=float(ablation[n,'cublas_full']['median_ms'])
    proxy_rows.append(f'| {n} | {full:.6f} | {leaf:.6f} | {7*leaf:.6f} | {full/(7*leaf):.4f}× |')
report=f'''# A100 上的一层 FP32 Strassen：分块对照与端到端验证

已完成 N=2048、叶子 1024 的计算循环、分块、输出更新对照，并接入真正的七乘积
Strassen。大尺寸成本余量可以转化为端到端收益；N=2048 仍慢于同批 cuBLAS。
以下数据来自同一块 NVIDIA A100 80GB PCIe、CUDA 12.2（nvcc 12.2.140）、
cuBLAS 120205。操作均为列主序 FP32 `C <- AB+C`，alpha=beta=1。

**五轮最终复测。** 下表列出本轮最终候选中的最低中位数，属于复测后的描述性比较。
加速比定义为同批 cuBLAS 完整 N×N GEMM 时间除以完整 Strassen 时间。

| N | 叶子尺寸 | cuBLAS ms | 最终候选中最快路径 | Strassen ms | 加速比 |
|---:|---:|---:|---|---:|---:|
{main_rows}

小尺寸排名在信号轮与最终复测之间发生变化。信号轮的新 `64x32` 七流为 1.154094 ms，
最终复测为 {float(final[2048,'pipe64x32_k16_r4_seven_atomic']['median_ms']):.6f} ms；
原版 `huge` 七流最终为 {float(final[2048,'huge_seven_atomic']['median_ms']):.6f} ms。
两者均未超过 cuBLAS，不能把单个 primitive 的约 1.52× 收益写成完整 Strassen 的收益。
本轮最稳妥的已测实现基线是 `Huge/SevenAtomic`；小尺寸的新分块保留为研究候选。

为保留选择过程，以下是迭代 2 提交中、最终复测前已经固定的方案。
该方案在 N=2048 的最终表现并非所有复测候选中的最好值。

| N | 预先选定的路径 | 中位数 ms | 加速比 | 轮次 CV |
|---:|---|---:|---:|---:|
{plan_rows}

![端到端复测](endtoend.png)

**成本余量如何解释。** 以下为迭代 1 同批 cuBLAS 的诊断代理。叶子使用父矩阵
leading dimension=N。`7×leaf` 假设七个乘积都能保持普通叶子 GEMM 的速度，
尚未计入输入和、多个输出更新、调度及并发资源竞争，既不是完整调用时间，也不是硬件下界。

| N | full GEMM ms | leaf GEMM ms | 7×leaf ms | full/(7×leaf) |
|---:|---:|---:|---:|---:|
{chr(10).join(proxy_rows)}

最终完整调用显示：大尺寸的这类余量有一部分兑现；N=2048 的普通叶子代理余量
不足以弥补当前融合 primitive 相对 cuBLAS leaf 的差距。判断依据是前面的完整计时表。

**1024 叶子的受控对照。** 以下全部取自迭代 1 的三轮诊断数据。
`normal add` 保留 beta=1，读取旧 C 后累加。M7 融合两个输入和、一个乘积、两个输出更新。

| 对照 | GEMM primitive ms | M7 primitive ms |
|---|---:|---:|
{ablation_rows}

同计算主体只改输出更新：GEMM 改善约 1.9%，M7 约 0.9%。单独引入双缓冲预取、
增大 BK，没有改善这组实现。主要收益来自较小的线程累加器与 `64x32` CTA 分块：
相对原版 large，GEMM 为 {table[-1]['gemm_vs_original']:.3f}×、M7 为 {table[-1]['m7_vs_original']:.3f}×。
这些数据支持优先改线程分工和网格规模；它们没有证明预取循环在所有布局上无用。

原版 large 在该尺寸只有 256 个 CTA、每块 64 线程，共 512 个 warp。
新候选为 512 个 CTA、每块 128 线程，共 2048 个 warp。Nsight 的实际资源数据如下：

| 路径 | 寄存器/线程 | 理论占用率 % | 实测占用率 % | Waves/SM |
|---|---:|---:|---:|---:|
{profile_rows}

原版低实测占用率同时受网格不足影响，不能全部归因于寄存器数量。
原始 Nsight CSV 还报告原版的全局访问额外 sector 占 54%、共享内存额外 wavefront 占 39%。
新候选对应为 6% 与 3%。这里采集的是单个 GEMM primitive，不代表七流重叠时的指标。
这些是诊断指标；NCU 采集的时长没有用于主表性能计算。

![分块与循环对照](ablation.png)

**大尺寸回归。** `64x32` 候选不能替换所有尺寸。迭代 1 的 N=20480 上，
其 GEMM/M7 为 165.764/200.717 ms，原版 huge 为 119.883/130.526 ms。
迭代 2 的完整七流路径也从 huge 的 882.382 ms 退化到该候选的 1328.033 ms。
小分块增加并行度，同时降低块内数据复用；实测说明必须保留尺寸相关的选择。
在 N=8192，把叶子 4096 从原选择器的 large 改成 huge，完整七流从约 70.860 ms
降到 56.465 ms（信号轮）。原选择器在 N=20480 已使用 huge。

**调度对照。** 全部路径计算同一组七个乘积，没有常规 GEMM 回退。
串行原子、串行普通累加、七流原子、按依赖分阶段普通累加均已测。
额外的四次启动方案依次执行 `(M1,M2)`、`(M3,M4)`、`(M5,M6)`、M7；
每对的有效输出互不重叠，普通累加安全。它在小尺寸未超过七流调度，
N=20480 的 huge 配对路径也仅接近七流。原始负面结果保存在 CSV 中。

**计时与正确性。** CUDA events 覆盖重复的完整调用，包括变换、乘积、输出更新、
跨流等待和调用流汇合。预建流/事件，A/B/C 输入准备在计时前；算法不清零 C。
cuBLAS 使用 `CUBLAS_PEDANTIC_MATH`、禁用原子，环境设置 `NVIDIA_TF32_OVERRIDE=0`。
所有候选先 warmup，再随机化每轮顺序。最终五轮、每批目标 200 ms、至少三次调用。
未手动锁频；必须使用本批 cuBLAS 分母，不能拼接上一轮硬件/时钟状态的数据。

最终常规复测共 {len(final_checks)} 条 FP64 检查记录，覆盖一次及连续三次累加；
最大抽样 relative L2 为 {max_error:.9g}，相对 L2 上限 3e-5、scaled max 上限 1e-4 均未改动。
小尺寸 N<=128 全量验证，其他尺寸请求 128 个含角点的位置并去重；primitive 还查未写象限。
额外检查包括普通累加配对的 N=120/2056（部分 tile、K 尾部）以及 huge 配对 N=256。
memcheck 和共享内存 racecheck 均通过；racecheck 检查共享内存竞争，跨流全局输出安全由
原子更新或显式输出依赖保障，并由完整结果重复累加检查共同验证。
可执行文件中共 {verification['ffma_instructions']} 条静态 FFMA 指令、
{verification['tensor_instructions']} 条 Tensor 指令；本批 cuBLAS 的动态 Tensor 指令数为
{verification['cublas_tensor_instructions']}。

六个仪器命令均以 0 退出。初次汇总脚本把 `HFMA2.MMA` 的 `.MMA` 修饰符误认成
Tensor 指令，因而汇总阶段以 1 退出。已改为解析 SASS 指令基名，并复用原始仪器日志
重新审计通过；性能与数值测试没有重跑或改写。初次失败日志保存在
`verification_console.log`，修正后的审计为 `verification/audit.status.json` 与 `summary.json`。

| 阶段 | 原始计时行 | FP64 校验记录 | 结果 |
|---|---:|---:|---|
{chr(10).join(stage_rows)}

这些校验记录包含对同一确定性输入的重复，不代表独立随机样本数量。
结果覆盖本次 A100 的三个测点；A800、V100、论文全部尺寸曲线和两层混合版本尚未测。
本轮没有定位精确的 crossover，也不把 8/7 的算术量比当作所有有限尺寸的性能上界。

**代码与复现。** 使用说明、操作边界和命令见 [STRASSEN_ONELEVEL.md](../../scripts/STRASSEN_ONELEVEL.md)。
实现为 `solution/strassen_cuda/engine.cuh`，最终复测使用迭代 2 的不可变构建。
当前 {source_match} 与该构建的归档文件逐字节一致，见 [current_source_match.json](current_source_match.json)。
原始头文件、构建命令、源文件 SHA256、GPU UUID、环境、轮次、校验和 telemetry 都已归档。

- [最终全部候选表](final_comparison.csv)、[预先固定方案表](recommended.csv)
- [分块对照表](ablation_comparison.csv)、[编译资源表](compiler_resources.csv)、[Nsight 资源对照](profile_comparison.csv)
- [小尺寸最终轮次](final_small/results.rounds.csv)、[大尺寸最终轮次](final_large/results.rounds.csv)
- [仪器检查结果](verification/summary.json)、[分块图 PDF](ablation.pdf)、[端到端图 PDF](endtoend.pdf)
'''
(ROOT/'REPORT.md').write_text(report)
print(f'REPORT PASS final_checks={len(final_checks)} max_relative_l2={max_error:.9g} current_source_match=True')
