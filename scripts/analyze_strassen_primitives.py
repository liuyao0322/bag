"""Summarize immutable profiling/complete-call evidence for the small crossover."""
import csv
import io
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()


def write_csv(name, rows):
    if rows:
        with (root / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


counters = []
for folder in sorted(root.glob('profile_*')):
    if folder.name == 'profile_baseline':
        continue  # Prefer the extended collection of the same baseline.
    for n in [1536, 2048]:
        status = folder / f'n{n}.status.json'
        if not status.exists():
            continue
        assert json.loads(status.read_text())['exit_code'] == 0, status
        content = (folder / f'n{n}.csv').read_text()
        raw = list(csv.DictReader(io.StringIO(content[content.index('"ID"'):])))
        raw = [r for r in raw if r.get('ID', '').isdigit()]
        contract_path = folder/'profile_contract.json'
        contract = json.loads(contract_path.read_text()) if contract_path.exists() else dict(launches=7)
        count = contract['launches']
        assert {r['ID'] for r in raw} == set(map(str, range(count)))
        representatives = [(0,0),(1,3),(2,4),(3,6)] if count == 4 else [(0,0),(3,3),(4,4),(6,6)]
        for launch_id, product in representatives:
            rs = [r for r in raw if r['ID'] == str(launch_id)]
            values = {r['Metric Name']: r['Metric Value'].replace(',', '') for r in rs if r.get('Metric Name')}
            def metric(name):
                return float(values[name]) if values.get(name) else None
            variant = {0: 'Var3: A2 B1 C2', 3: 'Var1: A2 B2 C1',
                       4: 'Var2: A1 B2 C2', 6: 'Var0: A2 B2 C2'}[product]
            counters.append(dict(profile=folder.name, parent_n=n, leaf=n//2, variant=variant,
                                 product=product, launch_id=launch_id, batched_pair=count==4 and launch_id<3, kernel=rs[0]['Kernel Name'],
                                 registers_per_thread=metric('Registers Per Thread'),
                                 threads_per_block=metric('Block Size'), grid_ctas=metric('Grid Size'),
                                 static_smem_bytes=metric('Static Shared Memory Per Block'),
                                 resident_blocks_register_limit=metric('Block Limit Registers'),
                                 resident_blocks_smem_limit=metric('Block Limit Shared Mem'),
                                 theoretical_occupancy_pct=metric('Theoretical Occupancy'),
                                 isolated_achieved_occupancy_pct=metric('Achieved Occupancy'),
                                 eligible_warps_per_scheduler=metric('Eligible Warps Per Scheduler'),
                                 l1tex_throughput_pct=metric('L1/TEX Cache Throughput'),
                                 dram_throughput_pct=metric('DRAM Throughput'),
                                 shared_load_bank_conflicts=metric('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'),
                                 shared_store_bank_conflicts=metric('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'),
                                 global_ld_requests=metric('l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'),
                                 global_ld_sectors=metric('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'),
                                 global_reduction_sectors=metric('l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum'),
                                 diagnostic_duration_ns=metric('Duration')))
write_csv('primitive_counters.csv', counters)

control = root / 'variant_costs/results.summary.csv'
if control.exists():
    lookup = {(int(r['parent_n']), r['case']): r for r in csv.DictReader(control.open())}
    controls = []
    for (n, case), row in lookup.items():
        if not case.endswith('_atomic'):
            continue
        normal = lookup[n, case.removesuffix('_atomic') + '_normal']
        a, b = float(row['median_ms']), float(normal['median_ms'])
        controls.append(dict(parent_n=n, case=case, atomic_ms=a, ordinary_add_ms=b,
                             ordinary_add_latency_reduction=1-b/a))
    write_csv('atomic_controls.csv', controls)

complete = []
inventory = []
for folder in sorted(root.iterdir()):
    status = folder / 'execution_status.json'
    if not folder.is_dir() or not status.exists():
        continue
    assert json.loads(status.read_text())['exit_code'] == 0, status
    checks = list(csv.DictReader((folder/'results.correctness.csv').open()))
    assert checks and all(r['pass'] == '1' for r in checks), folder
    inventory.append(dict(run=folder.name, fp64_checks=len(checks),
                          max_relative_l2=max(float(r['relative_l2_error']) for r in checks)))
    meta = json.loads((folder/'results.metadata.json').read_text())
    if meta['mode'] != 'endtoend' or not (folder/'results.summary.csv').exists():
        continue
    summary = list(csv.DictReader((folder/'results.summary.csv').open()))
    for r in summary:
        if r['case'] == 'cublas_full':
            continue
        full = next(x for x in summary if x['parent_n'] == r['parent_n'] and x['case'] == 'cublas_full')
        complete.append(dict(run=folder.name, parent_n=int(r['parent_n']), case=r['case'],
                             seed_offset=meta['seed_offset'], candidate_ms=float(r['median_ms']),
                             cublas_ms=float(full['median_ms']),
                             speedup=float(full['median_ms'])/float(r['median_ms']),
                             ranges_separated=float(r['max_ms']) < float(full['min_ms'])))
write_csv('complete_calls.csv', complete)
(root/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')

concurrent = []
for status in sorted((root/'range_profiles').glob('*.status.json')):
    assert json.loads(status.read_text())['exit_code'] == 0, status
    stem = status.name.removesuffix('.status.json')
    n, case = re.fullmatch(r'n(\d+)_(.*)', stem).groups()
    content = (status.parent/(stem+'.csv')).read_text()
    metrics = list(csv.DictReader(io.StringIO(content[content.index('"ID"'):])))
    values = {r['Metric Name']:float(r['Metric Value'].replace(',', ''))
              for r in metrics if r.get('Metric Name')}
    meta = json.loads((status.parent/(stem+'.metadata.json')).read_text())
    checks = list(csv.DictReader((status.parent/(stem+'.correctness.csv')).open()))
    assert meta['profile_calls'] == 10 and all(r['pass'] == '1' for r in checks)
    concurrent.append(dict(parent_n=int(n), case=case, profile_calls=10,
                           active_warp_occupancy_pct=values['sm__warps_active.avg.pct_of_peak_sustained_active'],
                           sm_throughput_pct=values['sm__throughput.avg.pct_of_peak_sustained_elapsed'],
                           l1tex_throughput_pct=values['l1tex__throughput.avg.pct_of_peak_sustained_elapsed'],
                           l1_sector_hit_rate_pct=values['l1tex__t_sector_hit_rate.pct'],
                           l2_sector_hit_rate_pct=values['lts__t_sector_hit_rate.pct'],
                           ctas_launched=values['sm__ctas_launched.sum']))
write_csv('concurrent_range_counters.csv', concurrent)

stalls = []
for status in sorted([*(root/'stall_profiles').glob('*.status.json'), *(root/'stall_profiles_selected').glob('*.status.json')]):
    assert json.loads(status.read_text())['exit_code'] == 0, status
    stem = status.name.removesuffix('.status.json')
    n, case = re.fullmatch(r'n(\d+)_(.*)', stem).groups()
    content = (status.parent/(stem+'.csv')).read_text()
    raw = list(csv.DictReader(io.StringIO(content[content.index('"ID"'):])))
    values = {r['Metric Name']:float(r['Metric Value'].replace(',', ''))
              for r in raw if r.get('Metric Name')}
    meta = json.loads((status.parent/(stem+'.metadata.json')).read_text())
    checks = list(csv.DictReader((status.parent/(stem+'.correctness.csv')).open()))
    assert meta['profile_calls'] == 10 and checks and all(r['pass'] == '1' for r in checks)
    row = dict(profile=status.parent.name, parent_n=int(n), case=case, profile_calls=10)
    for name in ['short_scoreboard','long_scoreboard','mio_throttle','math_pipe_throttle','barrier','wait','not_selected','lg_throttle']:
        row[name+'_pct'] = values['smsp__warp_issue_stalled_'+name+'_per_warp_active.pct']
    stalls.append(row)
write_csv('concurrent_stalls.csv', stalls)

lines = ['# A100 小尺寸融合 Strassen primitive 优化', '',
         '验收目标仍是一层完整调用在 N=1536 超过同批纯 FP32 cuBLAS；未达标不得以大尺寸收益替代。',
         '操作 C←AB+C，alpha=beta=1，cuBLAS PEDANTIC、禁用 TF32、现有误差阈值保持不变。',
         '主计时覆盖全部七乘积、融合输入和/输出更新及跨流依赖；排除分配、输入生成与流创建。', '',
         '## 四类 primitive 的测量', '',
         '详细寄存器、CTA、共享内存、理论与隔离 occupancy、加载/原子 sector 在 primitive_counters.csv。',
         'NCU kernel replay 会串行重放 kernel，这些 occupancy 和时长只用于解释隔离 primitive，',
         '不能当成并发时的占用率或主性能结论。Var1/Var2/Var3 各代表两乘积，Var0 代表 M7。',
         'four_batched_atomic 的前三个 launch 各覆盖两乘积，grid/sector 为该对产品的合计；',
         'CSV 的 batched_pair 和 launch_id 标明采集单位，不能与单产品计数直接作倍数结论。', '',
         '| 采集 | leaf | 变体 | 寄存器/线程 | CTA | 共享写 bank conflicts | reduction sectors |',
         '|---|---:|---|---:|---:|---:|---:|']
for r in counters:
    lines.append(f"| {r['profile']} | {r['leaf']} | {r['variant']} | {r['registers_per_thread']:g} | {r['grid_ctas']:g} | {r['shared_store_bank_conflicts']:g} | {r['global_reduction_sectors']:g} |")
lines += ['', '隔离 atomic/ordinary-add 受控计时见 atomic_controls.csv；仅切换同一计算主体的输出更新。',
          '普通加的隔离优势较小，不能直接推导七流并发时原子竞争为零。', '',
          '## 完整七流并发的计数器', '',
          '使用 NCU range replay，10 次完整调用为一个范围；保持并发，不按 kernel 串行重放。',
          'cache-control none、clock-control none 与普通测量保持相同策略；profiler 时长不作为主计时。',
          'SM throughput 是该指标定义的 SM 吞吐率，并非单独的 FFMA 利用率，也不是性能下界。', '',
          '| N | 配置 | active warp occupancy | SM throughput | L1 sector hit rate | L2 sector hit rate |',
          '|---:|---|---:|---:|---:|---:|']
for r in concurrent:
    lines.append(f"| {r['parent_n']} | {r['case']} | {r['active_warp_occupancy_pct']:.2f}% | {r['sm_throughput_pct']:.2f}% | {r['l1_sector_hit_rate_pct']:.2f}% | {r['l2_sector_hit_rate_pct']:.2f}% |")
lines += ['', '并发 occupancy 高于 cuBLAS 并未带来更短调用；不以提高 occupancy 本身作为优化验收。', '',
          '## 完整七乘积调用', '',
          '每个候选保留同批 cuBLAS，五轮随机顺序、每轮目标 200 ms。下表不作跨运行绝对时长比较。',
          '全部负结果见 complete_calls.csv；仅此表描述每轮 N=1536 的候选中位数最佳值，属于搜索结果，',
          '不能代替预先固定候选的独立输入复测。', '',
          '| 运行 | N=1536 最快配置 | 一层 ms | 同批 cuBLAS ms | 加速比 |',
          '|---|---|---:|---:|---:|']
for run in sorted({r['run'] for r in complete}):
    points = [r for r in complete if r['run'] == run and r['parent_n'] == 1536]
    if points:
        r = min(points, key=lambda x:x['candidate_ms'])
        lines.append(f"| {run} | {r['case']} | {r['candidate_ms']:.6f} | {r['cublas_ms']:.6f} | {r['speedup']:.4f}× |")
if stalls:
    lines += ['', '## 并发完整调用中的 warp 停顿', '',
              'NCU range replay、每范围 10 次完整调用。停顿比例以活跃 warp 周期为分母；',
              '不同配置的比例不能直接换算为可节省的总调用时间。', '',
              '| N | 配置 | short scoreboard | long scoreboard | MIO throttle | barrier |',
              '|---:|---|---:|---:|---:|---:|']
    for r in stalls:
        lines.append(f"| {r['parent_n']} | {r['case']} | {r['short_scoreboard_pct']:.2f}% | {r['long_scoreboard_pct']:.2f}% | {r['mio_throttle_pct']:.2f}% | {r['barrier_pct']:.2f}% |")
    lines += ['', '由此把 shared 到寄存器的预取和依赖隐藏作为后续实验方向；比例差并不证明物理下界。',
              '初次采集的 SASS FFMA 指标不支持 range replay，错误保留在 stall_profiles_initial/；',
              '移除不支持项后六个范围采集成功，主性能计时未修改。']

selected = 'deferred_exact_swizzle_coalesced64_k16_r4_seven_atomic'
holdout = [r for r in complete if r['run'] == 'holdout_seed1009' and r['case'] == selected]
if holdout:
    lines += ['', '## 第 9 轮冻结候选的独立复测：该候选未达标', '',
              '选择来自 iteration 9，第二组输入使用同一配置；并非在第二组输入重新择优。',
              '候选为 `'+selected+'`。', '',
              '| N | 一层 ms | 同批 cuBLAS ms | 加速比 | 持平还需降时 |',
              '|---:|---:|---:|---:|---:|']
    for r in holdout:
        lines.append(f"| {r['parent_n']} | {r['candidate_ms']:.6f} | {r['cublas_ms']:.6f} | {r['speedup']:.4f}× | {100*max(0,1-r['speedup']):.2f}% |")
column_holdout = [r for r in complete if r['run'] == 'holdout_column_seed1009' and r['case'] == 'column_tile64_k16_r4x8_seven_atomic']
if column_holdout:
    lines += ['', '## 4×8 列映射的冻结独立输入复测', '',
              '依据 iteration 15 搜索结果固定 column_tile64_k16_r4x8；',
              '独立输入 seed offset=1009，选择原因和二进制 SHA256 见 selection_column.json。', '',
              '| N | 一层 ms | 同批 cuBLAS ms | 加速比 |',
              '|---:|---:|---:|---:|']
    for r in column_holdout:
        lines.append(f"| {r['parent_n']} | {r['candidate_ms']:.6f} | {r['cublas_ms']:.6f} | {r['speedup']:.4f}× |")

final_case = 'column_tile64_k16_r4x8_four_batched_atomic'
final_holdouts = [r for r in complete if r['run'] in ('holdout_batched_seed1009','holdout_batched_seed2027') and r['case'] == final_case]
if final_holdouts:
    proof = [r for r in final_holdouts if r['parent_n'] == 1536]
    sanitizer = {tool: json.loads((root/'verification_selected'/f'{tool}.status.json').read_text())['exit_code']
                 for tool in ('memcheck','racecheck')}
    origin = root/'iter16_column_resources'
    project = Path(__file__).resolve().parents[1]
    source_matches = all(p.read_bytes() == (project/'solution/strassen_cuda'/p.name).read_bytes()
                         for p in (origin/'source/strassen_cuda').glob('*.cuh'))
    profile_complete = all(json.loads((root/'profile_selected_batched'/f'n{n}.status.json').read_text())['exit_code'] == 0
                           for n in (1536,2048))
    passed = (len(proof)==2 and all(r['speedup']>1 and r['ranges_separated'] for r in proof)
              and all(code==0 for code in sanitizer.values()) and source_matches and profile_complete)
    acceptance = dict(acceptance_met=passed, scope='A100 80GB PCIe, FP32 C<-AB+C, N1536 complete one-level Strassen versus same-run cuBLAS',
                      selected_case=final_case, n1536_independent_replays=proof,
                      source_matches_selected_archive=source_matches, selected_four_variant_profiles_complete=profile_complete,
                      sanitizer_exit_codes=sanitizer, a800_job='108136; report status separately')
    (root/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    intro = ['**当前验收：A100 上 N=1536 已超过同批 cuBLAS。**' if passed else '**当前验收仍未通过；查看 acceptance.json。**', '',
             '最终固定配置为 `column_tile64_k16_r4x8 / four_batched_atomic`，选择与 SHA256 见 selection_batched_column.json。',
             '两组新输入各七轮、每轮目标 300 ms；两组 N=1536 的 candidate 最慢轮仍快于 cuBLAS 最快轮。',
             '此结论针对本机 A100 / CUDA12.2 / cuBLAS12.2.5 的相对性能；原论文 V100 SXM2 / CUDA9.1 的绝对时长未作数值复刻。', '',
             '| 独立输入 offset | N | Strassen ms | 同批 cuBLAS ms | 加速比 |',
             '|---:|---:|---:|---:|---:|']
    for r in final_holdouts:
        intro.append(f"| {r['seed_offset']} | {r['parent_n']} | {r['candidate_ms']:.6f} | {r['cublas_ms']:.6f} | {r['speedup']:.4f}× |")
    intro += ['', 'memcheck：N=120/128/1536/2056；racecheck：N=120/128；均零错误/零 hazard。',
              '部分 tile 回退保留融合七乘积，不以普通 GEMM 替代。A800 固定候选验证作业为 108136，排队状态不视为实测。', '']
    lines[2:2] = intro
    curve = [r for r in complete if r['run']=='selected_curve_seed2027' and r['case']==final_case]
    if curve:
        lines += ['', '## 冻结候选的加密尺寸曲线', '',
                  '![固定候选的 crossover 曲线](selected_crossover.png)', '',
                  'offset2027，七轮×200ms。首个胜出的采样点为 N=1280，前一点 N=1152 仍落后；',
                  '两点之间尚未采样；不据此声称精确转折位置、所有整数尺寸或所有更大尺寸均胜出。', '',
                  '| N | Strassen ms | 同批 cuBLAS ms | 加速比 |',
                  '|---:|---:|---:|---:|']
        for r in curve:
            lines.append(f"| {r['parent_n']} | {r['candidate_ms']:.6f} | {r['cublas_ms']:.6f} | {r['speedup']:.4f}× |")
    large = [r for r in complete if r['run']=='selected_large_regression']
    if large:
        lines += ['', '## 大尺寸回归：小尺寸配置不能覆盖全部尺寸', '',
                  '保留全部七流/四类 batching 结果，不把 column 配置的退化隐藏为自动普通 GEMM 回退。', '',
                  '| N | 配置 | Strassen ms | 同批 cuBLAS ms | 加速比 |',
                  '|---:|---|---:|---:|---:|']
        for r in large:
            lines.append(f"| {r['parent_n']} | {r['case']} | {r['candidate_ms']:.4f} | {r['cublas_ms']:.4f} | {r['speedup']:.4f}× |")
        lines += ['', '小尺寸使用 column、较大尺寸保留 Huge/已有混合两层为不同固定配置；',
                  '该小尺寸优化不替代旧的大尺寸混合两层报告，也不据此宣称整个尺寸域有单调 crossover。']

lines += ['', '## 实现方向与证据边界', '',
          '- XOR swizzle 消除测得的共享 B 写 bank 冲突，并保留 float4 对齐；寄存器代价逐类报告。',
          '- 输出整理复用 shared packing 存储，不增加全局中间矩阵；保留全部原子加和两次阶段间同步。',
          '- 精确分块路径去除无用的边界分支，仅在 leaf%64==0 时启用；部分 tile 仍走 masked 融合路径。',
          '- 寄存器 8×4 与 K=8/32 的退化保留，不能由更大线程累加器推断完整调用更快。',
          '- 预取输入和的时序优化需要实际 SASS/计数器与完整调用验证，不以源码中存在双缓冲作为成功证据。', '',
          'memcheck 覆盖 N=120/128/1536/2056，含精确分块与 masked 回退；共享 racecheck 覆盖 N=120/128。',
          '验证对象和源版本以 verification/ 的来源记录为准，不能自动扩展到之后新增的实验内核。',
          '初次 range replay 因旧版 NCU 不支持 profile-from-start 参数而失败，保留在 range_profiles_initial/；',
          '去除该参数、按默认 Profiler API 范围重试后六次采集成功；未改写主性能数据。', '',
          '资料：[NVIDIA CUTLASS software pipelining](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)，',
          '[Nsight Compute kernel/range replay 的差别](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-replay)。',
          '环境、二进制来源、归档源码 SHA256、命令、每轮原始值、FP64 校验和 telemetry 均在各运行目录。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(profiles=len(counters), complete_call_cases=len(complete), inventory=inventory), indent=2))
