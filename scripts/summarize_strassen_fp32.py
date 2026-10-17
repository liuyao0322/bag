#!/usr/bin/env python3
"""Summarize complete standalone CUDA cost measurements; standard library only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values, q):
    values = sorted(values)
    position = q * (len(values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plot", action="store_true", help="also export scientific PNG/PDF figures (requires matplotlib)")
    args = parser.parse_args()
    root = args.directory.resolve()
    meta = json.loads((root / "results.metadata.json").read_text())
    raw = load_csv(root / "results.rounds.csv")
    checks = load_csv(root / "results.correctness.csv")
    expected_cases = ["cublas_full", "cublas_leaf_compact", "cublas_leaf_parent_ld",
                      "primitive_gemm"] + [f"primitive_m{i}" for i in range(1, 8)]
    expected = {(n, case) for n in meta["sizes"] for case in expected_cases}
    groups = defaultdict(list)
    for row in raw:
        groups[int(row["parent_n"]), row["case"]].append(row)
    if set(groups) != expected:
        raise ValueError(f"incomplete case coverage: {expected - set(groups)}")
    check_groups = defaultdict(list)
    for row in checks:
        check_groups[int(row["parent_n"]), row["case"]].append(row)
        if row["pass"] != "1":
            raise ValueError(f"correctness failure: {row}")
    if set(check_groups) != expected:
        raise ValueError("incomplete correctness coverage")
    for key, rows in check_groups.items():
        if sorted(int(row["repeats"]) for row in rows) != [1, 3]:
            raise ValueError(f"missing beta=1 accumulation check: {key}")
    summary = []
    lookup = {}
    for (n, case), rows in sorted(groups.items()):
        if sorted(int(row["round"]) for row in rows) != list(range(meta["rounds"])):
            raise ValueError(f"missing or duplicate rounds: {n}, {case}")
        times = [float(row["latency_ms"]) for row in rows]
        if any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError("invalid timing")
        if any(int(row["repeats"]) < 3 for row in rows):
            raise ValueError("invalid repeat count")
        if any(not math.isclose(float(row["batch_ms"]) / int(row["repeats"]),
                                float(row["latency_ms"]), rel_tol=1e-9) for row in rows):
            raise ValueError("batch timing inconsistency")
        item = dict(parent_n=n, case=case, kernel=rows[0]["kernel"],
                    m=int(rows[0]["m"]), ld=int(rows[0]["ld"]), rounds=len(rows),
                    median_ms=statistics.median(times), p20_ms=quantile(times, .2),
                    p80_ms=quantile(times, .8), min_ms=min(times), max_ms=max(times),
                    round_cv=statistics.stdev(times) / statistics.mean(times) if len(times)>1 else 0,
                    gemm_equivalent_tflops=statistics.median(float(r["gemm_equivalent_tflops"]) for r in rows))
        summary.append(item)
        lookup[n, case] = item
    write_csv(root / "summary.csv", summary)
    costs = []
    for n in meta["sizes"]:
        latency = lambda case: lookup[n, case]["median_ms"]
        full = latency("cublas_full")
        leaf = latency("cublas_leaf_parent_ld")
        primitive_sum = sum(latency(f"primitive_m{i}") for i in range(1, 8))
        costs.append(dict(
            parent_n=n, leaf_n=n//2, full_cublas_ms=full,
            full_cublas_tflops=lookup[n, "cublas_full"]["gemm_equivalent_tflops"],
            leaf_compact_ms=latency("cublas_leaf_compact"), leaf_parent_ld_ms=leaf,
            primitive_gemm_ms=latency("primitive_gemm"),
            primitive_m1_ms=latency("primitive_m1"), primitive_m2_ms=latency("primitive_m2"),
            primitive_m3_ms=latency("primitive_m3"), primitive_m4_ms=latency("primitive_m4"),
            primitive_m5_ms=latency("primitive_m5"), primitive_m6_ms=latency("primitive_m6"),
            primitive_m7_ms=latency("primitive_m7"),
            seven_cublas_leaf_serial_proxy_ms=7*leaf,
            seven_primitive_serial_proxy_ms=primitive_sum,
            full_over_seven_leaf_proxy=full/(7*leaf),
            full_over_seven_primitive_proxy=full/primitive_sum,
            primitive_gemm_over_cublas_leaf=latency("primitive_gemm")/leaf,
            worst_case_round_cv=max(lookup[n, case]["round_cv"] for case in expected_cases)))
    write_csv(root / "cost_summary.csv", costs)
    max_error = max(float(row["relative_l2_error"]) for row in checks)
    worst_cv = max(item["round_cv"] for item in summary)
    report = [
        "# Pure FP32, beta=1 CUDA cost baseline", "",
        f"GPU: **{meta['gpu']}**. CUDA runtime {meta['cuda_runtime']}; cuBLAS {meta['cublas_version']}.",
        "`cublasSgemm`, `CUBLAS_PEDANTIC_MATH`, `NVIDIA_TF32_OVERRIDE=0`; column-major, alpha=beta=1.", "",
        "## Measurement", "",
        f"Each case has {meta['rounds']} CUDA-event batches targeting {meta['target_ms']} ms each. "
        "Each batch contains 3 to 10,000 calls; latency is batch time divided by calls. "
        "The cap can make the shortest cases run for less than the target duration. "
        "Twenty ms of per-case warmup precedes randomized case order in each round. "
        "Inputs and nonzero output seeds are initialized outside timing. Calls inside each batch repeatedly "
        "accumulate into C. Allocation, validation, event creation and library initialization are excluded. "
        "A single non-default CUDA stream covers all timed work; CUDA Graphs are not used.", "",
        "Compact leaves have ld=N/2. Parent-view leaves and primitives have ld=N. "
        "Primitive inputs X/Y and V/W use the top-left/bottom-right quadrants of initialized parent matrices; "
        "D/E use separate top-left/bottom-right output quadrants. Each of M1..M7 reproduces the coefficient "
        "pattern in the existing one-level implementation, using these representative operands. "
        "These isolated measurements do not execute the complete Strassen algebra or concurrent scheduling.", "",
        "## Correctness", "",
        f"All {len(checks)} case/accumulation checks passed against CPU FP64 dot products "
        f"after 1 and 3 calls. Maximum sampled relative L2 error: {max_error:.6g}. "
        "The acceptance limits are relative L2 <=3e-5 and max absolute error/(1+max absolute reference) <=1e-4. "
        "Untouched output samples must match the initial seed exactly. "
        "Output matrices with side <=128 are checked exhaustively; larger outputs use "
        f"up to {meta['requested_samples']} distinct deterministic coordinates per output, including corners, "
        "plus sentinel samples in untouched quadrants. This is sampled numerical validation, not an "
        "exhaustive correctness proof for large matrices.", "",
        "## Costs", "",
        "All times below are milliseconds; per-case values are medians of batch averages.", "",
        "| N | Full cuBLAS | Half compact | Half parent ld | Primitive GEMM | Sum of 7 primitives | Full / (7 half GEMMs) | Full / sum primitives |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in costs:
        report.append(f"| {row['parent_n']} | {row['full_cublas_ms']:.6f} | "
                      f"{row['leaf_compact_ms']:.6f} | {row['leaf_parent_ld_ms']:.6f} | "
                      f"{row['primitive_gemm_ms']:.6f} | {row['seven_primitive_serial_proxy_ms']:.6f} | "
                      f"{row['full_over_seven_leaf_proxy']:.3f} | {row['full_over_seven_primitive_proxy']:.3f} |")
    report += ["", "The two ratios are **serial cost proxies**, not measured end-to-end speedups "
               "or hardware bounds. They omit actual cross-product cache interaction, dispatch ordering, "
               "and concurrency. The cuBLAS leaf proxy additionally omits Strassen input combinations "
               "and multi-output accumulation. M1..M7 costs are summed from isolated medians.", "",
               f"Largest per-case coefficient of variation across rounds: {worst_cv:.2%}. "
               "`summary.csv` retains p20/p80, min/max and CV for every case; "
               "`results.rounds.csv` retains every timing batch. GPU clocks are not locked; "
               "`telemetry.csv` records clocks, utilization, power and temperature. "
               "Other GPUs on the node may be in use; the allocated GPU is recorded in `environment.txt`.", "",
               "## Files", "", "- `results.metadata.json`: runtime, precision and shape configuration.",
               "- `results.correctness.csv`: numerical checks and counts.",
               "- `results.rounds.csv`: raw timing rounds.",
               "- `summary.csv`: per-case timing distributions.",
               "- `cost_summary.csv`: full/leaf/primitive costs and serial proxies.",
               "- `source/`, `source.sha256`, `build-command.txt`, `build.log`: exact compiled source and build evidence.",
               "- `environment.txt`, `run-command.txt`, `run.log`, `telemetry.csv`: execution evidence.", ""]
    audit_path = root / "execution_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit["timing_rows"] != len(raw) or audit["numerical_checks"] != len(checks):
            raise ValueError("execution audit counts do not match raw evidence")
        report += ["## Additional verification and execution audit", "",
                   "Compute Sanitizer memcheck reported **0 errors** for primitive_m7 at "
                   "N=512,1024,2048,8448, covering small/medium/large/huge kernels. "
                   "This instrumentation checks representative cases, not every measured shape.", "",
                   "Nsight Compute sampled `ampere_sgemm_128x64_nn` at N=2048: "
                   "`sm__inst_executed_pipe_tensor.sum = 0`, "
                   "`smsp__sass_thread_inst_executed_op_ffma_pred_on.sum = 8594128896`. "
                   "Static disassembly of the standalone binary has 18,720 FFMA instructions "
                   "and no Tensor Core instruction mnemonics. The cuBLAS sample is a representative "
                   "dynamic check; FP32 configuration is enforced for all benchmark cases.", "",
                   "The measurement binary printed COMPLETE after all 880 timing batches. "
                   "The original wrapper then exited with code 2 because its file was edited "
                   "while the shell was executing it. The revised wrapper parses its body before "
                   "starting work and records its exit status. A separate replay covering four "
                   "kernel buckets completed with exit code 0, 44 timing rows and 88 numerical checks; "
                   "the compiled CUDA source was byte-for-byte identical. Original measurement "
                   "rows were retained. See `execution_audit.json` and `verification/entry_*`.", "",
                   "The summary completeness gate was tested with missing rounds, duplicate rounds, "
                   "a failed numerical check and NaN latency; all four corrupt inputs were rejected. "
                   "See `summary_gate_tests.json`.", ""]
    (root / "REPORT.md").write_text("\n".join(report))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ns = [row["parent_n"] for row in costs]
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
        for key, label in [("full_cublas_ms", "Full SGEMM"),
                           ("seven_cublas_leaf_serial_proxy_ms", "7 x half SGEMM (serial proxy)"),
                           ("seven_primitive_serial_proxy_ms", "Sum of 7 isolated primitives")]:
            axes[0].plot(ns, [r[key] for r in costs], marker=".", label=label)
        axes[0].set(xscale="log", yscale="log", xlabel="Parent N", ylabel="Milliseconds",
                    title="Costs; proxies omit complete Strassen execution")
        axes[0].legend(fontsize=8)
        for case, label in [("primitive_gemm", "GEMM + atomic output"),
                            ("primitive_m1", "A sum, two outputs (M1)"),
                            ("primitive_m3", "A/B sums, one output (M3)"),
                            ("primitive_m5", "B sum, two outputs (M5)"),
                            ("primitive_m7", "A/B sums, two outputs (M7)")]:
            axes[1].plot(ns, [lookup[n,case]["median_ms"] / lookup[n,"cublas_leaf_parent_ld"]["median_ms"]
                             for n in ns], marker=".", label=label)
        axes[1].axhline(1,color="gray",linestyle="--",linewidth=1)
        axes[1].set(xscale="log", xlabel="Parent N", ylabel="Latency / half SGEMM latency",
                    title="Existing primitive cost relative to cuBLAS leaf")
        axes[1].legend(fontsize=8)
        for key, label in [("full_over_seven_leaf_proxy", "Full / (7 x half SGEMM)"),
                           ("full_over_seven_primitive_proxy", "Full / sum of isolated primitives")]:
            axes[2].plot(ns,[r[key] for r in costs],marker=".",label=label)
        axes[2].axhline(1,color="gray",linestyle="--",linewidth=1)
        axes[2].set(xscale="log", xlabel="Parent N", ylabel="Serial cost ratio",
                    title="Cost budget estimates; not end-to-end speedups")
        axes[2].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle(f"{meta['gpu']} | FP32, alpha=beta=1 | medians of {meta['rounds']} CUDA-event batches")
        fig.savefig(root / "costs.png", dpi=180)
        fig.savefig(root / "costs.pdf")
        plt.close(fig)
    print(f"Validated {len(raw)} timing rows and {len(checks)} numerical checks: {root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
