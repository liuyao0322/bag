#!/usr/bin/env python3
"""Benchmark one-level Triton BAG against ordinary cuBLAS GEMM."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import triton


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_ROOT = Path(
    os.environ.get("BAG_TRITON_IMPL_ROOT", REPOSITORY_ROOT / "src")
).resolve()
sys.path.insert(0, str(IMPLEMENTATION_ROOT))

from bag_triton import allocate_bag_workspace, bag_matmul  # noqa: E402


DEFAULT_SIZES = (512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True)
class Result:
    gpu: str
    schedule: str
    precision: str
    layout: str
    n: int
    bag_p20_ms: float
    bag_median_ms: float
    bag_p80_ms: float
    gemm_p20_ms: float
    gemm_median_ms: float
    gemm_p80_ms: float
    speedup: float
    relative_l2_error: float
    max_absolute_error: float
    bag_classical_equivalent_tflops: float
    bag_executed_gemm_tflops: float
    gemm_tflops: float


def _matrix(size: int, layout: str, *, random: bool = True) -> torch.Tensor:
    if layout == "row-major":
        tensor = torch.empty((size, size), device="cuda", dtype=torch.float32)
    elif layout == "column-major":
        tensor = torch.empty_strided(
            (size, size), (1, size), device="cuda", dtype=torch.float32
        )
    else:
        raise ValueError(f"unsupported layout: {layout}")
    return tensor.uniform_(-0.5, 0.5) if random else tensor


def _benchmark(fn: Callable[[], None], warmup_ms: int, rep_ms: int) -> tuple[float, float, float]:
    p20, median, p80 = triton.testing.do_bench(
        fn, warmup=warmup_ms, rep=rep_ms, quantiles=(0.2, 0.5, 0.8)
    )
    return float(p20), float(median), float(p80)


def _run_one(
    size: int,
    layout: str,
    precision: str,
    warmup_ms: int,
    rep_ms: int,
    schedule: str,
) -> Result:
    allow_tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    a = _matrix(size, layout)
    b = _matrix(size, layout)
    bag_output = _matrix(size, layout, random=False)
    gemm_output = _matrix(size, layout, random=False)
    workspace = allocate_bag_workspace(
        a,
        b,
        bag_output,
        multistream=schedule in ("multistream", "seven-stream"),
        full_concurrency=schedule == "seven-stream",
        hybrid=schedule == "hybrid",
        grouped=schedule == "grouped" or (schedule == "auto" and size <= 4096),
    )

    def run_bag() -> None:
        bag_matmul(
            a,
            b,
            out=bag_output,
            workspace=workspace,
            allow_tf32=allow_tf32,
            schedule=schedule,
        )

    def run_gemm() -> None:
        torch.mm(a, b, out=gemm_output)

    # Compile every Triton specialization, initialize cuBLAS, and raise GPU
    # clocks before correctness checks and event timing.
    for _ in range(5):
        run_bag()
        run_gemm()
    torch.cuda.synchronize()

    run_bag()
    run_gemm()
    torch.cuda.synchronize()
    difference = bag_output - gemm_output
    relative_l2 = float(
        torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(gemm_output)
    )
    max_absolute = float(difference.abs().max())
    relative_limit = 3.0e-4 if precision == "ieee" else 6.0e-3
    absolute_limit = 2.0e-3 + 2.0e-5 * size if precision == "ieee" else 0.15
    if relative_l2 > relative_limit or max_absolute > absolute_limit:
        raise AssertionError(
            f"BAG correctness failed for N={size}, {layout}, {precision}: "
            f"rel_l2={relative_l2:.6e}, max_abs={max_absolute:.6e}"
        )

    # Both callables overwrite preallocated output.  do_bench uses CUDA events
    # and flushes L2; allocation and JIT are outside the timed region.
    gemm_p20, gemm_median, gemm_p80 = _benchmark(run_gemm, warmup_ms, rep_ms)
    bag_p20, bag_median, bag_p80 = _benchmark(run_bag, warmup_ms, rep_ms)

    classical_flops = 2.0 * size**3
    bag_gemm_flops = 7.0 * 2.0 * (size // 2) ** 3
    return Result(
        gpu=torch.cuda.get_device_name(),
        schedule=schedule,
        precision=precision,
        layout=layout,
        n=size,
        bag_p20_ms=bag_p20,
        bag_median_ms=bag_median,
        bag_p80_ms=bag_p80,
        gemm_p20_ms=gemm_p20,
        gemm_median_ms=gemm_median,
        gemm_p80_ms=gemm_p80,
        speedup=gemm_median / bag_median,
        relative_l2_error=relative_l2,
        max_absolute_error=max_absolute,
        bag_classical_equivalent_tflops=classical_flops / bag_median / 1.0e9,
        bag_executed_gemm_tflops=bag_gemm_flops / bag_median / 1.0e9,
        gemm_tflops=classical_flops / gemm_median / 1.0e9,
    )


def _print_result(result: Result) -> None:
    print(
        f"{result.precision:4s} {result.layout:12s} N={result.n:4d}  "
        f"BAG={result.bag_median_ms:9.4f} ms  "
        f"GEMM={result.gemm_median_ms:9.4f} ms  "
        f"speedup={result.speedup:6.3f}x  "
        f"equiv={result.bag_classical_equivalent_tflops:7.2f} TFLOP/s  "
        f"rel_l2={result.relative_l2_error:.3e}"
    )


def _write_csv(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout",
        nargs="+",
        choices=("row-major", "column-major"),
        default=("row-major", "column-major"),
    )
    parser.add_argument(
        "--precision", nargs="+", choices=("ieee", "tf32"), default=("ieee", "tf32")
    )
    parser.add_argument("--size", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--schedule",
        choices=(
            "auto", "serial", "multistream", "seven-stream", "hybrid", "grouped"
        ),
        default="auto",
    )
    args = parser.parse_args()

    if any(size <= 0 or size % 2 for size in args.size):
        parser.error("every --size must be positive and even")
    torch.manual_seed(args.seed)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Implementation: {IMPLEMENTATION_ROOT}")
    print("BAG: depth=1, seven N/2 GEMMs, preallocated schedule-specific workspace")
    print("GEMM: torch.mm(out=...), cuBLAS, preallocated output")
    print(f"Schedule: {args.schedule}")

    results: list[Result] = []
    for precision in args.precision:
        for layout in args.layout:
            for size in args.size:
                result = _run_one(
                    size,
                    layout,
                    precision,
                    args.warmup_ms,
                    args.rep_ms,
                    args.schedule,
                )
                results.append(result)
                _print_result(result)

    speedup_geomean = math.exp(
        sum(math.log(result.speedup) for result in results) / len(results)
    )
    bag_geomean_ms = math.exp(
        sum(math.log(result.bag_median_ms) for result in results) / len(results)
    )
    print(f"BAG_GEOMEAN_MS={bag_geomean_ms:.6f}")
    print(f"SPEEDUP_GEOMEAN={speedup_geomean:.6f}")
    print("CORRECT=True")
    if args.output:
        _write_csv(args.output, results)
        print(f"CSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
