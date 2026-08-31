#!/usr/bin/env python3
"""Benchmark the fused Triton BAG primitive against a strong unfused baseline.

The unfused path preallocates all intermediate tensors and executes five GPU
operators:

1. ``left = x - y``
2. ``right = v - w``
3. ``product = left @ right``
4. ``d += product``
5. ``e += product``

This deliberately excludes allocator overhead, so the comparison isolates the
benefit of fusion, fewer kernel launches, and lower global-memory traffic.
"""

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

from bag_triton import bag_primitive  # noqa: E402


DEFAULT_SHAPES = (
    ("square-256", 256, 256, 256),
    ("square-512", 512, 512, 512),
    ("square-1024", 1024, 1024, 1024),
    ("square-2048", 2048, 2048, 2048),
    ("square-4096", 4096, 4096, 4096),
    ("tall", 4096, 1024, 1024),
    ("wide", 1024, 4096, 1024),
    ("large-k", 1024, 1024, 4096),
)


@dataclass(frozen=True)
class Result:
    gpu: str
    precision: str
    layout: str
    shape: str
    m: int
    n: int
    k: int
    fused_p20_ms: float
    fused_median_ms: float
    fused_p80_ms: float
    unfused_p20_ms: float
    unfused_median_ms: float
    unfused_p80_ms: float
    speedup: float
    fused_effective_tflops: float
    unfused_effective_tflops: float


def _matrix(shape: tuple[int, int], layout: str, *, random: bool = True) -> torch.Tensor:
    if layout == "column-major":
        tensor = torch.empty_strided(shape, (1, shape[0]), device="cuda", dtype=torch.float32)
    elif layout == "row-major":
        tensor = torch.empty(shape, device="cuda", dtype=torch.float32)
    else:
        raise ValueError(f"unsupported layout: {layout}")
    return tensor.uniform_(-1.0, 1.0) if random else tensor


def _unfused_callable(
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
) -> Callable[[], None]:
    # Preserve the logical layout and preallocate every temporary.  Allocator
    # overhead is intentionally not part of the benchmark.
    left = torch.empty_like(x, memory_format=torch.preserve_format)
    right = torch.empty_like(v, memory_format=torch.preserve_format)
    product = torch.empty_like(d, memory_format=torch.preserve_format)

    def run() -> None:
        torch.sub(x, y, out=left)
        torch.sub(v, w, out=right)
        torch.mm(left, right, out=product)
        d.add_(product)
        e.add_(product)

    return run


def _fused_callable(
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    *,
    allow_tf32: bool,
) -> Callable[[], None]:
    def run() -> None:
        bag_primitive(
            x,
            y,
            v,
            w,
            d,
            e,
            delta=-1,
            epsilon=-1,
            gamma_d=1,
            gamma_e=1,
            allow_tf32=allow_tf32,
        )

    return run


def _check_correctness(
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    layout: str,
    *,
    allow_tf32: bool,
) -> None:
    m, k = x.shape
    n = v.shape[1]
    initial_d = _matrix((m, n), layout)
    initial_e = _matrix((m, n), layout)
    reference_d = initial_d.clone(memory_format=torch.preserve_format)
    reference_e = initial_e.clone(memory_format=torch.preserve_format)
    fused_d = initial_d.clone(memory_format=torch.preserve_format)
    fused_e = initial_e.clone(memory_format=torch.preserve_format)

    _unfused_callable(x, y, v, w, reference_d, reference_e)()
    _fused_callable(
        x, y, v, w, fused_d, fused_e, allow_tf32=allow_tf32,
    )()
    torch.cuda.synchronize()

    if allow_tf32:
        atol, rtol = 5.0e-2 + 1.0e-4 * k, 1.0e-2
    else:
        atol, rtol = 3.0e-4 + 4.0e-6 * k, 5.0e-4
    torch.testing.assert_close(fused_d, reference_d, atol=atol, rtol=rtol)
    torch.testing.assert_close(fused_e, reference_e, atol=atol, rtol=rtol)


def _benchmark(fn: Callable[[], None], warmup_ms: int, rep_ms: int) -> tuple[float, float, float]:
    p20, median, p80 = triton.testing.do_bench(
        fn,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.2, 0.5, 0.8],
    )
    return float(p20), float(median), float(p80)


def _effective_flops(m: int, n: int, k: int) -> int:
    # Core GEMM plus two input combinations and two output accumulations.
    return 2 * m * n * k + m * k + k * n + 2 * m * n


def _run_one(
    name: str,
    m: int,
    n: int,
    k: int,
    layout: str,
    precision: str,
    warmup_ms: int,
    rep_ms: int,
) -> Result:
    allow_tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    x, y = _matrix((m, k), layout), _matrix((m, k), layout)
    v, w = _matrix((k, n), layout), _matrix((k, n), layout)
    unfused_d, unfused_e = _matrix((m, n), layout), _matrix((m, n), layout)
    fused_d, fused_e = _matrix((m, n), layout), _matrix((m, n), layout)

    _check_correctness(x, y, v, w, layout, allow_tf32=allow_tf32)

    unfused = _unfused_callable(x, y, v, w, unfused_d, unfused_e)
    fused = _fused_callable(x, y, v, w, fused_d, fused_e, allow_tf32=allow_tf32)

    # Run both paths before timing so JIT compilation, library initialization,
    # and initial GPU clock ramp-up are outside the measured region.
    for _ in range(5):
        unfused()
        fused()
    torch.cuda.synchronize()

    # Benchmark the unfused path first and then fused.  Each do_bench call has
    # its own time-based warmup and flushes L2 before every timed sample.
    unfused_p20, unfused_median, unfused_p80 = _benchmark(unfused, warmup_ms, rep_ms)
    fused_p20, fused_median, fused_p80 = _benchmark(fused, warmup_ms, rep_ms)

    operations = _effective_flops(m, n, k)
    return Result(
        gpu=torch.cuda.get_device_name(),
        precision=precision,
        layout=layout,
        shape=name,
        m=m,
        n=n,
        k=k,
        fused_p20_ms=fused_p20,
        fused_median_ms=fused_median,
        fused_p80_ms=fused_p80,
        unfused_p20_ms=unfused_p20,
        unfused_median_ms=unfused_median,
        unfused_p80_ms=unfused_p80,
        speedup=unfused_median / fused_median,
        fused_effective_tflops=operations / fused_median / 1.0e9,
        unfused_effective_tflops=operations / unfused_median / 1.0e9,
    )


def _print_result(result: Result) -> None:
    print(
        f"{result.precision:4s} {result.layout:12s} {result.shape:11s} "
        f"({result.m:4d},{result.n:4d},{result.k:4d})  "
        f"fused={result.fused_median_ms:8.4f} ms  "
        f"unfused={result.unfused_median_ms:8.4f} ms  "
        f"speedup={result.speedup:6.3f}x  "
        f"effective={result.fused_effective_tflops:6.2f} TFLOP/s"
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
        default=("column-major",),
    )
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=("ieee", "tf32"),
        default=("ieee",),
    )
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--shape",
        nargs="+",
        choices=tuple(item[0] for item in DEFAULT_SHAPES),
        help="Only benchmark the selected named shapes (default: all).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    results: list[Result] = []
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Implementation: {IMPLEMENTATION_ROOT}")
    print("Primitive: (X-Y)@(V-W), atomically accumulated into D and E")
    print("Unfused: 2 x subtract + GEMM + 2 x output add; all buffers preallocated")

    selected_shapes = [
        item for item in DEFAULT_SHAPES if args.shape is None or item[0] in args.shape
    ]
    for precision in args.precision:
        for layout in args.layout:
            for name, m, n, k in selected_shapes:
                result = _run_one(
                    name, m, n, k, layout, precision, args.warmup_ms, args.rep_ms,
                )
                results.append(result)
                _print_result(result)

    geometric_mean = math.exp(
        sum(math.log(result.speedup) for result in results) / len(results)
    )
    fused_geometric_mean_ms = math.exp(
        sum(math.log(result.fused_median_ms) for result in results) / len(results)
    )
    print(f"FUSED_GEOMEAN_MS={fused_geometric_mean_ms:.6f}")
    print(f"SPEEDUP_GEOMEAN={geometric_mean:.6f}")
    print("CORRECT=True")

    if args.output:
        _write_csv(args.output, results)
        print(f"CSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
