#!/usr/bin/env python3
"""Correctness and interface tests for one-level Triton BAG matmul."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_ROOT = Path(
    os.environ.get("BAG_TRITON_IMPL_ROOT", REPOSITORY_ROOT / "src")
).resolve()
sys.path.insert(0, str(IMPLEMENTATION_ROOT))

from bag_triton import allocate_bag_workspace, bag_matmul  # noqa: E402


def _matrix(size: int, layout: str) -> torch.Tensor:
    if layout == "row-major":
        tensor = torch.empty((size, size), device="cuda", dtype=torch.float32)
    elif layout == "column-major":
        tensor = torch.empty_strided(
            (size, size), (1, size), device="cuda", dtype=torch.float32
        )
    elif layout == "sliced":
        base = torch.empty((2 * size, 3 * size), device="cuda", dtype=torch.float32)
        tensor = base[::2, ::3]
    else:
        raise AssertionError(f"unknown layout {layout}")
    return tensor.uniform_(-0.5, 0.5)


def _run_case(size: int, layout: str, precision: str) -> tuple[float, float]:
    allow_tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    a = _matrix(size, layout)
    b = _matrix(size, layout)
    output = _matrix(size, layout)
    workspace = allocate_bag_workspace(a, b, output)
    reference = torch.mm(a, b)
    returned = bag_matmul(
        a, b, out=output, workspace=workspace, allow_tf32=allow_tf32
    )
    torch.cuda.synchronize()

    if returned is not output:
        raise AssertionError("bag_matmul must return the supplied output object")
    difference = output - reference
    relative_l2 = float(
        torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)
    )
    max_absolute = float(difference.abs().max())
    relative_limit = 3.0e-4 if precision == "ieee" else 6.0e-3
    absolute_limit = 2.0e-3 + 2.0e-5 * size if precision == "ieee" else 0.15
    if relative_l2 > relative_limit or max_absolute > absolute_limit:
        raise AssertionError(
            f"BAG error too large: rel_l2={relative_l2:.6e}, "
            f"max_abs={max_absolute:.6e}, limits=({relative_limit}, {absolute_limit})"
        )
    return relative_l2, max_absolute


def _run_interface_checks() -> None:
    a = _matrix(8, "row-major")
    b = _matrix(8, "row-major")
    try:
        bag_matmul(a[:7, :7], b[:7, :7])
    except ValueError:
        pass
    else:
        raise AssertionError("odd matrix size must be rejected")

    try:
        bag_matmul(a, b.to(torch.float16))
    except TypeError:
        pass
    else:
        raise AssertionError("non-FP32 input must be rejected")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    sizes = (2, 6, 64) if args.quick else (2, 6, 64, 510, 1024)
    layouts = ("row-major", "column-major", "sliced")
    precisions = ("ieee", "tf32")
    count = 0
    for precision in precisions:
        for layout in layouts:
            for size in sizes:
                if layout == "sliced" and size > 510:
                    continue
                relative_l2, max_absolute = _run_case(size, layout, precision)
                count += 1
                print(
                    f"PASS precision={precision:4s} layout={layout:12s} N={size:4d} "
                    f"rel_l2={relative_l2:.3e} max_abs={max_absolute:.3e}"
                )
    _run_interface_checks()
    print(f"All {count} Triton BAG numerical cases and interface checks passed.")


if __name__ == "__main__":
    main()
