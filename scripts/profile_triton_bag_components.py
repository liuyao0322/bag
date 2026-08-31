#!/usr/bin/env python3
"""CUDA-event breakdown of the seven primitive calls inside one BAG level."""

from __future__ import annotations

import argparse
import importlib
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
    if layout == "column-major":
        tensor = torch.empty_strided(
            (size, size), (1, size), device="cuda", dtype=torch.float32
        )
    else:
        tensor = torch.empty((size, size), device="cuda", dtype=torch.float32)
    return tensor.uniform_(-0.5, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--layout", choices=("row-major", "column-major"), default="row-major")
    parser.add_argument("--precision", choices=("ieee", "tf32"), default="ieee")
    args = parser.parse_args()

    allow_tf32 = args.precision == "tf32"
    a, b = _matrix(args.size, args.layout), _matrix(args.size, args.layout)
    output = _matrix(args.size, args.layout)
    workspace = allocate_bag_workspace(a, b, output)
    for _ in range(3):
        bag_matmul(a, b, out=output, workspace=workspace, allow_tf32=allow_tf32)
    torch.cuda.synchronize()

    bag_module = importlib.import_module("bag_triton.bag")
    original_primitive = bag_module.bag_primitive
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def measured_primitive(*primitive_args, **primitive_kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_primitive(*primitive_args, **primitive_kwargs)
        end.record()
        events.append((start, end))
        return result

    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    bag_module.bag_primitive = measured_primitive
    total_start.record()
    bag_matmul(a, b, out=output, workspace=workspace, allow_tf32=allow_tf32)
    total_end.record()
    torch.cuda.synchronize()
    bag_module.bag_primitive = original_primitive

    labels = ("M1", "M2", "M3", "M4", "M5", "M6", "M7")
    primitive_sum = 0.0
    for label, (start, end) in zip(labels, events):
        elapsed = start.elapsed_time(end)
        primitive_sum += elapsed
        print(f"{label}={elapsed:.6f} ms")
    total = total_start.elapsed_time(total_end)
    print(f"PRIMITIVE_SUM={primitive_sum:.6f} ms")
    print(f"TRANSFORM_AND_HOST_GAP={total - primitive_sum:.6f} ms")
    print(f"TOTAL={total:.6f} ms")


if __name__ == "__main__":
    main()
