#!/usr/bin/env python3
"""Correctness smoke tests for the Triton BAG primitive."""

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

from bag_triton import bag_primitive  # noqa: E402


def _matrix(shape: tuple[int, int], *, layout: str) -> torch.Tensor:
    if layout == "column-major":
        tensor = torch.empty_strided(shape, (1, shape[0]), device="cuda", dtype=torch.float32)
        return tensor.uniform_(-1.0, 1.0)
    if layout == "sliced":
        # Exercise non-contiguous, non-unit positive strides on both axes.
        base = torch.empty(
            (shape[0] * 2, max(shape[1] * 3, 1)), device="cuda", dtype=torch.float32
        )
        return base[::2, : shape[1] * 3 : 3].uniform_(-1.0, 1.0)
    if layout == "row-major":
        return torch.empty(shape, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
    raise AssertionError(f"unexpected layout: {layout}")


def _expected_output(initial: torch.Tensor, product: torch.Tensor, gamma: int) -> torch.Tensor:
    if gamma == 0:
        return initial
    if gamma == 1:
        return initial + product
    if gamma == -1:
        return initial - product
    if gamma == 2:
        return product
    if gamma == -2:
        return -product
    raise AssertionError(f"unexpected gamma: {gamma}")


def _run_case(
    shape: tuple[int, int, int],
    coefficients: tuple[int, int, int, int],
    *,
    layout: str,
) -> None:
    m, n, k = shape
    delta, epsilon, gamma_d, gamma_e = coefficients
    x = _matrix((m, k), layout=layout)
    y = _matrix((m, k), layout=layout) if delta else None
    v = _matrix((k, n), layout=layout)
    w = _matrix((k, n), layout=layout) if epsilon else None
    d = _matrix((m, n), layout=layout) if gamma_d else None
    e = _matrix((m, n), layout=layout) if gamma_e else None

    left = x if delta == 0 else x + delta * y
    right = v if epsilon == 0 else v + epsilon * w
    product = left @ right
    expected_d = None if d is None else _expected_output(d.clone(), product, gamma_d)
    expected_e = None if e is None else _expected_output(e.clone(), product, gamma_e)

    returned_d, returned_e = bag_primitive(
        x,
        y,
        v,
        w,
        d,
        e,
        delta=delta,
        epsilon=epsilon,
        gamma_d=gamma_d,
        gamma_e=gamma_e,
    )
    torch.cuda.synchronize()

    if returned_d is not d or returned_e is not e:
        raise AssertionError("bag_primitive must return the original output objects")
    tolerance = {"atol": 2.0e-4 + 2.0e-6 * k, "rtol": 2.0e-4}
    if d is not None:
        torch.testing.assert_close(d, expected_d, **tolerance)
    if e is not None:
        torch.testing.assert_close(e, expected_e, **tolerance)


def _run_concurrent_accumulation_case() -> None:
    """Verify atomic output updates from independent CUDA streams."""

    m, n, k = 73, 61, 47
    x0, y0 = _matrix((m, k), layout="row-major"), _matrix((m, k), layout="row-major")
    v0, w0 = _matrix((k, n), layout="row-major"), _matrix((k, n), layout="row-major")
    x1, y1 = _matrix((m, k), layout="row-major"), _matrix((m, k), layout="row-major")
    v1, w1 = _matrix((k, n), layout="row-major"), _matrix((k, n), layout="row-major")
    output = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    expected = (x0 - y0) @ (v0 + w0) + (x1 - y1) @ (v1 + w1)

    # Warm up this specialization before testing overlap between the launches.
    warmup = torch.zeros_like(output)
    bag_primitive(
        x0, y0, v0, w0, warmup, None,
        delta=-1, epsilon=1, gamma_d=1, gamma_e=0,
    )
    torch.cuda.synchronize()

    stream0, stream1 = torch.cuda.Stream(), torch.cuda.Stream()
    ready = torch.cuda.Event()
    ready.record()
    stream0.wait_event(ready)
    stream1.wait_event(ready)
    with torch.cuda.stream(stream0):
        bag_primitive(
            x0, y0, v0, w0, output, None,
            delta=-1, epsilon=1, gamma_d=1, gamma_e=0,
        )
    with torch.cuda.stream(stream1):
        bag_primitive(
            x1, y1, v1, w1, output, None,
            delta=-1, epsilon=1, gamma_d=1, gamma_e=0,
        )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=3.0e-4, rtol=2.0e-4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")

    cases = [
        ((1, 1, 1), (0, 0, 2, 0)),
        ((37, 29, 21), (1, -1, 1, -1)),
        ((63, 97, 35), (-1, 1, -2, 2)),
        ((257, 65, 33), (0, 1, 1, 1)),
        ((257, 64, 33), (-1, -1, 1, 1)),
        ((64, 257, 33), (-1, -1, 1, 1)),
        ((511, 509, 513), (-1, -1, 1, 1)),
        ((96, 80, 0), (0, 0, 2, -2)),
    ]

    layouts = ("row-major", "column-major", "sliced")
    for layout in layouts:
        for shape, coefficients in cases:
            _run_case(shape, coefficients, layout=layout)
            print(f"PASS layout={layout:12s} shape={shape} coeffs={coefficients}")

    # Compile and exercise the 128x128 launch bucket without making every
    # strided-layout case allocate a large backing tensor.
    _run_case((2048, 2048, 33), (-1, -1, 1, 1), layout="row-major")
    print("PASS layout=row-major    shape=(2048, 2048, 33) large launch bucket")

    _run_concurrent_accumulation_case()
    print("PASS concurrent atomic accumulation on two CUDA streams")
    print(f"All {len(cases) * len(layouts) + 2} Triton BAG primitive cases passed.")


if __name__ == "__main__":
    main()
