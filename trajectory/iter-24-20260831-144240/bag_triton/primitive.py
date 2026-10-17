"""Fused Triton implementation of the BAG matrix-multiplication primitive.

The primitive computes::

    product = (x + delta * y) @ (v + epsilon * w)
    d += gamma_d * product
    e += gamma_e * product

``delta`` and ``epsilon`` are compile-time values in ``{-1, 0, 1}``.  The
output coefficients use the convention of the existing CUDA BAG kernels:

* ``0``: do not access the output;
* ``1`` / ``-1``: atomically accumulate ``+product`` / ``-product``;
* ``2`` / ``-2``: overwrite the output with ``+product`` / ``-product``.

The overwrite encoding is useful for the first writer to a BAG output block;
it avoids a separate zero-fill and the cost of atomic operations.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl


class _LaunchConfig(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int


@triton.jit
def _bag_primitive_kernel(
    x_ptr,
    y_ptr,
    v_ptr,
    w_ptr,
    d_ptr,
    e_ptr,
    m,
    n,
    k,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    stride_vk,
    stride_vn,
    stride_wk,
    stride_wn,
    stride_dm,
    stride_dn,
    stride_em,
    stride_en,
    DELTA: tl.constexpr,
    EPSILON: tl.constexpr,
    GAMMA_D: tl.constexpr,
    GAMMA_E: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Compute one output tile of the fused BAG primitive."""

    # Group programs along M so neighboring programs reuse input tiles through
    # L2.  This is the same launch ordering used by Triton's matmul tutorial.
    program_id = tl.program_id(axis=0)
    num_programs_m = tl.cdiv(m, BLOCK_M)
    num_programs_n = tl.cdiv(n, BLOCK_N)
    programs_per_group = GROUP_M * num_programs_n
    group_id = program_id // programs_per_group
    first_program_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_programs_m - first_program_m, GROUP_M)
    program_m = first_program_m + ((program_id % programs_per_group) % group_size_m)
    program_n = (program_id % programs_per_group) // group_size_m

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # DELTA and EPSILON are constexpr values.  Triton removes the unused input
    # pointer, loads, and arithmetic for the common X@V specialization.
    for k_start in range(0, tl.cdiv(k, BLOCK_K)):
        current_k = k_start * BLOCK_K + offsets_k
        x_offsets = offsets_m[:, None] * stride_xm + current_k[None, :] * stride_xk
        v_offsets = current_k[:, None] * stride_vk + offsets_n[None, :] * stride_vn
        x_mask = (offsets_m[:, None] < m) & (current_k[None, :] < k)
        v_mask = (current_k[:, None] < k) & (offsets_n[None, :] < n)

        left = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
        if DELTA == 1:
            y_offsets = offsets_m[:, None] * stride_ym + current_k[None, :] * stride_yk
            left += tl.load(y_ptr + y_offsets, mask=x_mask, other=0.0)
        elif DELTA == -1:
            y_offsets = offsets_m[:, None] * stride_ym + current_k[None, :] * stride_yk
            left -= tl.load(y_ptr + y_offsets, mask=x_mask, other=0.0)

        right = tl.load(v_ptr + v_offsets, mask=v_mask, other=0.0)
        if EPSILON == 1:
            w_offsets = current_k[:, None] * stride_wk + offsets_n[None, :] * stride_wn
            right += tl.load(w_ptr + w_offsets, mask=v_mask, other=0.0)
        elif EPSILON == -1:
            w_offsets = current_k[:, None] * stride_wk + offsets_n[None, :] * stride_wn
            right -= tl.load(w_ptr + w_offsets, mask=v_mask, other=0.0)

        accumulator += tl.dot(left, right, input_precision=INPUT_PRECISION)

    output_mask = (offsets_m[:, None] < m) & (offsets_n[None, :] < n)

    # Additive modes are atomic so independent BAG primitives can be launched
    # concurrently even when they update the same output block.
    if GAMMA_D != 0:
        d_offsets = offsets_m[:, None] * stride_dm + offsets_n[None, :] * stride_dn
        if GAMMA_D == 1:
            tl.atomic_add(d_ptr + d_offsets, accumulator, mask=output_mask)
        elif GAMMA_D == -1:
            tl.atomic_add(d_ptr + d_offsets, -accumulator, mask=output_mask)
        elif GAMMA_D == 2:
            tl.store(d_ptr + d_offsets, accumulator, mask=output_mask)
        elif GAMMA_D == -2:
            tl.store(d_ptr + d_offsets, -accumulator, mask=output_mask)

    if GAMMA_E != 0:
        e_offsets = offsets_m[:, None] * stride_em + offsets_n[None, :] * stride_en
        if GAMMA_E == 1:
            tl.atomic_add(e_ptr + e_offsets, accumulator, mask=output_mask)
        elif GAMMA_E == -1:
            tl.atomic_add(e_ptr + e_offsets, -accumulator, mask=output_mask)
        elif GAMMA_E == 2:
            tl.store(e_ptr + e_offsets, accumulator, mask=output_mask)
        elif GAMMA_E == -2:
            tl.store(e_ptr + e_offsets, -accumulator, mask=output_mask)


def _select_launch_config(
    m: int,
    n: int,
    k: int,
    *,
    delta: int,
    epsilon: int,
) -> _LaunchConfig:
    """Select a mutation-safe launch configuration without runtime autotuning.

    Triton's autotuner executes a candidate kernel several times.  That is
    undesirable for an in-place, atomic-accumulation primitive because every
    trial would mutate D and E.  These shape buckets mirror the CUDA kernels in
    this repository while retaining boundary masks for arbitrary dimensions.
    """

    block_k = 16 if k < 32 else 32
    has_input_combination = delta != 0 or epsilon != 0
    if m >= 4 * n:
        if has_input_combination and n >= 512:
            return _LaunchConfig(128, 128, 16, 8, 16, 3)
        # Eight warps keep per-thread accumulators bounded while a 64-wide
        # output tile improves reuse along the short matrix dimension.
        num_stages = 2 if has_input_combination else 3
        num_warps = 8 if has_input_combination else 4
        block_n = 64 if has_input_combination else 32
        return _LaunchConfig(128, block_n, block_k, 8, num_warps, num_stages)
    if n >= 4 * m:
        if has_input_combination and m >= 512:
            return _LaunchConfig(128, 128, 16, 8, 16, 3)
        num_stages = 2 if has_input_combination else 3
        num_warps = 8 if has_input_combination else 4
        block_m = 64 if has_input_combination else 32
        return _LaunchConfig(block_m, 128, block_k, 8, num_warps, num_stages)
    if max(m, n) <= 256:
        return _LaunchConfig(32, 32, block_k, 8, 4, 3)
    if max(m, n) <= 512 and has_input_combination:
        # Medium-small fused GEMMs need enough CTAs to fill the GPU.  The
        # previous 128x64 tile launches only 32 programs for a 512x512 output
        # and nearly fills an Ampere SM's shared memory with four input
        # streams.  A compact tile exposes 256 programs and permits multiple
        # resident CTAs while retaining a three-stage reduction pipeline.
        return _LaunchConfig(32, 32, 16, 8, 4, 2)
    if min(m, n) >= 1024:
        # Large square matrices benefit from the greater output-tile reuse;
        # halve K so four pipelined inputs remain within shared-memory limits.
        if has_input_combination:
            return _LaunchConfig(128, 128, 16, 8, 16, 3)
        if min(m, n) >= 2048:
            return _LaunchConfig(128, 128, block_k, 8, 8, 3)
    if has_input_combination:
        return _LaunchConfig(128, 64, block_k, 8, 8, 2)
    return _LaunchConfig(64, 64, block_k, 8, 4, 3)


def _check_coefficient(name: str, value: int, allowed: tuple[int, ...]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
        choices = ", ".join(str(item) for item in allowed)
        raise ValueError(f"{name} must be one of {{{choices}}}, got {value!r}")


def _check_matrix(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, int],
    device: torch.device,
    *,
    output: bool = False,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2 or tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
    if any(stride < 0 for stride in tensor.stride()):
        raise ValueError(f"{name} must have non-negative strides, got {tensor.stride()}")
    if output and any(size > 1 and stride == 0 for size, stride in zip(tensor.shape, tensor.stride())):
        raise ValueError(f"{name} must not have broadcasted output dimensions")
    if tensor.requires_grad:
        raise ValueError(f"{name} must not require gradients; bag_primitive is an in-place low-level kernel")


def _is_column_major(tensor: torch.Tensor) -> bool:
    """Return whether a logical matrix has a conventional column-major view."""

    rows, columns = tensor.shape
    row_contiguous = rows <= 1 or tensor.stride(0) == 1
    valid_leading_dimension = columns <= 1 or tensor.stride(1) >= max(rows, 1)
    return row_contiguous and valid_leading_dimension


def bag_primitive(
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    v: torch.Tensor,
    w: Optional[torch.Tensor],
    d: Optional[torch.Tensor],
    e: Optional[torch.Tensor],
    *,
    delta: int,
    epsilon: int,
    gamma_d: int,
    gamma_e: int,
    allow_tf32: bool = False,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run the fused BAG primitive on CUDA tensors.

    Args:
        x: Left input with logical shape ``(M, K)``.
        y: Optional second left input.  It may be ``None`` only when
            ``delta == 0``.
        v: Right input with logical shape ``(K, N)``.
        w: Optional second right input.  It may be ``None`` only when
            ``epsilon == 0``.
        d: First in-place output with shape ``(M, N)``.  It may be ``None``
            only when ``gamma_d == 0``.
        e: Second in-place output with shape ``(M, N)``.  It may be ``None``
            only when ``gamma_e == 0``.
        delta: Left input coefficient in ``{-1, 0, 1}``.
        epsilon: Right input coefficient in ``{-1, 0, 1}``.
        gamma_d: D update mode.  ``+/-1`` atomically accumulates and ``+/-2``
            overwrites with the signed product.
        gamma_e: E update mode, with the same encoding as ``gamma_d``.
        allow_tf32: Use TensorFloat-32 tensor-core products when true.  The
            default uses IEEE FP32 products to match the CUDA primitive's
            ``fmaf`` computation more closely.

    Returns:
        The same ``(d, e)`` objects after their in-place updates.

    Notes:
        Inputs and outputs may use row-major, column-major, or sliced positive
        strides.  Outputs must not overlap the input matrices.  Additive modes
        are safe for concurrent launches; overwrite modes require a single
        logical writer for each output element.
    """

    _check_coefficient("delta", delta, (-1, 0, 1))
    _check_coefficient("epsilon", epsilon, (-1, 0, 1))
    _check_coefficient("gamma_d", gamma_d, (-2, -1, 0, 1, 2))
    _check_coefficient("gamma_e", gamma_e, (-2, -1, 0, 1, 2))
    if not isinstance(allow_tf32, bool):
        raise TypeError(f"allow_tf32 must be bool, got {type(allow_tf32).__name__}")

    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError("x must be a two-dimensional torch.Tensor")
    if not x.is_cuda:
        raise ValueError(f"x must be a CUDA tensor, got device {x.device}")

    m, k = x.shape
    if not isinstance(v, torch.Tensor) or v.ndim != 2:
        raise ValueError("v must be a two-dimensional torch.Tensor")
    if v.shape[0] != k:
        raise ValueError(f"v must have shape ({k}, N), got {tuple(v.shape)}")
    n = v.shape[1]
    device = x.device

    _check_matrix("x", x, (m, k), device)
    _check_matrix("v", v, (k, n), device)

    if y is None:
        if delta != 0:
            raise ValueError("y may be None only when delta == 0")
        y_kernel = x
    else:
        _check_matrix("y", y, (m, k), device)
        y_kernel = y

    if w is None:
        if epsilon != 0:
            raise ValueError("w may be None only when epsilon == 0")
        w_kernel = v
    else:
        _check_matrix("w", w, (k, n), device)
        w_kernel = w

    if d is None:
        if gamma_d != 0:
            raise ValueError("d may be None only when gamma_d == 0")
        d_kernel = x
    else:
        _check_matrix("d", d, (m, n), device, output=True)
        d_kernel = d

    if e is None:
        if gamma_e != 0:
            raise ValueError("e may be None only when gamma_e == 0")
        e_kernel = x
    else:
        _check_matrix("e", e, (m, n), device, output=True)
        e_kernel = e

    if gamma_d == 0 and gamma_e == 0:
        return d, e
    if m == 0 or n == 0:
        return d, e
    if k == 0:
        # An empty reduction produces zero.  Additive updates are no-ops,
        # whereas the direct-write modes must materialize that zero.
        if abs(gamma_d) == 2:
            assert d is not None
            d.zero_()
        if abs(gamma_e) == 2:
            assert e is not None
            e.zero_()
        return d, e

    required_matrices = [x, v]
    if delta != 0:
        required_matrices.append(y_kernel)
    if epsilon != 0:
        required_matrices.append(w_kernel)
    if gamma_d != 0:
        required_matrices.append(d_kernel)
    if gamma_e != 0:
        required_matrices.append(e_kernel)
    transpose_problem = all(_is_column_major(tensor) for tensor in required_matrices)

    if transpose_problem:
        # Column-major C=A@B is physically row-major C^T=B^T@A^T.  Swapping
        # operands and M/N lets Triton's dot lowering use its efficient
        # row-major operand layouts without moving or materializing any data.
        kernel_x, kernel_y = v, w_kernel
        kernel_v, kernel_w = x, y_kernel
        kernel_m, kernel_n = n, m
        kernel_delta, kernel_epsilon = epsilon, delta
        stride_xm, stride_xk = v.stride(1), v.stride(0)
        stride_ym, stride_yk = w_kernel.stride(1), w_kernel.stride(0)
        stride_vk, stride_vn = x.stride(1), x.stride(0)
        stride_wk, stride_wn = y_kernel.stride(1), y_kernel.stride(0)
        stride_dm, stride_dn = d_kernel.stride(1), d_kernel.stride(0)
        stride_em, stride_en = e_kernel.stride(1), e_kernel.stride(0)
    else:
        kernel_x, kernel_y = x, y_kernel
        kernel_v, kernel_w = v, w_kernel
        kernel_m, kernel_n = m, n
        kernel_delta, kernel_epsilon = delta, epsilon
        stride_xm, stride_xk = x.stride(0), x.stride(1)
        stride_ym, stride_yk = y_kernel.stride(0), y_kernel.stride(1)
        stride_vk, stride_vn = v.stride(0), v.stride(1)
        stride_wk, stride_wn = w_kernel.stride(0), w_kernel.stride(1)
        stride_dm, stride_dn = d_kernel.stride(0), d_kernel.stride(1)
        stride_em, stride_en = e_kernel.stride(0), e_kernel.stride(1)

    config = _select_launch_config(
        kernel_m,
        kernel_n,
        k,
        delta=kernel_delta,
        epsilon=kernel_epsilon,
    )
    grid = (
        triton.cdiv(kernel_m, config.block_m) * triton.cdiv(kernel_n, config.block_n),
    )
    input_precision = "tf32" if allow_tf32 else "ieee"

    _bag_primitive_kernel[grid](
        kernel_x,
        kernel_y,
        kernel_v,
        kernel_w,
        d_kernel,
        e_kernel,
        kernel_m,
        kernel_n,
        k,
        stride_xm,
        stride_xk,
        stride_ym,
        stride_yk,
        stride_vk,
        stride_vn,
        stride_wk,
        stride_wn,
        stride_dm,
        stride_dn,
        stride_em,
        stride_en,
        DELTA=kernel_delta,
        EPSILON=kernel_epsilon,
        GAMMA_D=gamma_d,
        GAMMA_E=gamma_e,
        INPUT_PRECISION=input_precision,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        GROUP_M=config.group_m,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return d, e
