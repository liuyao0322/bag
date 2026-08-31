"""One-level BAG matrix multiplication assembled from the Triton primitive.

For square FP32 matrices split into four equally sized quadrants, BAG computes
seven half-sized matrix products instead of the eight used by a conventional
blocked GEMM.  This implementation algebraically simplifies the forward basis
transform so only two three-term blocks need to be materialized::

    SA = A12 - A21 + A22
    SB = B12 - B21 + B22

All remaining linear combinations are fused into :func:`bag_primitive`.  The
four transformed output blocks are accumulated directly in ``out`` and then
decoded in place by a final Triton kernel.  The public operation implements
one BAG recursion level (``depth=1`` in the terminology used by ``BAG.md``).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from .primitive import bag_primitive


class BagWorkspace(NamedTuple):
    """Preallocated workspace for :func:`bag_matmul`.

    Both tensors have shape ``(N / 2, N / 2)``.  Reusing this object keeps
    allocation out of steady-state execution and is recommended for timing or
    repeated calls.
    """

    transformed_a12: torch.Tensor
    transformed_b12: torch.Tensor


@triton.jit
def _prepare_three_term_blocks_kernel(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    half,
    stride_a0,
    stride_a1,
    stride_b0,
    stride_b1,
    stride_sa0,
    stride_sa1,
    stride_sb0,
    stride_sb1,
    BLOCK: tl.constexpr,
):
    """Materialize SA=A12-A21+A22 and SB=B12-B21+B22."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < half * half
    rows = offsets // half
    columns = offsets - rows * half

    a12_offsets = rows * stride_a0 + (columns + half) * stride_a1
    a21_offsets = (rows + half) * stride_a0 + columns * stride_a1
    a22_offsets = (rows + half) * stride_a0 + (columns + half) * stride_a1
    b12_offsets = rows * stride_b0 + (columns + half) * stride_b1
    b21_offsets = (rows + half) * stride_b0 + columns * stride_b1
    b22_offsets = (rows + half) * stride_b0 + (columns + half) * stride_b1

    transformed_a12 = (
        tl.load(a_ptr + a12_offsets, mask=mask)
        - tl.load(a_ptr + a21_offsets, mask=mask)
        + tl.load(a_ptr + a22_offsets, mask=mask)
    )
    transformed_b12 = (
        tl.load(b_ptr + b12_offsets, mask=mask)
        - tl.load(b_ptr + b21_offsets, mask=mask)
        + tl.load(b_ptr + b22_offsets, mask=mask)
    )

    tl.store(sa_ptr + rows * stride_sa0 + columns * stride_sa1, transformed_a12, mask=mask)
    tl.store(sb_ptr + rows * stride_sb0 + columns * stride_sb1, transformed_b12, mask=mask)


@triton.jit
def _inverse_basis_transform_kernel(
    output_ptr,
    half,
    stride_output0,
    stride_output1,
    BLOCK: tl.constexpr,
):
    """Decode four C-psi blocks into C, in place."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < half * half
    rows = offsets // half
    columns = offsets - rows * half

    c11_offsets = rows * stride_output0 + columns * stride_output1
    c12_offsets = rows * stride_output0 + (columns + half) * stride_output1
    c21_offsets = (rows + half) * stride_output0 + columns * stride_output1
    c22_offsets = (rows + half) * stride_output0 + (columns + half) * stride_output1

    c_psi11 = tl.load(output_ptr + c11_offsets, mask=mask)
    c_psi12 = tl.load(output_ptr + c12_offsets, mask=mask)
    c_psi21 = tl.load(output_ptr + c21_offsets, mask=mask)
    c_psi22 = tl.load(output_ptr + c22_offsets, mask=mask)

    tl.store(output_ptr + c11_offsets, c_psi11, mask=mask)
    tl.store(output_ptr + c12_offsets, c_psi12 - c_psi21, mask=mask)
    tl.store(output_ptr + c21_offsets, -c_psi12 + c_psi22, mask=mask)
    tl.store(
        output_ptr + c22_offsets,
        -c_psi12 + c_psi21 + c_psi22,
        mask=mask,
    )


@triton.jit
def _complete_four_term_blocks_kernel(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    half,
    stride_a0,
    stride_a1,
    stride_b0,
    stride_b1,
    stride_sa0,
    stride_sa1,
    stride_sb0,
    stride_sb1,
    BLOCK: tl.constexpr,
):
    """Convert SA/SB in place to SA-A11 and SB-B11 after M3."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < half * half
    rows = offsets // half
    columns = offsets - rows * half
    sa_offsets = rows * stride_sa0 + columns * stride_sa1
    sb_offsets = rows * stride_sb0 + columns * stride_sb1
    a11_offsets = rows * stride_a0 + columns * stride_a1
    b11_offsets = rows * stride_b0 + columns * stride_b1
    tl.store(
        sa_ptr + sa_offsets,
        tl.load(sa_ptr + sa_offsets, mask=mask)
        - tl.load(a_ptr + a11_offsets, mask=mask),
        mask=mask,
    )
    tl.store(
        sb_ptr + sb_offsets,
        tl.load(sb_ptr + sb_offsets, mask=mask)
        - tl.load(b_ptr + b11_offsets, mask=mask),
        mask=mask,
    )


def _is_column_major(tensor: torch.Tensor) -> bool:
    rows, columns = tensor.shape
    return (rows <= 1 or tensor.stride(0) == 1) and (
        columns <= 1 or tensor.stride(1) >= max(rows, 1)
    )


def _check_square_matrix(
    name: str,
    tensor: torch.Tensor,
    size: int,
    device: torch.device,
    *,
    output: bool = False,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2 or tuple(tensor.shape) != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
    if any(stride < 0 for stride in tensor.stride()):
        raise ValueError(f"{name} must have non-negative strides, got {tensor.stride()}")
    if output and any(
        extent > 1 and stride == 0 for extent, stride in zip(tensor.shape, tensor.stride())
    ):
        raise ValueError(f"{name} must not have broadcasted output dimensions")
    if tensor.requires_grad:
        raise ValueError(f"{name} must not require gradients")


def allocate_bag_workspace(
    a: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> BagWorkspace:
    """Allocate the two quarter-matrix buffers needed by one BAG level.

    Column-major storage is retained when every supplied matrix is
    column-major; otherwise the workspace is row-major.  Allocation is a setup
    operation and should be done outside timed regions.
    """

    if not isinstance(a, torch.Tensor) or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square two-dimensional torch.Tensor")
    size = a.shape[0]
    if size <= 0 or size % 2:
        raise ValueError(f"BAG requires a positive even matrix size, got {size}")
    matrices = [a]
    if b is not None:
        matrices.append(b)
    if out is not None:
        matrices.append(out)
    use_column_major = all(_is_column_major(matrix) for matrix in matrices)
    half = size // 2
    strides = (1, half) if use_column_major else (half, 1)
    sa = torch.empty_strided((half, half), strides, dtype=torch.float32, device=a.device)
    sb = torch.empty_strided((half, half), strides, dtype=torch.float32, device=a.device)
    return BagWorkspace(sa, sb)


def bag_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[BagWorkspace] = None,
    allow_tf32: bool = False,
) -> torch.Tensor:
    """Compute ``out = a @ b`` with one level of the BAG algorithm.

    Args:
        a: Square, even-sized FP32 CUDA matrix.
        b: Matrix with the same shape, dtype, and device as ``a``.
        out: Optional preallocated output.  It may have row-major,
            column-major, or another non-overlapping positive-stride layout.
        workspace: Optional buffers from :func:`allocate_bag_workspace`.
            Supplying one prevents allocation on every call.
        allow_tf32: Select TF32 tensor-core products.  The default is IEEE
            FP32, matching :func:`bag_primitive`.

    Returns:
        ``out`` when supplied, otherwise a newly allocated output tensor.

    Notes:
        This operation implements exactly one recursive BAG level.  Input,
        output, and workspace tensors must not overlap.  It is an inference /
        low-level kernel and does not provide autograd support.
    """

    if not isinstance(a, torch.Tensor) or a.ndim != 2:
        raise ValueError("a must be a two-dimensional torch.Tensor")
    if not a.is_cuda:
        raise ValueError(f"a must be a CUDA tensor, got device {a.device}")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square, got shape {tuple(a.shape)}")
    size = a.shape[0]
    if size <= 0 or size % 2:
        raise ValueError(f"BAG requires a positive even matrix size, got {size}")
    if not isinstance(allow_tf32, bool):
        raise TypeError(f"allow_tf32 must be bool, got {type(allow_tf32).__name__}")

    device = a.device
    _check_square_matrix("a", a, size, device)
    _check_square_matrix("b", b, size, device)

    if out is None:
        out = torch.empty_like(a, memory_format=torch.preserve_format)
    else:
        _check_square_matrix("out", out, size, device, output=True)

    if workspace is None:
        workspace = allocate_bag_workspace(a, b, out)
    elif not isinstance(workspace, BagWorkspace):
        try:
            workspace = BagWorkspace(*workspace)
        except (TypeError, ValueError) as error:
            raise TypeError("workspace must be a BagWorkspace or a pair of tensors") from error

    half = size // 2
    sa, sb = workspace
    _check_square_matrix("workspace.transformed_a12", sa, half, device, output=True)
    _check_square_matrix("workspace.transformed_b12", sb, half, device, output=True)

    element_count = half * half
    block = 256
    _prepare_three_term_blocks_kernel[(triton.cdiv(element_count, block),)](
        a,
        b,
        sa,
        sb,
        half,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        sa.stride(0),
        sa.stride(1),
        sb.stride(0),
        sb.stride(1),
        BLOCK=block,
        num_warps=4,
    )

    a11 = a[:half, :half]
    a12 = a[:half, half:]
    a21 = a[half:, :half]
    a22 = a[half:, half:]
    b11 = b[:half, :half]
    b12 = b[:half, half:]
    b21 = b[half:, :half]
    b22 = b[half:, half:]
    c_psi11 = out[:half, :half]
    c_psi12 = out[:half, half:]
    c_psi21 = out[half:, :half]
    c_psi22 = out[half:, half:]

    # M1=(A12+A22)(B12+B22) -> Cpsi22
    bag_primitive(
        a12, a22, b12, b22, c_psi22, None,
        delta=1, epsilon=1, gamma_d=2, gamma_e=0, allow_tf32=allow_tf32,
        _atomic_accumulate=False,
    )
    # M2=(A22-A21)(B22-B21) -> Cpsi21
    bag_primitive(
        a22, a21, b22, b21, c_psi21, None,
        delta=-1, epsilon=-1, gamma_d=2, gamma_e=0, allow_tf32=allow_tf32,
        _atomic_accumulate=False,
    )
    # M3=SA*SB -> Cpsi12; M4=A11*B11 -> Cpsi11
    bag_primitive(
        sa, None, sb, None, c_psi12, None,
        delta=0, epsilon=0, gamma_d=2, gamma_e=0, allow_tf32=allow_tf32,
        _atomic_accumulate=False,
    )
    # M3 was the last consumer of the three-term blocks.  Complete both
    # four-term combinations in place so M6/M7 avoid an input subtraction in
    # every K tile.  This also avoids a severe IEEE lowering cliff when the
    # physical right operand alone contains a fused linear combination.
    _complete_four_term_blocks_kernel[(triton.cdiv(element_count, block),)](
        a,
        b,
        sa,
        sb,
        half,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        sa.stride(0),
        sa.stride(1),
        sb.stride(0),
        sb.stride(1),
        BLOCK=block,
        num_warps=4,
    )
    bag_primitive(
        a11, None, b11, None, c_psi11, None,
        delta=0, epsilon=0, gamma_d=2, gamma_e=0, allow_tf32=allow_tf32,
        _atomic_accumulate=False,
    )
    # M5 simplifies to A12*B21 and contributes to two transformed outputs.
    bag_primitive(
        a12, None, b21, None, c_psi11, c_psi12,
        delta=0, epsilon=0, gamma_d=1, gamma_e=1, allow_tf32=allow_tf32,
        _atomic_accumulate=False,
    )
    # M6=SA'*B12, where SA' now stores SA-A11.
    bag_primitive(
        sa, None, b12, None, c_psi12, c_psi22,
        delta=0, epsilon=0, gamma_d=-1, gamma_e=-1, allow_tf32=allow_tf32,
        _atomic_accumulate=False,
    )
    # M7=A21*SB', where SB' now stores SB-B11.
    bag_primitive(
        a21, None, sb, None, c_psi12, c_psi21,
        delta=0, epsilon=0, gamma_d=1, gamma_e=1, allow_tf32=allow_tf32,
        _atomic_accumulate=False,
    )

    _inverse_basis_transform_kernel[(triton.cdiv(element_count, block),)](
        out,
        half,
        out.stride(0),
        out.stride(1),
        BLOCK=block,
        num_warps=4,
    )
    return out
