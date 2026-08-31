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

    Core transform tensors have shape ``(N / 2, N / 2)``; optional scheduling
    buffers are populated by :func:`allocate_bag_workspace`. Reusing this
    object keeps allocation out of steady-state execution and is recommended
    for timing or repeated calls.
    """

    transformed_a12: torch.Tensor
    transformed_b12: torch.Tensor
    streams: Optional[tuple[torch.cuda.Stream, ...]] = None
    transformed_a12_m6: Optional[torch.Tensor] = None
    transformed_b12_m7: Optional[torch.Tensor] = None
    leaf_product: Optional[torch.Tensor] = None
    grouped_left: Optional[torch.Tensor] = None
    grouped_right: Optional[torch.Tensor] = None
    grouped_products: Optional[torch.Tensor] = None


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
    COLUMN_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Materialize SA=A12-A21+A22 and SB=B12-B21+B22."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < half * half
    if COLUMN_MAJOR:
        columns = offsets // half
        rows = offsets - columns * half
    else:
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
def _prepare_full_concurrency_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    sa_ptr,
    sb_ptr,
    ua_ptr,
    ub_ptr,
    half,
    stride_a0,
    stride_a1,
    stride_b0,
    stride_b1,
    stride_output0,
    stride_output1,
    stride_sa0,
    stride_sa1,
    stride_sb0,
    stride_sb1,
    stride_ua0,
    stride_ua1,
    stride_ub0,
    stride_ub1,
    ZERO_OUTPUT: tl.constexpr,
    COLUMN_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Materialize independent M3/M6/M7 inputs and zero all C-psi blocks."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < half * half
    if COLUMN_MAJOR:
        columns = offsets // half
        rows = offsets - columns * half
    else:
        rows = offsets // half
        columns = offsets - rows * half

    a11_offsets = rows * stride_a0 + columns * stride_a1
    a12_offsets = rows * stride_a0 + (columns + half) * stride_a1
    a21_offsets = (rows + half) * stride_a0 + columns * stride_a1
    a22_offsets = (rows + half) * stride_a0 + (columns + half) * stride_a1
    b11_offsets = rows * stride_b0 + columns * stride_b1
    b12_offsets = rows * stride_b0 + (columns + half) * stride_b1
    b21_offsets = (rows + half) * stride_b0 + columns * stride_b1
    b22_offsets = (rows + half) * stride_b0 + (columns + half) * stride_b1

    sa = (
        tl.load(a_ptr + a12_offsets, mask=mask)
        - tl.load(a_ptr + a21_offsets, mask=mask)
        + tl.load(a_ptr + a22_offsets, mask=mask)
    )
    sb = (
        tl.load(b_ptr + b12_offsets, mask=mask)
        - tl.load(b_ptr + b21_offsets, mask=mask)
        + tl.load(b_ptr + b22_offsets, mask=mask)
    )
    workspace_offsets_a = rows * stride_sa0 + columns * stride_sa1
    workspace_offsets_b = rows * stride_sb0 + columns * stride_sb1
    ua_offsets = rows * stride_ua0 + columns * stride_ua1
    ub_offsets = rows * stride_ub0 + columns * stride_ub1
    tl.store(sa_ptr + workspace_offsets_a, sa, mask=mask)
    tl.store(sb_ptr + workspace_offsets_b, sb, mask=mask)
    tl.store(
        ua_ptr + ua_offsets,
        sa - tl.load(a_ptr + a11_offsets, mask=mask),
        mask=mask,
    )
    tl.store(
        ub_ptr + ub_offsets,
        sb - tl.load(b_ptr + b11_offsets, mask=mask),
        mask=mask,
    )

    if ZERO_OUTPUT:
        c11_offsets = rows * stride_output0 + columns * stride_output1
        c12_offsets = rows * stride_output0 + (columns + half) * stride_output1
        c21_offsets = (rows + half) * stride_output0 + columns * stride_output1
        c22_offsets = (rows + half) * stride_output0 + (columns + half) * stride_output1
        zero = tl.zeros((BLOCK,), dtype=tl.float32)
        tl.store(output_ptr + c11_offsets, zero, mask=mask)
        tl.store(output_ptr + c12_offsets, zero, mask=mask)
        tl.store(output_ptr + c21_offsets, zero, mask=mask)
        tl.store(output_ptr + c22_offsets, zero, mask=mask)


@triton.jit
def _inverse_basis_transform_kernel(
    output_ptr,
    half,
    stride_output0,
    stride_output1,
    COLUMN_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Decode four C-psi blocks into C, in place."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < half * half
    if COLUMN_MAJOR:
        columns = offsets // half
        rows = offsets - columns * half
    else:
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
def _merge_hybrid_m6_m7_kernel(
    c12_ptr,
    c21_ptr,
    c22_ptr,
    element_count,
    stride_c120,
    stride_c121,
    stride_c210,
    stride_c211,
    stride_c220,
    stride_c221,
    half,
    COLUMN_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Merge positive temporary M6/M7 products into C-psi blocks."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < element_count
    if COLUMN_MAJOR:
        columns = offsets // half
        rows = offsets - columns * half
    else:
        rows = offsets // half
        columns = offsets - rows * half
    c12_offsets = rows * stride_c120 + columns * stride_c121
    c21_offsets = rows * stride_c210 + columns * stride_c211
    c22_offsets = rows * stride_c220 + columns * stride_c221
    m6 = tl.load(c22_ptr + c22_offsets, mask=mask)
    m7 = tl.load(c21_ptr + c21_offsets, mask=mask)
    c12 = tl.load(c12_ptr + c12_offsets, mask=mask)
    tl.store(c12_ptr + c12_offsets, c12 - m6 + m7, mask=mask)
    tl.store(c22_ptr + c22_offsets, -m6, mask=mask)


@triton.jit
def _merge_hybrid_m5_kernel(
    product_ptr,
    c11_ptr,
    c12_ptr,
    element_count,
    stride_product0,
    stride_product1,
    stride_c110,
    stride_c111,
    stride_c120,
    stride_c121,
    half,
    COLUMN_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Add the materialized M5 product to C-psi11 and C-psi12."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < element_count
    if COLUMN_MAJOR:
        columns = offsets // half
        rows = offsets - columns * half
    else:
        rows = offsets // half
        columns = offsets - rows * half
    product_offsets = rows * stride_product0 + columns * stride_product1
    c11_offsets = rows * stride_c110 + columns * stride_c111
    c12_offsets = rows * stride_c120 + columns * stride_c121
    product = tl.load(product_ptr + product_offsets, mask=mask)
    c11 = tl.load(c11_ptr + c11_offsets, mask=mask)
    c12 = tl.load(c12_ptr + c12_offsets, mask=mask)
    tl.store(c11_ptr + c11_offsets, c11 + product, mask=mask)
    tl.store(c12_ptr + c12_offsets, c12 + product, mask=mask)


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
    COLUMN_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Convert SA/SB in place to SA-A11 and SB-B11 after M3."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < half * half
    if COLUMN_MAJOR:
        columns = offsets // half
        rows = offsets - columns * half
    else:
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


@triton.jit
def _prepare_two_term_product_kernel(
    x_ptr,
    y_ptr,
    v_ptr,
    w_ptr,
    left_ptr,
    right_ptr,
    half,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_v0,
    stride_v1,
    stride_w0,
    stride_w1,
    stride_left0,
    stride_left1,
    stride_right0,
    stride_right1,
    SIGN: tl.constexpr,
    COLUMN_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Materialize two same-sign input combinations for a plain product."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < half * half
    if COLUMN_MAJOR:
        columns = offsets // half
        rows = offsets - columns * half
    else:
        rows = offsets // half
        columns = offsets - rows * half
    x_offsets = rows * stride_x0 + columns * stride_x1
    y_offsets = rows * stride_y0 + columns * stride_y1
    v_offsets = rows * stride_v0 + columns * stride_v1
    w_offsets = rows * stride_w0 + columns * stride_w1
    left_offsets = rows * stride_left0 + columns * stride_left1
    right_offsets = rows * stride_right0 + columns * stride_right1
    if SIGN == 1:
        left = tl.load(x_ptr + x_offsets, mask=mask) + tl.load(
            y_ptr + y_offsets, mask=mask
        )
        right = tl.load(v_ptr + v_offsets, mask=mask) + tl.load(
            w_ptr + w_offsets, mask=mask
        )
    else:
        left = tl.load(x_ptr + x_offsets, mask=mask) - tl.load(
            y_ptr + y_offsets, mask=mask
        )
        right = tl.load(v_ptr + v_offsets, mask=mask) - tl.load(
            w_ptr + w_offsets, mask=mask
        )
    tl.store(left_ptr + left_offsets, left, mask=mask)
    tl.store(right_ptr + right_offsets, right, mask=mask)


@triton.jit
def _prepare_outer_product_pairs_kernel(
    a_ptr,
    b_ptr,
    m1_left_ptr,
    m1_right_ptr,
    m2_left_ptr,
    m2_right_ptr,
    half,
    stride_a0,
    stride_a1,
    stride_b0,
    stride_b1,
    stride_m1_left0,
    stride_m1_left1,
    stride_m1_right0,
    stride_m1_right1,
    stride_m2_left0,
    stride_m2_left1,
    stride_m2_right0,
    stride_m2_right1,
    COLUMN_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Materialize both M1 and M2 input pairs in one coalesced pass."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < half * half
    if COLUMN_MAJOR:
        columns = offsets // half
        rows = offsets - columns * half
    else:
        rows = offsets // half
        columns = offsets - rows * half
    a12_offsets = rows * stride_a0 + (columns + half) * stride_a1
    a21_offsets = (rows + half) * stride_a0 + columns * stride_a1
    a22_offsets = (rows + half) * stride_a0 + (columns + half) * stride_a1
    b12_offsets = rows * stride_b0 + (columns + half) * stride_b1
    b21_offsets = (rows + half) * stride_b0 + columns * stride_b1
    b22_offsets = (rows + half) * stride_b0 + (columns + half) * stride_b1
    m1_left_offsets = rows * stride_m1_left0 + columns * stride_m1_left1
    m1_right_offsets = rows * stride_m1_right0 + columns * stride_m1_right1
    m2_left_offsets = rows * stride_m2_left0 + columns * stride_m2_left1
    m2_right_offsets = rows * stride_m2_right0 + columns * stride_m2_right1
    a22 = tl.load(a_ptr + a22_offsets, mask=mask)
    b22 = tl.load(b_ptr + b22_offsets, mask=mask)
    tl.store(
        m1_left_ptr + m1_left_offsets,
        tl.load(a_ptr + a12_offsets, mask=mask) + a22,
        mask=mask,
    )
    tl.store(
        m1_right_ptr + m1_right_offsets,
        tl.load(b_ptr + b12_offsets, mask=mask) + b22,
        mask=mask,
    )
    tl.store(
        m2_left_ptr + m2_left_offsets,
        a22 - tl.load(a_ptr + a21_offsets, mask=mask),
        mask=mask,
    )
    tl.store(
        m2_right_ptr + m2_right_offsets,
        b22 - tl.load(b_ptr + b21_offsets, mask=mask),
        mask=mask,
    )


@triton.jit
def _decode_hybrid_products_kernel(
    m1_ptr,
    m2_ptr,
    m3_ptr,
    m4_ptr,
    m5_ptr,
    m6_ptr,
    m7_ptr,
    output_ptr,
    element_count,
    half,
    stride_m10,
    stride_m11,
    stride_m20,
    stride_m21,
    stride_m30,
    stride_m31,
    stride_m40,
    stride_m41,
    stride_m50,
    stride_m51,
    stride_m60,
    stride_m61,
    stride_m70,
    stride_m71,
    stride_output0,
    stride_output1,
    COLUMN_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Decode seven stored leaf products directly into the final C basis."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < element_count
    if COLUMN_MAJOR:
        columns = offsets // half
        rows = offsets - columns * half
    else:
        rows = offsets // half
        columns = offsets - rows * half
    m1_offsets = rows * stride_m10 + columns * stride_m11
    m2_offsets = rows * stride_m20 + columns * stride_m21
    m3_offsets = rows * stride_m30 + columns * stride_m31
    m4_offsets = rows * stride_m40 + columns * stride_m41
    m5_offsets = rows * stride_m50 + columns * stride_m51
    m6_offsets = rows * stride_m60 + columns * stride_m61
    m7_offsets = rows * stride_m70 + columns * stride_m71
    m1 = tl.load(m1_ptr + m1_offsets, mask=mask)
    m2 = tl.load(m2_ptr + m2_offsets, mask=mask)
    m3 = tl.load(m3_ptr + m3_offsets, mask=mask)
    m4 = tl.load(m4_ptr + m4_offsets, mask=mask)
    m5 = tl.load(m5_ptr + m5_offsets, mask=mask)
    m6 = tl.load(m6_ptr + m6_offsets, mask=mask)
    m7 = tl.load(m7_ptr + m7_offsets, mask=mask)
    c11_offsets = rows * stride_output0 + columns * stride_output1
    c12_offsets = rows * stride_output0 + (columns + half) * stride_output1
    c21_offsets = (rows + half) * stride_output0 + columns * stride_output1
    c22_offsets = (rows + half) * stride_output0 + (columns + half) * stride_output1
    tl.store(output_ptr + c11_offsets, m4 + m5, mask=mask)
    tl.store(output_ptr + c12_offsets, m3 + m5 - m6 - m2, mask=mask)
    tl.store(output_ptr + c21_offsets, m1 - m3 - m5 - m7, mask=mask)
    tl.store(output_ptr + c22_offsets, m1 + m2 - m3 - m5, mask=mask)


@triton.jit
def _prepare_grouped_inputs_kernel(
    a_ptr,
    b_ptr,
    left_ptr,
    right_ptr,
    half,
    stride_a0,
    stride_a1,
    stride_b0,
    stride_b1,
    stride_left_batch,
    stride_left0,
    stride_left1,
    stride_right_batch,
    stride_right0,
    stride_right1,
    COLUMN_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Materialize all seven BAG input pairs for one grouped GEMM launch."""

    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < half * half
    if COLUMN_MAJOR:
        columns = offsets // half
        rows = offsets - columns * half
    else:
        rows = offsets // half
        columns = offsets - rows * half
    a11_offsets = rows * stride_a0 + columns * stride_a1
    a12_offsets = rows * stride_a0 + (columns + half) * stride_a1
    a21_offsets = (rows + half) * stride_a0 + columns * stride_a1
    a22_offsets = (rows + half) * stride_a0 + (columns + half) * stride_a1
    b11_offsets = rows * stride_b0 + columns * stride_b1
    b12_offsets = rows * stride_b0 + (columns + half) * stride_b1
    b21_offsets = (rows + half) * stride_b0 + columns * stride_b1
    b22_offsets = (rows + half) * stride_b0 + (columns + half) * stride_b1
    a11 = tl.load(a_ptr + a11_offsets, mask=mask)
    a12 = tl.load(a_ptr + a12_offsets, mask=mask)
    a21 = tl.load(a_ptr + a21_offsets, mask=mask)
    a22 = tl.load(a_ptr + a22_offsets, mask=mask)
    b11 = tl.load(b_ptr + b11_offsets, mask=mask)
    b12 = tl.load(b_ptr + b12_offsets, mask=mask)
    b21 = tl.load(b_ptr + b21_offsets, mask=mask)
    b22 = tl.load(b_ptr + b22_offsets, mask=mask)
    left_offsets = rows * stride_left0 + columns * stride_left1
    right_offsets = rows * stride_right0 + columns * stride_right1

    # Product order matches BAG's M1..M7 notation.
    tl.store(left_ptr + 0 * stride_left_batch + left_offsets, a12 + a22, mask=mask)
    tl.store(left_ptr + 1 * stride_left_batch + left_offsets, a22 - a21, mask=mask)
    tl.store(
        left_ptr + 2 * stride_left_batch + left_offsets,
        a12 - a21 + a22,
        mask=mask,
    )
    tl.store(
        left_ptr + 5 * stride_left_batch + left_offsets,
        a12 - a21 + a22 - a11,
        mask=mask,
    )
    tl.store(right_ptr + 0 * stride_right_batch + right_offsets, b12 + b22, mask=mask)
    tl.store(right_ptr + 1 * stride_right_batch + right_offsets, b22 - b21, mask=mask)
    tl.store(
        right_ptr + 2 * stride_right_batch + right_offsets,
        b12 - b21 + b22,
        mask=mask,
    )
    tl.store(
        right_ptr + 6 * stride_right_batch + right_offsets,
        b12 - b21 + b22 - b11,
        mask=mask,
    )


@triton.jit
def _grouped_bag_gemm_kernel(
    left_ptr,
    right_ptr,
    left3_ptr,
    left4_ptr,
    left56_ptr,
    right3_ptr,
    right4_ptr,
    right56_ptr,
    products_ptr,
    half,
    stride_left_batch,
    stride_left_m,
    stride_left_k,
    stride_right_batch,
    stride_right_k,
    stride_right_n,
    stride_direct_left_m,
    stride_direct_left_k,
    stride_direct_right_k,
    stride_direct_right_n,
    stride_products_batch,
    stride_products_m,
    stride_products_n,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TRANSPOSE_PROBLEM: tl.constexpr,
):
    """Compute seven BAG products, loading unchanged quadrants in place."""

    program_id = tl.program_id(axis=0)
    product_id = tl.program_id(axis=1)
    num_programs_m = tl.cdiv(half, BLOCK_M)
    num_programs_n = tl.cdiv(half, BLOCK_N)
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
    if product_id == 3:
        left_base = left3_ptr
        current_stride_left_m = stride_direct_left_m
        current_stride_left_k = stride_direct_left_k
        right_base = right3_ptr
        current_stride_right_k = stride_direct_right_k
        current_stride_right_n = stride_direct_right_n
    elif product_id == 4:
        left_base = left4_ptr
        current_stride_left_m = stride_direct_left_m
        current_stride_left_k = stride_direct_left_k
        right_base = right4_ptr
        current_stride_right_k = stride_direct_right_k
        current_stride_right_n = stride_direct_right_n
    elif product_id == 5:
        if TRANSPOSE_PROBLEM:
            left_base = left56_ptr
            current_stride_left_m = stride_direct_left_m
            current_stride_left_k = stride_direct_left_k
            right_base = right_ptr + product_id * stride_right_batch
            current_stride_right_k = stride_right_k
            current_stride_right_n = stride_right_n
        else:
            left_base = left_ptr + product_id * stride_left_batch
            current_stride_left_m = stride_left_m
            current_stride_left_k = stride_left_k
            right_base = right56_ptr
            current_stride_right_k = stride_direct_right_k
            current_stride_right_n = stride_direct_right_n
    elif product_id == 6:
        if TRANSPOSE_PROBLEM:
            left_base = left_ptr + product_id * stride_left_batch
            current_stride_left_m = stride_left_m
            current_stride_left_k = stride_left_k
            right_base = right56_ptr
            current_stride_right_k = stride_direct_right_k
            current_stride_right_n = stride_direct_right_n
        else:
            left_base = left56_ptr
            current_stride_left_m = stride_direct_left_m
            current_stride_left_k = stride_direct_left_k
            right_base = right_ptr + product_id * stride_right_batch
            current_stride_right_k = stride_right_k
            current_stride_right_n = stride_right_n
    else:
        left_base = left_ptr + product_id * stride_left_batch
        current_stride_left_m = stride_left_m
        current_stride_left_k = stride_left_k
        right_base = right_ptr + product_id * stride_right_batch
        current_stride_right_k = stride_right_k
        current_stride_right_n = stride_right_n
    for k_start in range(0, tl.cdiv(half, BLOCK_K)):
        current_k = k_start * BLOCK_K + offsets_k
        left_offsets = (
            offsets_m[:, None] * current_stride_left_m
            + current_k[None, :] * current_stride_left_k
        )
        right_offsets = (
            current_k[:, None] * current_stride_right_k
            + offsets_n[None, :] * current_stride_right_n
        )
        left = tl.load(
            left_base + left_offsets,
            mask=(offsets_m[:, None] < half) & (current_k[None, :] < half),
            other=0.0,
        )
        right = tl.load(
            right_base + right_offsets,
            mask=(current_k[:, None] < half) & (offsets_n[None, :] < half),
            other=0.0,
        )
        accumulator += tl.dot(left, right, input_precision=INPUT_PRECISION)
    product_offsets = (
        product_id * stride_products_batch
        + offsets_m[:, None] * stride_products_m
        + offsets_n[None, :] * stride_products_n
    )
    tl.store(
        products_ptr + product_offsets,
        accumulator,
        mask=(offsets_m[:, None] < half) & (offsets_n[None, :] < half),
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
    *,
    multistream: bool = True,
    full_concurrency: bool = False,
    hybrid: bool = False,
    grouped: Optional[bool] = None,
) -> BagWorkspace:
    """Allocate the precomputed quarter-matrix buffers needed by one BAG level.

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
    if not isinstance(multistream, bool):
        raise TypeError(f"multistream must be bool, got {type(multistream).__name__}")
    if not isinstance(full_concurrency, bool):
        raise TypeError(
            f"full_concurrency must be bool, got {type(full_concurrency).__name__}"
        )
    if not isinstance(hybrid, bool):
        raise TypeError(f"hybrid must be bool, got {type(hybrid).__name__}")
    if grouped is not None and not isinstance(grouped, bool):
        raise TypeError(
            f"grouped must be bool or None, got {type(grouped).__name__}"
        )
    allocate_grouped = size <= 8192 if grouped is None else grouped
    sa = torch.empty_strided((half, half), strides, dtype=torch.float32, device=a.device)
    sb = torch.empty_strided((half, half), strides, dtype=torch.float32, device=a.device)
    ua = None
    ub = None
    if full_concurrency or hybrid:
        ua = torch.empty_strided(
            (half, half), strides, dtype=torch.float32, device=a.device
        )
        ub = torch.empty_strided(
            (half, half), strides, dtype=torch.float32, device=a.device
        )
    leaf_product = None
    if hybrid:
        leaf_product = torch.empty_strided(
            (half, half), strides, dtype=torch.float32, device=a.device
        )
    grouped_left = None
    grouped_right = None
    grouped_products = None
    if allocate_grouped:
        batch_strides = (half * half, strides[0], strides[1])
        grouped_left = torch.empty_strided(
            (7, half, half), batch_strides, dtype=torch.float32, device=a.device
        )
        grouped_right = torch.empty_strided(
            (7, half, half), batch_strides, dtype=torch.float32, device=a.device
        )
        grouped_products = torch.empty_strided(
            (7, half, half), batch_strides, dtype=torch.float32, device=a.device
        )
    streams = None
    if multistream or full_concurrency:
        stream_count = 7 if full_concurrency else 4
        streams = tuple(torch.cuda.Stream(device=a.device) for _ in range(stream_count))
    return BagWorkspace(
        sa,
        sb,
        streams,
        ua,
        ub,
        leaf_product,
        grouped_left,
        grouped_right,
        grouped_products,
    )


def bag_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[BagWorkspace] = None,
    allow_tf32: bool = False,
    schedule: str = "auto",
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
        schedule: ``"serial"`` always submits on the caller's current stream;
            ``"multistream"`` uses two dependency batches for N below 4096;
            ``"seven-stream"`` launches all seven products concurrently using
            independent precomputed inputs and atomic output accumulation;
            ``"hybrid"`` uses Triton transforms around seven serial cuBLAS
            leaf GEMMs as a diagnostic upper-bound implementation;
            ``"grouped"`` materializes seven pairs and executes all seven
            products in one batched Triton launch;
            ``"auto"`` selects grouped execution through N=8192 when grouped
            workspace is supplied, and otherwise uses the measured serial or
            dependency-batched fallback.

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
    if schedule not in (
        "auto", "serial", "multistream", "seven-stream", "hybrid", "grouped"
    ):
        raise ValueError(
            "schedule must be 'auto', 'serial', 'multistream', "
            f"'seven-stream', 'hybrid', or 'grouped', got {schedule!r}"
        )

    device = a.device
    _check_square_matrix("a", a, size, device)
    _check_square_matrix("b", b, size, device)

    if out is None:
        out = torch.empty_like(a, memory_format=torch.preserve_format)
    else:
        _check_square_matrix("out", out, size, device, output=True)

    if workspace is None:
        workspace = allocate_bag_workspace(
            a,
            b,
            out,
            multistream=schedule in ("multistream", "seven-stream"),
            full_concurrency=schedule == "seven-stream",
            hybrid=schedule == "hybrid",
            grouped=schedule == "grouped" or (schedule == "auto" and size <= 8192),
        )
    elif not isinstance(workspace, BagWorkspace):
        try:
            workspace = BagWorkspace(*workspace)
        except (TypeError, ValueError) as error:
            raise TypeError("workspace must be a BagWorkspace or a pair of tensors") from error

    half = size // 2
    sa = workspace.transformed_a12
    sb = workspace.transformed_b12
    _check_square_matrix("workspace.transformed_a12", sa, half, device, output=True)
    _check_square_matrix("workspace.transformed_b12", sb, half, device, output=True)
    worker_streams = workspace.streams
    if worker_streams is not None:
        if len(worker_streams) < 4:
            raise ValueError("multistream workspace must contain at least four CUDA streams")
        if any(stream.device != device for stream in worker_streams):
            raise ValueError("every workspace stream must be on the input device")
    auto_multistream = schedule == "auto" and size == 2048
    forced_multistream = schedule == "multistream" and size < 4096
    use_multistream = (auto_multistream or forced_multistream) and worker_streams is not None
    use_seven_stream = schedule == "seven-stream"
    use_hybrid = schedule == "hybrid"
    grouped_workspace_available = all(
        isinstance(tensor, torch.Tensor)
        for tensor in (
            workspace.grouped_left,
            workspace.grouped_right,
            workspace.grouped_products,
        )
    )
    use_grouped = schedule == "grouped" or (
        schedule == "auto" and size <= 8192 and grouped_workspace_available
    )
    if schedule == "multistream" and size < 4096 and worker_streams is None:
        raise ValueError(
            "schedule='multistream' requires workspace from "
            "allocate_bag_workspace(..., multistream=True)"
        )
    ua = workspace.transformed_a12_m6
    ub = workspace.transformed_b12_m7
    if use_seven_stream:
        if worker_streams is None or len(worker_streams) < 7:
            raise ValueError(
                "schedule='seven-stream' requires seven preallocated streams; use "
                "allocate_bag_workspace(..., full_concurrency=True)"
            )
        if ua is None or ub is None:
            raise ValueError(
                "schedule='seven-stream' requires the full-concurrency workspace"
            )
        _check_square_matrix("workspace.transformed_a12_m6", ua, half, device, output=True)
        _check_square_matrix("workspace.transformed_b12_m7", ub, half, device, output=True)
    leaf_product = workspace.leaf_product
    if use_hybrid:
        if leaf_product is None:
            raise ValueError(
                "schedule='hybrid' requires workspace from "
                "allocate_bag_workspace(..., hybrid=True)"
            )
        _check_square_matrix("workspace.leaf_product", leaf_product, half, device, output=True)
        if ua is None or ub is None:
            raise ValueError("schedule='hybrid' requires four transform buffers")
        _check_square_matrix("workspace.transformed_a12_m6", ua, half, device, output=True)
        _check_square_matrix("workspace.transformed_b12_m7", ub, half, device, output=True)
        if bool(torch.backends.cuda.matmul.allow_tf32) != allow_tf32:
            raise ValueError(
                "schedule='hybrid' requires torch.backends.cuda.matmul.allow_tf32 "
                "to match allow_tf32"
            )
    grouped_left = workspace.grouped_left
    grouped_right = workspace.grouped_right
    grouped_products = workspace.grouped_products
    if use_grouped:
        grouped_tensors = [
            ("workspace.grouped_left", grouped_left),
            ("workspace.grouped_right", grouped_right),
        ]
        grouped_tensors.append(("workspace.grouped_products", grouped_products))
        for name, tensor in grouped_tensors:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f"schedule={schedule!r} requires workspace from "
                    "allocate_bag_workspace(..., grouped=True)"
                )
            if tuple(tensor.shape) != (7, half, half):
                raise ValueError(f"{name} must have shape (7, {half}, {half})")
            if tensor.device != device or tensor.dtype != torch.float32:
                raise ValueError(f"{name} must be an FP32 tensor on {device}")

    element_count = half * half
    block = 256
    layout_matrices = (a, b, out, sa, sb)
    if use_seven_stream or use_hybrid:
        layout_matrices += (ua, ub)
    column_major = all(_is_column_major(matrix) for matrix in layout_matrices)
    if use_seven_stream or use_hybrid:
        _prepare_full_concurrency_kernel[(triton.cdiv(element_count, block),)](
            a,
            b,
            out,
            sa,
            sb,
            ua,
            ub,
            half,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            sa.stride(0), sa.stride(1),
            sb.stride(0), sb.stride(1),
            ua.stride(0), ua.stride(1),
            ub.stride(0), ub.stride(1),
            ZERO_OUTPUT=use_seven_stream,
            COLUMN_MAJOR=column_major,
            BLOCK=block,
            num_warps=4,
        )
    elif not use_grouped:
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
            COLUMN_MAJOR=column_major,
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

    if use_grouped:
        _prepare_grouped_inputs_kernel[(triton.cdiv(element_count, block),)](
            a,
            b,
            grouped_left,
            grouped_right,
            half,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            grouped_left.stride(0), grouped_left.stride(1), grouped_left.stride(2),
            grouped_right.stride(0), grouped_right.stride(1), grouped_right.stride(2),
            COLUMN_MAJOR=column_major,
            BLOCK=block,
            num_warps=4,
        )

    if use_grouped:
        if half <= 64:
            block_m, block_n, block_k = 32, 32, 16
            num_warps, num_stages = 4, 3
        elif half <= 256:
            block_m, block_n, block_k = 64, 64, 32
            num_warps, num_stages = 4, 4
        elif half <= 512:
            block_m, block_n = 64, 64
            block_k = 16 if allow_tf32 else 32
            num_warps, num_stages = 4, 3
        elif half <= 1024:
            block_m, block_n = 64, 64
            block_k = 16 if allow_tf32 else 32
            num_warps = 4
            num_stages = 3 if allow_tf32 else 2
        else:
            block_m, block_n = 128, 128
            block_k = 16 if allow_tf32 else 32
            num_warps = 8
            num_stages = 3 if half <= 2048 else 4
        if column_major:
            kernel_left = grouped_right
            kernel_right = grouped_left
            kernel_left3, kernel_left4, kernel_left56 = b11, b21, b12
            kernel_right3, kernel_right4, kernel_right56 = a11, a12, a21
            stride_left_m, stride_left_k = grouped_right.stride(2), grouped_right.stride(1)
            stride_right_k, stride_right_n = grouped_left.stride(2), grouped_left.stride(1)
            stride_direct_left_m, stride_direct_left_k = b.stride(1), b.stride(0)
            stride_direct_right_k, stride_direct_right_n = a.stride(1), a.stride(0)
            stride_products_m = grouped_products.stride(2)
            stride_products_n = grouped_products.stride(1)
        else:
            kernel_left = grouped_left
            kernel_right = grouped_right
            kernel_left3, kernel_left4, kernel_left56 = a11, a12, a21
            kernel_right3, kernel_right4, kernel_right56 = b11, b21, b12
            stride_left_m, stride_left_k = grouped_left.stride(1), grouped_left.stride(2)
            stride_right_k, stride_right_n = grouped_right.stride(1), grouped_right.stride(2)
            stride_direct_left_m, stride_direct_left_k = a.stride(0), a.stride(1)
            stride_direct_right_k, stride_direct_right_n = b.stride(0), b.stride(1)
            stride_products_m = grouped_products.stride(1)
            stride_products_n = grouped_products.stride(2)
        grouped_grid = (
            triton.cdiv(half, block_m) * triton.cdiv(half, block_n),
            7,
        )
        _grouped_bag_gemm_kernel[grouped_grid](
            kernel_left,
            kernel_right,
            kernel_left3,
            kernel_left4,
            kernel_left56,
            kernel_right3,
            kernel_right4,
            kernel_right56,
            grouped_products,
            half,
            kernel_left.stride(0),
            stride_left_m,
            stride_left_k,
            kernel_right.stride(0),
            stride_right_k,
            stride_right_n,
            stride_direct_left_m,
            stride_direct_left_k,
            stride_direct_right_k,
            stride_direct_right_n,
            grouped_products.stride(0),
            stride_products_m,
            stride_products_n,
            INPUT_PRECISION="tf32" if allow_tf32 else "ieee",
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            TRANSPOSE_PROBLEM=column_major,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        products = tuple(grouped_products[index] for index in range(7))
        _decode_hybrid_products_kernel[(triton.cdiv(element_count, block),)](
            products[0], products[1], products[2], products[3],
            products[4], products[5], products[6], out,
            element_count,
            half,
            products[0].stride(0), products[0].stride(1),
            products[1].stride(0), products[1].stride(1),
            products[2].stride(0), products[2].stride(1),
            products[3].stride(0), products[3].stride(1),
            products[4].stride(0), products[4].stride(1),
            products[5].stride(0), products[5].stride(1),
            products[6].stride(0), products[6].stride(1),
            out.stride(0), out.stride(1),
            COLUMN_MAJOR=column_major,
            BLOCK=block,
            num_warps=4,
        )
        return out

    if use_hybrid:
        # Store all seven products in four output blocks and three buffers,
        # then decode directly into the final basis in one bandwidth pass.
        torch.mm(sa, sb, out=c_psi11)       # M3
        torch.mm(ua, b12, out=c_psi12)      # M6
        torch.mm(a21, ub, out=c_psi21)      # M7
        _prepare_outer_product_pairs_kernel[(triton.cdiv(element_count, block),)](
            a,
            b,
            sa,
            sb,
            ua,
            ub,
            half,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            sa.stride(0), sa.stride(1),
            sb.stride(0), sb.stride(1),
            ua.stride(0), ua.stride(1),
            ub.stride(0), ub.stride(1),
            COLUMN_MAJOR=column_major,
            BLOCK=block,
            num_warps=4,
        )
        torch.mm(sa, sb, out=c_psi22)       # M1
        torch.mm(ua, ub, out=leaf_product)  # M2
        torch.mm(a11, b11, out=sa)          # M4
        torch.mm(a12, b21, out=sb)          # M5
        _decode_hybrid_products_kernel[(triton.cdiv(element_count, block),)](
            c_psi22,
            leaf_product,
            c_psi11,
            sa,
            sb,
            c_psi12,
            c_psi21,
            out,
            element_count,
            half,
            c_psi22.stride(0), c_psi22.stride(1),
            leaf_product.stride(0), leaf_product.stride(1),
            c_psi11.stride(0), c_psi11.stride(1),
            sa.stride(0), sa.stride(1),
            sb.stride(0), sb.stride(1),
            c_psi12.stride(0), c_psi12.stride(1),
            c_psi21.stride(0), c_psi21.stride(1),
            out.stride(0), out.stride(1),
            COLUMN_MAJOR=column_major,
            BLOCK=block,
            num_warps=4,
        )
        return out

    if use_seven_stream:
        caller_stream = torch.cuda.current_stream(device=device)
        streams = worker_streams[:7]
        for stream in streams:
            stream.wait_stream(caller_stream)
        launches = (
            (a12, a22, b12, b22, c_psi22, None, 1, 1, 1, 0),
            (a22, a21, b22, b21, c_psi21, None, -1, -1, 1, 0),
            (sa, None, sb, None, c_psi12, None, 0, 0, 1, 0),
            (a11, None, b11, None, c_psi11, None, 0, 0, 1, 0),
            (a12, None, b21, None, c_psi11, c_psi12, 0, 0, 1, 1),
            (ua, None, b12, None, c_psi12, c_psi22, 0, 0, -1, -1),
            (a21, None, ub, None, c_psi12, c_psi21, 0, 0, 1, 1),
        )
        for stream, launch in zip(streams, launches):
            x, y, v, w, d, e, delta, epsilon, gamma_d, gamma_e = launch
            with torch.cuda.stream(stream):
                bag_primitive(
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
                    allow_tf32=allow_tf32,
                    _atomic_accumulate=True,
                )
        for stream in streams:
            caller_stream.wait_stream(stream)
        _inverse_basis_transform_kernel[(triton.cdiv(element_count, block),)](
            out,
            half,
            out.stride(0),
            out.stride(1),
            COLUMN_MAJOR=column_major,
            BLOCK=block,
            num_warps=4,
        )
        return out

    if use_multistream:
        # Batch 1 has four independent output blocks.  Every worker waits for
        # the caller-stream preparation kernel, preserving arbitrary upstream
        # producer dependencies without a host synchronization.
        caller_stream = torch.cuda.current_stream(device=device)
        stream_m1, stream_m2, stream_m3, stream_m4 = worker_streams[:4]
        for stream in (stream_m1, stream_m2, stream_m3, stream_m4):
            stream.wait_stream(caller_stream)
        with torch.cuda.stream(stream_m1):
            bag_primitive(
                a12, a22, b12, b22, c_psi22, None,
                delta=1, epsilon=1, gamma_d=2, gamma_e=0,
                allow_tf32=allow_tf32, _atomic_accumulate=False,
            )
        with torch.cuda.stream(stream_m2):
            bag_primitive(
                a22, a21, b22, b21, c_psi21, None,
                delta=-1, epsilon=-1, gamma_d=2, gamma_e=0,
                allow_tf32=allow_tf32, _atomic_accumulate=False,
            )
        with torch.cuda.stream(stream_m3):
            bag_primitive(
                sa, None, sb, None, c_psi12, None,
                delta=0, epsilon=0, gamma_d=2, gamma_e=0,
                allow_tf32=allow_tf32, _atomic_accumulate=False,
            )
            _complete_four_term_blocks_kernel[(triton.cdiv(element_count, block),)](
                a,
                b,
                sa,
                sb,
                half,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                sa.stride(0), sa.stride(1),
                sb.stride(0), sb.stride(1),
                COLUMN_MAJOR=column_major,
                BLOCK=block,
                num_warps=4,
            )
        with torch.cuda.stream(stream_m4):
            bag_primitive(
                a11, None, b11, None, c_psi11, None,
                delta=0, epsilon=0, gamma_d=2, gamma_e=0,
                allow_tf32=allow_tf32, _atomic_accumulate=False,
            )

        # Join batch 1 on the caller stream, then make the three workers wait
        # for that join before atomically accumulating the overlapping batch.
        for stream in (stream_m1, stream_m2, stream_m3, stream_m4):
            caller_stream.wait_stream(stream)
        for stream in (stream_m1, stream_m2, stream_m3):
            stream.wait_stream(caller_stream)
        with torch.cuda.stream(stream_m1):
            bag_primitive(
                a12, None, b21, None, c_psi11, c_psi12,
                delta=0, epsilon=0, gamma_d=1, gamma_e=1,
                allow_tf32=allow_tf32, _atomic_accumulate=True,
            )
        with torch.cuda.stream(stream_m2):
            bag_primitive(
                sa, None, b12, None, c_psi12, c_psi22,
                delta=0, epsilon=0, gamma_d=-1, gamma_e=-1,
                allow_tf32=allow_tf32, _atomic_accumulate=True,
            )
        with torch.cuda.stream(stream_m3):
            bag_primitive(
                a21, None, sb, None, c_psi12, c_psi21,
                delta=0, epsilon=0, gamma_d=1, gamma_e=1,
                allow_tf32=allow_tf32, _atomic_accumulate=True,
            )
        for stream in (stream_m1, stream_m2, stream_m3):
            caller_stream.wait_stream(stream)
        _inverse_basis_transform_kernel[(triton.cdiv(element_count, block),)](
            out,
            half,
            out.stride(0),
            out.stride(1),
            COLUMN_MAJOR=column_major,
            BLOCK=block,
            num_warps=4,
        )
        return out

    materialize_outer_products = half >= 2048
    if not materialize_outer_products:
        # At smaller sizes, fusing each combination avoids two extra launches.
        bag_primitive(
            a12, a22, b12, b22, c_psi22, None,
            delta=1, epsilon=1, gamma_d=2, gamma_e=0, allow_tf32=allow_tf32,
            _atomic_accumulate=False,
        )
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
        COLUMN_MAJOR=column_major,
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
    # M6=SA'*B12, where SA' now stores SA-A11.  On the large materialized
    # path this product initializes Cpsi22 with -M6 before M1 is added.
    bag_primitive(
        sa, None, b12, None, c_psi12, c_psi22,
        delta=0,
        epsilon=0,
        gamma_d=-1,
        gamma_e=-2 if materialize_outer_products else -1,
        allow_tf32=allow_tf32,
        _atomic_accumulate=False,
    )
    # M7=A21*SB'.  The large path initializes Cpsi21 with M7 before M2.
    bag_primitive(
        a21, None, sb, None, c_psi12, c_psi21,
        delta=0,
        epsilon=0,
        gamma_d=1,
        gamma_e=2 if materialize_outer_products else 1,
        allow_tf32=allow_tf32,
        _atomic_accumulate=False,
    )

    if materialize_outer_products:
        # SA/SB are dead after M6/M7.  Reuse them to materialize M1 and M2,
        # trading two coalesced linear passes for fewer loads in O(N^3) loops.
        _prepare_two_term_product_kernel[(triton.cdiv(element_count, block),)](
            a12,
            a22,
            b12,
            b22,
            sa,
            sb,
            half,
            a12.stride(0), a12.stride(1),
            a22.stride(0), a22.stride(1),
            b12.stride(0), b12.stride(1),
            b22.stride(0), b22.stride(1),
            sa.stride(0), sa.stride(1),
            sb.stride(0), sb.stride(1),
            SIGN=1,
            COLUMN_MAJOR=column_major,
            BLOCK=block,
            num_warps=4,
        )
        bag_primitive(
            sa, None, sb, None, c_psi22, None,
            delta=0, epsilon=0, gamma_d=1, gamma_e=0, allow_tf32=allow_tf32,
            _atomic_accumulate=False,
        )
        _prepare_two_term_product_kernel[(triton.cdiv(element_count, block),)](
            a22,
            a21,
            b22,
            b21,
            sa,
            sb,
            half,
            a22.stride(0), a22.stride(1),
            a21.stride(0), a21.stride(1),
            b22.stride(0), b22.stride(1),
            b21.stride(0), b21.stride(1),
            sa.stride(0), sa.stride(1),
            sb.stride(0), sb.stride(1),
            SIGN=-1,
            COLUMN_MAJOR=column_major,
            BLOCK=block,
            num_warps=4,
        )
        bag_primitive(
            sa, None, sb, None, c_psi21, None,
            delta=0, epsilon=0, gamma_d=1, gamma_e=0, allow_tf32=allow_tf32,
            _atomic_accumulate=False,
        )

    _inverse_basis_transform_kernel[(triton.cdiv(element_count, block),)](
        out,
        half,
        out.stride(0),
        out.stride(1),
        COLUMN_MAJOR=column_major,
        BLOCK=block,
        num_warps=4,
    )
    return out
