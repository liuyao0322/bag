# BAG: Alternative Basis Algorithm Implementation on GPUs

## Project Overview

**Alternative Basis Matrix Multiplication (ABMM)** is a fast matrix multiplication algorithm that accelerates matrix multiplication through basis transformations. **BAG (Basis Alternative Algorithm on GPUs)** is a GPU implementation of this fast matrix multiplication algorithm.

* **Strassen Algorithm** (including the Winograd variant): reduces the computational complexity of matrix multiplication from (O(n^3)) to (O(n^{2.81})) through recursive divide-and-conquer.
* **BAG Algorithm**: further improves matrix multiplication performance through basis transformation techniques.
---

# Core Algorithm

## Strassen Algorithm (Winograd Variant)

The Strassen algorithm partitions matrices into (2\times2) submatrices and replaces the conventional 8 matrix multiplications with 7 recursive multiplications.

```
For C = A × B, partition:

A = [A11  A12]    B = [B11  B12]    C = [C11  C12]
    [A21  A22]        [B21  B22]        [C21  C22]


Winograd variant computes 7 products:

S1 = A21 + A22        T1 = B12 - B11
S2 = S1 - A11         T2 = B22 - T1
S3 = A11 - A21        T3 = B22 - B12
S4 = A12 - S2         T4 = B21 - T2

P1 = A11 * B11        C11 = P1 + P2
P2 = A12 * B21        C12 = P1 + P3 + P5 + P6
P3 = S1 * T1          C21 = P1 + P3 + P4 - P7
P4 = S2 * T2          C22 = P1 + P3 + P4 + P5
P5 = S3 * T3
P6 = S4 * B22
P7 = A22 * T4
```

---

# BAG Algorithm

The **BAG algorithm** transforms matrix multiplication into a more efficient computational form through basis transformations. The key idea is to perform matrix multiplication in a transformed basis, reducing the number of expensive multiplication operations.

## Recursive Algorithm Structure

BAG is a recursive algorithm where **each recursion level performs a complete basis transformation and inverse transformation**.

```
bag(A, B, C, depth):

    if depth == 0:
        C += A * B        # call cuBLAS GEMM

    else:
        tA, tB = copy(A), copy(B)

        basis_trans(tA, tB)          # forward basis transformation

        Partition tA and tB into 2×2 sub-blocks

        Recursively compute 7 sub-products:
            bag(subA_i, subB_i, depth-1)

        basis_trans_inverse(C)       # inverse basis transformation
```

### Important Observation

Each recursion level independently performs a full basis transformation and inverse transformation, rather than applying the transformation only at the top level.

Therefore:

* **depth = 1**:

  * Apply basis transformation once on the original matrix.
  * Compute 7 leaf GEMMs using cuBLAS.

* **depth = 2**:

  * Apply basis transformation on the original matrix.
  * Each of the 7 recursive subproblems performs another basis transformation.

* **depth = n**:

  * The total number of basis transformations is:

[
\frac{7^n-1}{6}
]

---

# Step 1: Forward Basis Transformation

Given input matrices (A) and (B), BAG applies basis transformations:

[
\phi(A), \phi(B)
]

## Basis Transformation of Matrix A

[
A =
\begin{bmatrix}
A_{11} & A_{12}\
A_{21} & A_{22}
\end{bmatrix}
\rightarrow
A^\psi =
\begin{bmatrix}
A^\psi_{11} & A^\psi_{12}\
A^\psi_{21} & A^\psi_{22}
\end{bmatrix}
]

where:

* (A^\psi_{11}=A_{11})

* (A^\psi_{12}=A_{12}-A_{21}+A_{22})

* (A^\psi_{21}=A_{22}-A_{21})

* (A^\psi_{22}=A_{12}+A_{22})

## Basis Transformation of Matrix B

[
B =
\begin{bmatrix}
B_{11} & B_{12}\
B_{21} & B_{22}
\end{bmatrix}
\rightarrow
B^\phi
]

where:

* (B^\phi_{11}=B_{11})

* (B^\phi_{12}=B_{12}-B_{21}+B_{22})

* (B^\phi_{21}=B_{22}-B_{21})

* (B^\phi_{22}=B_{12}+B_{22})

---

# Step 2: Matrix Multiplication in the Transformed Basis

Seven matrix multiplications are performed (same multiplication count as Strassen).

## Basic Products (M1-M4)

[
\begin{aligned}
M_1 &= A^\psi_{22}\times B^\phi_{22}
&\Rightarrow C^\psi_{22} &+=M_1
\
M_2 &= A^\psi_{21}\times B^\phi_{21}
&\Rightarrow C^\psi_{21} &+=M_2
\
M_3 &= A^\psi_{12}\times B^\phi_{12}
&\Rightarrow C^\psi_{12} &+=M_3
\
M_4 &= A^\psi_{11}\times B^\phi_{11}
&\Rightarrow C^\psi_{11} &+=M_4
\end{aligned}
]

## Combined Products (M5-M7)

[
\begin{aligned}
M_5 &=
(A^\psi_{12}-A^\psi_{21})
(B^\phi_{22}-B^\phi_{12})
\
&\Rightarrow
C^\psi_{11}+=M_5,
C^\psi_{12}+=M_5
\end{aligned}
]

[
\begin{aligned}
M_6 &=
(A^\psi_{12}-A^\psi_{11})
(B^\phi_{12}-B^\phi_{21})
\
&\Rightarrow
C^\psi_{12}-=M_6,
C^\psi_{22}-=M_6
\end{aligned}
]

[
\begin{aligned}
M_7 &=
(A^\psi_{22}-A^\psi_{12})
(B^\phi_{12}-B^\phi_{11})
\
&\Rightarrow
C^\psi_{12}+=M_7,
C^\psi_{21}+=M_7
\end{aligned}
]

## Decoding Step

The seven products are combined to reconstruct the transformed output:

[
\begin{aligned}
C^\psi_{11}&=M_4+M_5\
C^\psi_{12}&=M_3+M_5-M_6+M_7\
C^\psi_{21}&=M_2+M_7\
C^\psi_{22}&=M_1-M_6
\end{aligned}
]

---

# Step 3: Inverse Basis Transformation

The transformed result (C^\psi) is converted back to the original basis:

[
C^\psi
\rightarrow C
]

where:

* (C_{11}=C^\psi_{11})

* (C_{12}=C^\psi_{12}-C^\psi_{21})

* (C_{21}=-C^\psi_{12}+C^\psi_{22})

* (C_{22}=-C^\psi_{12}+C^\psi_{21}+C^\psi_{22})

---

# Implementation Details

## Basis Transformation (`basis_trans`)

The implementation performs the following steps:

1. Allocate temporary buffers to preserve original submatrices.
2. Use cuBLAS `cublasSgeam` to perform matrix addition/subtraction operations.
3. Update matrices in-place to reduce memory overhead.

---

# Triton Fused Primitive

The Triton implementation is available in `src/bag_triton/primitive.py`.  It
implements the fused BAG primitive without materializing either input linear
combination:

```text
P = (X + delta * Y) @ (V + epsilon * W)
D += gamma_d * P
E += gamma_e * P
```

`delta` and `epsilon` are compile-time specializations in `{-1, 0, 1}`.  For
compatibility with the CUDA kernels, output modes `gamma_d` and `gamma_e` use
the following encoding:

* `0`: skip the output entirely;
* `1` / `-1`: atomically accumulate `+P` / `-P`;
* `2` / `-2`: directly overwrite the output with `+P` / `-P`.

The public `bag_primitive` wrapper accepts FP32 CUDA tensors with row-major,
column-major, or sliced positive strides.  Boundary masks support dimensions
that are not multiples of a kernel tile.  IEEE FP32 dot products are used by
default; pass `allow_tf32=True` to use TF32 tensor-core products.

Example:

```python
from bag_triton import bag_primitive

bag_primitive(
    x, y, v, w, d, e,
    delta=-1,
    epsilon=1,
    gamma_d=1,
    gamma_e=-1,
)
```

Run the GPU correctness suite from the repository root with:

```bash
PYTHONPATH=src python scripts/test_triton_primitive.py
```

---

## One-level Triton BAG matrix multiplication

`src/bag_triton/bag.py` assembles a complete depth-1 BAG operation from seven
primitive products.  It avoids full transformed copies of A and B by
materializing only two quarter-matrix blocks:

```text
SA = A12 - A21 + A22
SB = B12 - B21 + B22
```

M3 consumes `SA` and `SB`.  The same buffers are then updated in place to
`SA-A11` and `SB-B11` for M6/M7, and the large-size path reuses them once more
to materialize M1/M2 inputs.  Total workspace is therefore `N^2/2` FP32
elements instead of two full transformed matrices.

The auto scheduler uses two dependency-batched multi-stream waves at N=2048:
M1-M4 first on four streams, then atomic M5-M7 on three streams.  Other sizes
use the serial schedule selected by measurement.  All workers wait on the
caller's current stream before reading inputs, and the inverse transform is
joined back onto that caller stream.

Create output, workspace, and streams outside repeated or timed calls:

```python
import torch
from bag_triton import allocate_bag_workspace, bag_matmul

a = torch.randn((2048, 2048), device="cuda", dtype=torch.float32)
b = torch.randn_like(a)
out = torch.empty_like(a)

# Allocates two quarter-matrix buffers and four reusable CUDA streams.
workspace = allocate_bag_workspace(a, b, out, multistream=True)

# No allocation or stream construction occurs in this call.
bag_matmul(
    a,
    b,
    out=out,
    workspace=workspace,
    allow_tf32=False,
    schedule="auto",
)
```

Run correctness and the end-to-end BAG-vs-cuBLAS benchmark with:

```bash
PYTHONPATH=src python scripts/test_triton_bag.py
PYTHONPATH=src python scripts/benchmark_triton_bag.py --output bag-results.csv
```

`speedup` in the CSV is `ordinary_GEMM_ms / BAG_ms`; a value above one means
BAG is faster.  The repository result report is
`benchmark_results/bag_vs_gemm_rtx3090.md`.

---

## Inverse Transformation (`basis_trans_inverse`)

The inverse transformation:

1. Saves transformed submatrices into temporary buffers.
2. Restores the original representation according to inverse transformation formulas.
3. Releases temporary workspace.

---

# Challenges and Optimization Directions for High-Performance BAG Implementation

The main challenge of implementing BAG efficiently on GPUs is not the recursive structure itself, but reducing the overhead introduced by basis transformations, temporary storage, and recursive scheduling while preserving the theoretical advantage of seven recursive multiplications.

The major challenges and optimization directions are summarized below.

---

## Key Challenges

### 1. Controlling Recursive Overhead

Since every recursion level performs:

* forward basis transformation,
* seven recursive multiplications,
* inverse basis transformation,

the practical runtime depends not only on reduced multiplication complexity, but also heavily on:

* transformation overhead,
* kernel launch latency,
* synchronization cost,
* recursive task management.

A key challenge is preventing these additional costs from eliminating the theoretical advantage.

---

### 2. Memory Bandwidth Bottleneck During Basis Transformation

Compared with GEMM operations, basis transformations mainly consist of matrix additions, subtractions, and data rearrangements.

Therefore, they are more likely to become bandwidth-bound.

The following factors significantly affect performance:

* memory layout of sub-blocks,
* temporary buffer reuse,
* unnecessary intermediate memory writes,
* data movement overhead.

---

### 3. Balancing Recursive Depth and Leaf Matrix Size

Increasing recursion depth reduces multiplication operations but rapidly decreases the size of leaf GEMM problems.

When leaf matrices become too small:

* cuBLAS cannot fully utilize Tensor Cores,
* GPU occupancy decreases,
* scheduling and memory overhead dominate execution time.

Therefore, choosing an appropriate recursion depth and leaf size is critical.
