#ifndef PRIM_INTER_CUH
#define PRIM_INTER_CUH

#include "prim_small.cuh"
#include "prim_medium.cuh"
#include "prim_large.cuh"
#include "prim_huge.cuh"
#include "prim_tall.cuh"
#include "prim_wide.cuh"
#include <cuda_runtime.h>

/*
 * prim_inter: Intelligent primitive selector
 * 
 * Selection strategy (inspired by Cutlass):
 * 1. Shape-based selection (for non-square matrices):
 *    - prim_tall (128×32): When M/N > 2.0, optimized for tall matrices
 *    - prim_wide (32×128): When N/M > 2.0, optimized for wide matrices
 * 
 * 2. Size-based selection (for square-ish matrices):
 *    - prim_small (16×16, 64 threads): M,N ≤ 256, minimal resource usage
 *    - prim_medium (32×32, 64 threads): 256 < M,N ≤ 1024, balanced performance
 *    - prim_large (64×64, 64 threads): 1024 < M,N ≤ 4096, high occupancy
 *    - prim_huge (128×128, 256 threads): M,N > 4096, maximum throughput
 * 
 * Priority: Shape characteristics > Absolute size
 */

// Helper macro to call the appropriate kernel
#define LAUNCH_PRIM_KERNEL(kernel_name, delta_v, epsilon_v, gamma0_v, gamma1_v) \
    kernel_name<delta_v, epsilon_v, gamma0_v, gamma1_v><<<grid, block, 0, stream>>>( \
        X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde)

inline void prim_inter(
    const float* X, const float* Y,
    const float* V, const float* W,
    float* D, float* E,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde,
    int delta, int epsilon, int gamma0, int gamma1,
    cudaStream_t stream);

template<int delta, int epsilon, int gamma0, int gamma1>
inline void prim_inter_templated(
    const float* X, const float* Y,
    const float* V, const float* W,
    float* D, float* E,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde,
    cudaStream_t stream = 0
) {
    // Calculate matrix shape ratio
    float m_n_ratio = (float)M / (float)N;
    float n_m_ratio = (float)N / (float)M;
    
    // Select strategy based on matrix dimensions
    // Note: grid.x corresponds to M direction (rows), grid.y corresponds to N direction (cols)
    // because kernels use: X += blockIdx.x * tile_M, V += blockIdx.y * tile_N
    if (m_n_ratio > 2.0f) {
        // Tall matrix: M >> N, use primitive_tall (128x32)
        dim3 grid((M + 127) / 128, (N + 31) / 32);
        dim3 block(128);
        LAUNCH_PRIM_KERNEL(primitive_tall, delta, epsilon, gamma0, gamma1);
    } 
    else if (n_m_ratio > 2.0f) {
        // Wide matrix: N >> M, use primitive_wide (32x128)
        dim3 grid((M + 31) / 32, (N + 127) / 128);
        dim3 block(128);
        LAUNCH_PRIM_KERNEL(primitive_wide, delta, epsilon, gamma0, gamma1);
    }
    else if (M <= 384 && N <= 384) {
        // Small matrix: use primitive_small (16x16)
        dim3 grid((M + 15) / 16, (N + 15) / 16);
        dim3 block(64);
        LAUNCH_PRIM_KERNEL(primitive_small, delta, epsilon, gamma0, gamma1);
    }
    else if (M == 1024 && N == 1024 && K == 1024) {
        // Tuned 1024 square case: prim_large is ~6% faster than prim_medium on A800.
        dim3 grid((M + 63) / 64, (N + 63) / 64);
        dim3 block(64);
        LAUNCH_PRIM_KERNEL(primitive_large, delta, epsilon, gamma0, gamma1);
    }
    else if (M <= 1024 && N <= 1024) {
        // Medium matrix: use primitive_medium (32x32)
        dim3 grid((M + 31) / 32, (N + 31) / 32);
        dim3 block(64);
        LAUNCH_PRIM_KERNEL(primitive_medium, delta, epsilon, gamma0, gamma1);
    }
    else if (M <= 4096 && N <= 4096) {
        // Large matrix: use primitive_large (64x64)
        dim3 grid((M + 63) / 64, (N + 63) / 64);
        dim3 block(64);
        LAUNCH_PRIM_KERNEL(primitive_large, delta, epsilon, gamma0, gamma1);
    }
    else {
        // Huge matrix: use primitive_huge (128x128)
        dim3 grid((M + 127) / 128, (N + 127) / 128);
        dim3 block(256);
        LAUNCH_PRIM_KERNEL(primitive_huge, delta, epsilon, gamma0, gamma1);
    }
}

inline void prim_inter_batched_pair_same_variant(
    const PrimBatchedPairArgs& args,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde,
    int delta, int epsilon, int gamma0, int gamma1,
    cudaStream_t stream = 0
) {
    if (delta == -1 && epsilon == -1 && gamma0 == 1 && gamma1 == 1 &&
        M <= 1024 && N <= 1024 && M > 384 && N > 384) {
        prim_inter_batched_pair_same_variant_medium(args, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
        return;
    }

    prim_inter(args.X0, args.Y0, args.V0, args.W0, args.D0, args.E0,
               M, N, K, ldx, ldy, ldv, ldw, ldd, lde,
               delta, epsilon, gamma0, gamma1, stream);
    prim_inter(args.X1, args.Y1, args.V1, args.W1, args.D1, args.E1,
               M, N, K, ldx, ldy, ldv, ldw, ldd, lde,
               delta, epsilon, gamma0, gamma1, stream);
}

// Runtime dispatch based on template parameters
inline void prim_inter(
    const float* X, const float* Y,
    const float* V, const float* W,
    float* D, float* E,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde,
    int delta, int epsilon, int gamma0, int gamma1,
    cudaStream_t stream = 0
) {
    if (gamma0 == 2) gamma0 = 1;
    else if (gamma0 == -2) gamma0 = -1;
    if (gamma1 == 2) gamma1 = 1;
    else if (gamma1 == -2) gamma1 = -1;

    // Dispatch to templated version based on parameter values
    // This uses compile-time optimization for common cases
    if (delta == 1 && epsilon == 1 && gamma0 == 1 && gamma1 == 1) {
        prim_inter_templated<1, 1, 1, 1>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
    }
    else if (delta == 1 && epsilon == 0 && gamma0 == 1 && gamma1 == -1) {
        prim_inter_templated<1, 0, 1, -1>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
    }
    else if (delta == -1 && epsilon == 1 && gamma0 == 0 && gamma1 == 1) {
        prim_inter_templated<-1, 1, 0, 1>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
    }
    else if (delta == -1 && epsilon == 1 && gamma0 == 1 && gamma1 == 0) {
        prim_inter_templated<-1, 1, 1, 0>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
    }
    else if (delta == 0 && epsilon == -1 && gamma0 == 1 && gamma1 == 1) {
        prim_inter_templated<0, -1, 1, 1>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
    }
    else if (delta == 1 && epsilon == 1 && gamma0 == 1 && gamma1 == 0) {
        prim_inter_templated<1, 1, 1, 0>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
    }
    else if (delta == 0 && epsilon == 1 && gamma0 == 1 && gamma1 == 1) {
        prim_inter_templated<0, 1, 1, 1>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
    }
    else if (delta == 1 && epsilon == 0 && gamma0 == 1 && gamma1 == 1) {
        prim_inter_templated<1, 0, 1, 1>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
    }
    else if (delta == -1 && epsilon == -1 && gamma0 == 1 && gamma1 == 1) {
        prim_inter_templated<-1, -1, 1, 1>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
    }
    else if (delta == -1 && epsilon == -1 && gamma0 == -1 && gamma1 == -1) {
        prim_inter_templated<-1, -1, -1, -1>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
    }
    else if (delta == 0 && epsilon == 0 && gamma0 == 1 && gamma1 == 0) {
        prim_inter_templated<0, 0, 1, 0>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
    }
    else {
        // Default case - use most common variant
        prim_inter_templated<1, 1, 1, 1>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde, stream);
    }
}

#undef LAUNCH_PRIM_KERNEL

#endif // PRIM_INTER_CUH