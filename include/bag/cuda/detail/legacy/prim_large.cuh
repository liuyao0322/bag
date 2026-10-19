#ifndef PRIM_LARGE_CUH
#define PRIM_LARGE_CUH

#include "base_components.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void cp_async_ca_shared_global_16(void* dst_shared, const void* src_global) {
    const unsigned shared_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst_shared));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(shared_addr), "l"(src_global));
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group0() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}
#else
__device__ __forceinline__ void cp_async_ca_shared_global_16(void* dst_shared, const void* src_global) {
    *reinterpret_cast<float4*>(dst_shared) = *reinterpret_cast<const float4*>(src_global);
}

__device__ __forceinline__ void cp_async_commit_group() {}
__device__ __forceinline__ void cp_async_wait_group0() {}
#endif

__device__ __forceinline__ void load_raw_large_tile_async(
    float* sX, float* sY, float* sV, float* sW,
    const float* X, const float* Y, const float* V, const float* W,
    int ldx, int ldy, int ldv, int ldw, int tx) {
    const int ms = 64;
    const int ks = 8;
    const int load_row_A = (tx % 8) * 8;
    const int load_col_A = tx / 8;
    const int load_col_B = tx;

    cp_async_ca_shared_global_16(sX + load_col_A * ms + load_row_A,
                                 X + load_row_A + load_col_A * ldx);
    cp_async_ca_shared_global_16(sX + load_col_A * ms + load_row_A + 4,
                                 X + load_row_A + load_col_A * ldx + 4);
    cp_async_ca_shared_global_16(sY + load_col_A * ms + load_row_A,
                                 Y + load_row_A + load_col_A * ldy);
    cp_async_ca_shared_global_16(sY + load_col_A * ms + load_row_A + 4,
                                 Y + load_row_A + load_col_A * ldy + 4);

    cp_async_ca_shared_global_16(sV + load_col_B * ks,
                                 V + load_col_B * ldv);
    cp_async_ca_shared_global_16(sV + load_col_B * ks + 4,
                                 V + load_col_B * ldv + 4);
    cp_async_ca_shared_global_16(sW + load_col_B * ks,
                                 W + load_col_B * ldw);
    cp_async_ca_shared_global_16(sW + load_col_B * ks + 4,
                                 W + load_col_B * ldw + 4);
}
        
/*
    M = (X + \delta Y)(V + \epsilon W)
    D += \gamma_0 M
    E += \gamma_1 M

    where:
    - X, Y, V, W are the input matrices
    - X, Y are column major matrices [M, K], ldx/ldy = M
    - V, W are column major matrices [K, N], ldv/ldw = K  (优化后：改为列主序)
    - \delta, \epsilon are the scaling factors
    - \gamma_0, \gamma_1 are the scaling factors for D and E
    - M is the output matrix [M, N]
    - D, E are the output matrices (column major, ldd/lde = M)
    
    访存优化说明：
    - 所有矩阵都使用列主序，使得相邻线程访问连续内存地址
    - X, Y: M x K 列主序，沿 M 维度连续
    - V, W: K x N 列主序，沿 K 维度连续
*/

// 融合加载和计算的辅助函数 - 用于 A (2 x float4)
template<int alpha>
__device__ __forceinline__ void read_vec_large(
    float4* a_ptr, float4 x_vec[2], float4 y_vec[2]);

template<>
__device__ __forceinline__ void read_vec_large<0>(
    float4* a_ptr, float4 x_vec[2], float4 y_vec[2]) {
    a_ptr[0] = x_vec[0];
    a_ptr[1] = x_vec[1];
}

template<>
__device__ __forceinline__ void read_vec_large<1>(
    float4* a_ptr, float4 x_vec[2], float4 y_vec[2]) {
    a_ptr[0].x = x_vec[0].x + y_vec[0].x;
    a_ptr[0].y = x_vec[0].y + y_vec[0].y;
    a_ptr[0].z = x_vec[0].z + y_vec[0].z;
    a_ptr[0].w = x_vec[0].w + y_vec[0].w;
    a_ptr[1].x = x_vec[1].x + y_vec[1].x;
    a_ptr[1].y = x_vec[1].y + y_vec[1].y;
    a_ptr[1].z = x_vec[1].z + y_vec[1].z;
    a_ptr[1].w = x_vec[1].w + y_vec[1].w;
}

template<>
__device__ __forceinline__ void read_vec_large<-1>(
    float4* a_ptr, float4 x_vec[2], float4 y_vec[2]) {
    a_ptr[0].x = x_vec[0].x - y_vec[0].x;
    a_ptr[0].y = x_vec[0].y - y_vec[0].y;
    a_ptr[0].z = x_vec[0].z - y_vec[0].z;
    a_ptr[0].w = x_vec[0].w - y_vec[0].w;
    a_ptr[1].x = x_vec[1].x - y_vec[1].x;
    a_ptr[1].y = x_vec[1].y - y_vec[1].y;
    a_ptr[1].z = x_vec[1].z - y_vec[1].z;
    a_ptr[1].w = x_vec[1].w - y_vec[1].w;
}

// 散列存储 (用于 B) - 将 2 x float4 存储到 [k][n] 布局
// buffer[k * stride + col] 布局，一次存储 8 个连续的 k 值
template<int alpha>
__device__ __forceinline__ void store_vec_B_strided_large(
    float* buffer, float4 x_vec[2], float4 y_vec[2], int row, int col, int stride);

template<>
__device__ __forceinline__ void store_vec_B_strided_large<0>(
    float* buffer, float4 x_vec[2], float4 y_vec[2], int row, int col, int stride) {
    buffer[(row + 0) * stride + col] = x_vec[0].x;
    buffer[(row + 1) * stride + col] = x_vec[0].y;
    buffer[(row + 2) * stride + col] = x_vec[0].z;
    buffer[(row + 3) * stride + col] = x_vec[0].w;
    buffer[(row + 4) * stride + col] = x_vec[1].x;
    buffer[(row + 5) * stride + col] = x_vec[1].y;
    buffer[(row + 6) * stride + col] = x_vec[1].z;
    buffer[(row + 7) * stride + col] = x_vec[1].w;
}

template<>
__device__ __forceinline__ void store_vec_B_strided_large<1>(
    float* buffer, float4 x_vec[2], float4 y_vec[2], int row, int col, int stride) {
    buffer[(row + 0) * stride + col] = x_vec[0].x + y_vec[0].x;
    buffer[(row + 1) * stride + col] = x_vec[0].y + y_vec[0].y;
    buffer[(row + 2) * stride + col] = x_vec[0].z + y_vec[0].z;
    buffer[(row + 3) * stride + col] = x_vec[0].w + y_vec[0].w;
    buffer[(row + 4) * stride + col] = x_vec[1].x + y_vec[1].x;
    buffer[(row + 5) * stride + col] = x_vec[1].y + y_vec[1].y;
    buffer[(row + 6) * stride + col] = x_vec[1].z + y_vec[1].z;
    buffer[(row + 7) * stride + col] = x_vec[1].w + y_vec[1].w;
}

template<>
__device__ __forceinline__ void store_vec_B_strided_large<-1>(
    float* buffer, float4 x_vec[2], float4 y_vec[2], int row, int col, int stride) {
    buffer[(row + 0) * stride + col] = x_vec[0].x - y_vec[0].x;
    buffer[(row + 1) * stride + col] = x_vec[0].y - y_vec[0].y;
    buffer[(row + 2) * stride + col] = x_vec[0].z - y_vec[0].z;
    buffer[(row + 3) * stride + col] = x_vec[0].w - y_vec[0].w;
    buffer[(row + 4) * stride + col] = x_vec[1].x - y_vec[1].x;
    buffer[(row + 5) * stride + col] = x_vec[1].y - y_vec[1].y;
    buffer[(row + 6) * stride + col] = x_vec[1].z - y_vec[1].z;
    buffer[(row + 7) * stride + col] = x_vec[1].w - y_vec[1].w;
}

// write_vec_large_vec: gamma=1/-1 用 atomicAdd，gamma=2 直接写（无竞争时使用）
template<int gamma>
__device__ __forceinline__ void write_vec_large_vec(
    float* C, const float4* acc, int ldc);
template<>
__device__ __forceinline__ void write_vec_large_vec<0>(
    float* C, const float4* acc, int ldc) {
}
template<>
__device__ __forceinline__ void write_vec_large_vec<1>(
    float* C, const float4* acc, int ldc) {
    #pragma unroll
    for (int col = 0; col < 8; ++col) {
        const float4 lo = acc[col * 2 + 0];
        const float4 hi = acc[col * 2 + 1];
        atomicAdd(C + 0 + col * ldc, lo.x);
        atomicAdd(C + 1 + col * ldc, lo.y);
        atomicAdd(C + 2 + col * ldc, lo.z);
        atomicAdd(C + 3 + col * ldc, lo.w);
        atomicAdd(C + 4 + col * ldc, hi.x);
        atomicAdd(C + 5 + col * ldc, hi.y);
        atomicAdd(C + 6 + col * ldc, hi.z);
        atomicAdd(C + 7 + col * ldc, hi.w);
    }
}
template<>
__device__ __forceinline__ void write_vec_large_vec<-1>(
    float* C, const float4* acc, int ldc) {
    #pragma unroll
    for (int col = 0; col < 8; ++col) {
        const float4 lo = acc[col * 2 + 0];
        const float4 hi = acc[col * 2 + 1];
        atomicAdd(C + 0 + col * ldc, -lo.x);
        atomicAdd(C + 1 + col * ldc, -lo.y);
        atomicAdd(C + 2 + col * ldc, -lo.z);
        atomicAdd(C + 3 + col * ldc, -lo.w);
        atomicAdd(C + 4 + col * ldc, -hi.x);
        atomicAdd(C + 5 + col * ldc, -hi.y);
        atomicAdd(C + 6 + col * ldc, -hi.z);
        atomicAdd(C + 7 + col * ldc, -hi.w);
    }
}
// gamma=2: 直接写（覆盖，不累加），用于无竞争场景
template<>
__device__ __forceinline__ void write_vec_large_vec<2>(
    float* C, const float4* acc, int ldc) {
    #pragma unroll
    for (int col = 0; col < 8; ++col) {
        const float4 lo = acc[col * 2 + 0];
        const float4 hi = acc[col * 2 + 1];
        C[0 + col * ldc] = lo.x;
        C[1 + col * ldc] = lo.y;
        C[2 + col * ldc] = lo.z;
        C[3 + col * ldc] = lo.w;
        C[4 + col * ldc] = hi.x;
        C[5 + col * ldc] = hi.y;
        C[6 + col * ldc] = hi.z;
        C[7 + col * ldc] = hi.w;
    }
}
// gamma=-2: 直接写负值
template<>
__device__ __forceinline__ void write_vec_large_vec<-2>(
    float* C, const float4* acc, int ldc) {
    #pragma unroll
    for (int col = 0; col < 8; ++col) {
        const float4 lo = acc[col * 2 + 0];
        const float4 hi = acc[col * 2 + 1];
        C[0 + col * ldc] = -lo.x;
        C[1 + col * ldc] = -lo.y;
        C[2 + col * ldc] = -lo.z;
        C[3 + col * ldc] = -lo.w;
        C[4 + col * ldc] = -hi.x;
        C[5 + col * ldc] = -hi.y;
        C[6 + col * ldc] = -hi.z;
        C[7 + col * ldc] = -hi.w;
    }
}

#define ACCUM_COL(acc_lo, acc_hi, a_lo, a_hi, bval) \
    do { \
        (acc_lo).x = fmaf((a_lo).x, (bval), (acc_lo).x); \
        (acc_lo).y = fmaf((a_lo).y, (bval), (acc_lo).y); \
        (acc_lo).z = fmaf((a_lo).z, (bval), (acc_lo).z); \
        (acc_lo).w = fmaf((a_lo).w, (bval), (acc_lo).w); \
        (acc_hi).x = fmaf((a_hi).x, (bval), (acc_hi).x); \
        (acc_hi).y = fmaf((a_hi).y, (bval), (acc_hi).y); \
        (acc_hi).z = fmaf((a_hi).z, (bval), (acc_hi).z); \
        (acc_hi).w = fmaf((a_hi).w, (bval), (acc_hi).w); \
    } while (0)

/*
 * primitive_large: 优化版本
 * 
 * Tile 大小: ms=64, ns=64, ks=8
 * 线程数: 64
 * 
 * 访存优化：
 * - X, Y: 列主序 [M, K]，ldx/ldy = M，沿 M 维度连续加载
 * - V, W: 列主序 [K, N]，ldv/ldw = K，沿 K 维度连续加载
 * - 相邻线程访问连续内存地址，实现合并访存
 * 
 * 共享内存布局优化：
 * - buffer_A: [ks, ms] = [8, 64]，按 k 维度优先存储
 * - buffer_B: [ks, ns] = [8, 64]，按 k 维度优先存储
 * - 使用双缓冲隐藏全局内存延迟
 */
template <int delta, int epsilon, int gamma0, int gamma1>
__global__ __launch_bounds__(64)
void primitive_large(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const float* __restrict__ V,
    const float* __restrict__ W,
    float* __restrict__ D,
    float* __restrict__ E,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde
) {
    const int ms = 64, ns = 64, ks = 8;
    const int mw = 32, nw = 64, mr = 8, nr = 8;

    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;

    __shared__ float sAB[ks * ms + ks * ns];
    float* buffer_A = sAB;
    float* buffer_B = sAB + ks * ms;

    X += bx * ms;
    Y += bx * ms;
    V += by * ns * ldv;
    W += by * ns * ldw;

    const int floats_per_col_A = ms;
    const int float4x2_per_col_A = floats_per_col_A / 8;
    const int load_row_A = (tx % float4x2_per_col_A) * 8;
    const int load_col_A = tx / float4x2_per_col_A;
    const int load_col_B = tx;

    const int num_warp_B = ns / nw;
    int warp_id = tx / 32;
    int lane_id = tx & 31;
    int warp_id_B = warp_id % num_warp_B;
    int warp_id_A = warp_id / num_warp_B;
    int warp_offset_A = warp_id_A * mw;
    int warp_offset_B = warp_id_B * nw;
    int thread_id_B = lane_id % (nw / nr);
    int thread_id_A = lane_id / (nw / nr);
    int thread_offset_A = thread_id_A * mr;
    int thread_offset_B = thread_id_B * nr;

    float4 acc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    for (int k_outer = 0; k_outer < K; k_outer += ks) {
        const float* tile_X = X + k_outer * ldx;
        const float* tile_Y = Y + k_outer * ldy;
        const float* tile_V = V + k_outer;
        const float* tile_W = W + k_outer;

        float4 x_vec[2];
        float4 y_vec[2];
        x_vec[0] = *reinterpret_cast<const float4*>(tile_X + load_row_A + load_col_A * ldx);
        x_vec[1] = *reinterpret_cast<const float4*>(tile_X + load_row_A + load_col_A * ldx + 4);
        y_vec[0] = *reinterpret_cast<const float4*>(tile_Y + load_row_A + load_col_A * ldy);
        y_vec[1] = *reinterpret_cast<const float4*>(tile_Y + load_row_A + load_col_A * ldy + 4);
        float4* store_A = reinterpret_cast<float4*>(buffer_A + load_col_A * ms + load_row_A);
        read_vec_large<delta>(store_A, x_vec, y_vec);

        float4 v_vec[2];
        float4 w_vec[2];
        v_vec[0] = *reinterpret_cast<const float4*>(tile_V + load_col_B * ldv);
        v_vec[1] = *reinterpret_cast<const float4*>(tile_V + load_col_B * ldv + 4);
        w_vec[0] = *reinterpret_cast<const float4*>(tile_W + load_col_B * ldw);
        w_vec[1] = *reinterpret_cast<const float4*>(tile_W + load_col_B * ldw + 4);
        store_vec_B_strided_large<epsilon>(buffer_B, v_vec, w_vec, 0, load_col_B, ns);

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < ks; ++kk) {
            const float4 a_lo = *reinterpret_cast<const float4*>(buffer_A + kk * ms + warp_offset_A + thread_offset_A);
            const float4 a_hi = *reinterpret_cast<const float4*>(buffer_A + kk * ms + warp_offset_A + thread_offset_A + 4);
            const int base_n = kk * ns + warp_offset_B + thread_offset_B;
            const float4 b_lo = make_float4(buffer_B[base_n + 0], buffer_B[base_n + 1], buffer_B[base_n + 2], buffer_B[base_n + 3]);
            const float4 b_hi = make_float4(buffer_B[base_n + 4], buffer_B[base_n + 5], buffer_B[base_n + 6], buffer_B[base_n + 7]);

            ACCUM_COL(acc[0],  acc[1],  a_lo, a_hi, b_lo.x);
            ACCUM_COL(acc[2],  acc[3],  a_lo, a_hi, b_lo.y);
            ACCUM_COL(acc[4],  acc[5],  a_lo, a_hi, b_lo.z);
            ACCUM_COL(acc[6],  acc[7],  a_lo, a_hi, b_lo.w);
            ACCUM_COL(acc[8],  acc[9],  a_lo, a_hi, b_hi.x);
            ACCUM_COL(acc[10], acc[11], a_lo, a_hi, b_hi.y);
            ACCUM_COL(acc[12], acc[13], a_lo, a_hi, b_hi.z);
            ACCUM_COL(acc[14], acc[15], a_lo, a_hi, b_hi.w);
        }

        __syncthreads();
    }

    int out_m = bx * ms + warp_offset_A + thread_offset_A;
    int out_n = by * ns + warp_offset_B + thread_offset_B;

    D += out_m + out_n * ldd;
    write_vec_large_vec<gamma0>(D, acc, ldd);

    E += out_m + out_n * lde;
    write_vec_large_vec<gamma1>(E, acc, lde);
}

#endif 
