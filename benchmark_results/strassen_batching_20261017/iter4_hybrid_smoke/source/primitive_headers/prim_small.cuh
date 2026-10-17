#ifndef PRIM_SMALL_CUH
#define PRIM_SMALL_CUH

#include "base_components.cuh"
        
/*
    M = (X + \delta Y)(V + \epsilon W)
    D += \gamma_0 M
    E += \gamma_1 M

    where:
    - X, Y, V, W are the input matrices
    - X, Y are column major matrices [M, K], ldx/ldy = M
    - V, W are column major matrices [K, N], ldv/ldw = K  
    - \delta, \epsilon are the scaling factors
    - \gamma_0, \gamma_1 are the scaling factors for D and E
    - M is the output matrix [M, N]
    - D, E are the output matrices (column major, ldd/lde = M)
    
    访存优化说明：
    - 所有矩阵都使用列主序，使得相邻线程访问连续内存地址
    - X, Y: M x K 列主序，沿 M 维度连续
    - V, W: K x N 列主序，沿 K 维度连续
    - 计算 C = A * B 时，V 实际存储的是 B^T
*/

// 融合加载和计算的辅助函数 - 连续存储 (用于 A)
template<int alpha>
__device__ __forceinline__ void read_vec_small(
    float2* a_ptr, float2 x_vec, float2 y_vec);

template<>
__device__ __forceinline__ void read_vec_small<0>(
    float2* a_ptr, float2 x_vec, float2 y_vec) {
    a_ptr->x = x_vec.x;
    a_ptr->y = x_vec.y;
}

template<>
__device__ __forceinline__ void read_vec_small<1>(
    float2* a_ptr, float2 x_vec, float2 y_vec) {
    a_ptr->x = x_vec.x + y_vec.x;
    a_ptr->y = x_vec.y + y_vec.y;
}

template<>
__device__ __forceinline__ void read_vec_small<-1>(
    float2* a_ptr, float2 x_vec, float2 y_vec) {
    a_ptr->x = x_vec.x - y_vec.x;
    a_ptr->y = x_vec.y - y_vec.y;
}

// 散列存储 (用于 B) - 将 float2 存储到 [k][n] 布局
// buffer[k * stride + col] 布局
template<int alpha>
__device__ __forceinline__ void store_vec_B_strided(
    float* buffer, float2 x_vec, float2 y_vec, int row, int col, int stride);

template<>
__device__ __forceinline__ void store_vec_B_strided<0>(
    float* buffer, float2 x_vec, float2 y_vec, int row, int col, int stride) {
    buffer[row * stride + col] = x_vec.x;
    buffer[(row + 1) * stride + col] = x_vec.y;
}

template<>
__device__ __forceinline__ void store_vec_B_strided<1>(
    float* buffer, float2 x_vec, float2 y_vec, int row, int col, int stride) {
    buffer[row * stride + col] = x_vec.x + y_vec.x;
    buffer[(row + 1) * stride + col] = x_vec.y + y_vec.y;
}

template<>
__device__ __forceinline__ void store_vec_B_strided<-1>(
    float* buffer, float2 x_vec, float2 y_vec, int row, int col, int stride) {
    buffer[row * stride + col] = x_vec.x - y_vec.x;
    buffer[(row + 1) * stride + col] = x_vec.y - y_vec.y;
}

// write_vec_small: 使用 atomicAdd 累加结果
template<int gamma>
__device__ __forceinline__ void write_vec_small(
    float* C, float* res, int ldc);
template<>
__device__ __forceinline__ void write_vec_small<0>(
    float* C, float* res, int ldc) {
}
template<>
__device__ __forceinline__ void write_vec_small<1>(
    float* C, float* res, int ldc) {
    atomicAdd(C, res[0]);
    atomicAdd(C + ldc, res[1]);
    atomicAdd(C + 1, res[2]);
    atomicAdd(C + 1 + ldc, res[3]);
}
template<>
__device__ __forceinline__ void write_vec_small<-1>(
    float* C, float* res, int ldc) {
    atomicAdd(C, -res[0]);
    atomicAdd(C + ldc, -res[1]);
    atomicAdd(C + 1, -res[2]);
    atomicAdd(C + 1 + ldc, -res[3]);
}
// gamma=2: 直接写
template<>
__device__ __forceinline__ void write_vec_small<2>(
    float* C, float* res, int ldc) {
    *((float2*)(C + 0)) = make_float2(res[0], res[2]);
    *((float2*)(C + ldc)) = make_float2(res[1], res[3]);
}
// gamma=-2: 直接写负值
template<>
__device__ __forceinline__ void write_vec_small<-2>(
    float* C, float* res, int ldc) {
    *((float2*)(C + 0)) = make_float2(-res[0], -res[2]);
    *((float2*)(C + ldc)) = make_float2(-res[1], -res[3]);
}

/*
 * primitive_small: 优化版本
 * 
 * 访存优化：
 * - X, Y: 列主序 [M, K]，ldx/ldy = M，沿 M 维度连续加载
 * - V, W: 列主序 [K, N]，ldv/ldw = K，沿 K 维度连续加载（无需转置！）
 * - 相邻线程访问连续内存地址，实现合并访存
 * 
 * 共享内存布局优化：
 * - buffer_A: [ks, ms] = [8, 16]，按 k 维度优先存储
 * - buffer_B: [ks, ns] = [8, 16]，按 k 维度优先存储
 * - 使用双缓冲隐藏全局内存延迟
 */
template <int delta, int epsilon, int gamma0, int gamma1>
__global__ __launch_bounds__(64)
void primitive_small(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const float* __restrict__ V,
    const float* __restrict__ W,
    float* __restrict__ D,
    float* __restrict__ E,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde
) {
    // Tile dimensions
    const int ms = 16, ns = 16, ks = 8;
    const int mw = 8, nw = 16, mr = 2, nr = 2;
    
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;

    // 共享内存：双缓冲，buffer_A [2][ks][ms], buffer_B [2][ks][ns]
    __shared__ float sAB[2 * ks * ms + 2 * ks * ns];  // 512 floats
    
    const int buffer_A_offset = 0;
    const int buffer_B_offset = 2 * ks * ms;

    // ========== 全局内存指针设置 ==========
    // X, Y 列主序 [M, K]: 块级偏移沿 M 维度
    X += bx * ms;
    Y += bx * ms;
    
    // V, W 列主序 [K, N]: 块级偏移沿 N 维度
    V += by * ns * ldv;
    W += by * ns * ldw;

    // ========== X, Y 加载索引（列主序 [M, K]）==========
    // 64 线程加载 ms * ks = 16 * 8 = 128 floats
    // 每列 ms=16 元素，共 ks=8 列
    // 每个线程加载 2 个连续 float (float2)
    const int float2_per_col_A = ms / 2;  // = 8
    int load_row_A = (tx % float2_per_col_A) * 2;  // 0, 2, 4, ..., 14
    int load_col_A = tx / float2_per_col_A;        // 0, 1, ..., 7
    X += load_row_A + load_col_A * ldx;
    Y += load_row_A + load_col_A * ldy;

    // ========== V, W 加载索引（列主序 [K, N]）==========
    // 64 线程加载 ks * ns = 8 * 16 = 128 floats
    // 每列 ks=8 元素，共 ns=16 列
    // 优化：相邻线程访问连续地址（沿 K 维度）
    const int float2_per_col_B = ks / 2;  // = 4
    int load_row_B = (tx % float2_per_col_B) * 2;  // 0, 2, 4, 6 (K 维度)
    int load_col_B = tx / float2_per_col_B;        // 0, 1, ..., 15 (N 维度)
    V += load_row_B + load_col_B * ldv;
    W += load_row_B + load_col_B * ldw;

    // ========== 第一次预取 ==========
    int ping = 0;
    float* buffer_A = sAB + buffer_A_offset + ping * ks * ms;
    float* buffer_B = sAB + buffer_B_offset + ping * ks * ns;

    float2 prefetch_X = *((float2*)X);
    float2 prefetch_Y = *((float2*)Y);
    float2 prefetch_V = *((float2*)V);
    float2 prefetch_W = *((float2*)W);

    // 存储到共享内存
    // buffer_A 布局: [ks][ms]，A[k][m] = buffer_A[k * ms + m]
    // 线程 tx 加载 A[load_col_A, load_row_A:load_row_A+2]
    // 存储到 buffer_A[load_col_A * ms + load_row_A : load_row_A+2]
    float2* store_A = ((float2*)(buffer_A + load_col_A * ms)) + (load_row_A / 2);
    read_vec_small<delta>(store_A, prefetch_X, prefetch_Y);
    
    // buffer_B 布局: [ks][ns]，B[k][n] = buffer_B[k * ns + n]
    // 线程 tx 加载 V[load_row_B:load_row_B+2, load_col_B]
    // 需要存储到 buffer_B[load_row_B * ns + load_col_B] 和 buffer_B[(load_row_B+1) * ns + load_col_B]
    store_vec_B_strided<epsilon>(buffer_B, prefetch_V, prefetch_W, load_row_B, load_col_B, ns);

    __syncthreads();

    // ========== Warp 和线程级索引 ==========
    const int num_warp_B = ns / nw;  // = 1
    int warp_id = tx / 32;
    int lane_id = tx & 31;
    
    int warp_id_B = warp_id % num_warp_B;
    int warp_id_A = warp_id / num_warp_B;
    
    int warp_offset_A = warp_id_A * mw;
    int warp_offset_B = warp_id_B * nw;
    
    int thread_id_B = lane_id % (nw / nr);  // 0-7
    int thread_id_A = lane_id / (nw / nr);  // 0-3
    
    int thread_offset_A = thread_id_A * mr;
    int thread_offset_B = thread_id_B * nr;

    // ========== 寄存器 ==========
    float2 vec_A[2], vec_B[2];
    float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // 预取第一个 k 的数据到寄存器
    // 共享内存布局: buffer[k * dim + offset]
    vec_A[0] = *(float2*)(buffer_A + 0 * ms + warp_offset_A + thread_offset_A);
    vec_B[0] = *(float2*)(buffer_B + 0 * ns + warp_offset_B + thread_offset_B);

    // ========== K 循环 ==========
    for (int k_outer = 0; k_outer < K; k_outer += ks) {
        // 预取下一个 tile
        X += ks * ldx;
        Y += ks * ldy;
        V += ks;  // 列主序 V [K, N]，K 维度偏移 ks
        W += ks;
        
        if (k_outer + ks < K) {
            prefetch_X = *((float2*)X);
            prefetch_Y = *((float2*)Y);
            prefetch_V = *((float2*)V);
            prefetch_W = *((float2*)W);
        }

        // 内层 k 循环
        #pragma unroll
        for (int kk = 0; kk < ks; ++kk) {
            int curr = kk & 1;
            int next = (kk + 1) & 1;
            int next_kk = (kk + 1) % ks;

            // 预取下一个 k 的数据
            vec_A[next] = *(float2*)(buffer_A + next_kk * ms + warp_offset_A + thread_offset_A);
            vec_B[next] = *(float2*)(buffer_B + next_kk * ns + warp_offset_B + thread_offset_B);

            // 计算 2x2 外积
            res[0] += vec_A[curr].x * vec_B[curr].x;
            res[1] += vec_A[curr].x * vec_B[curr].y;
            res[2] += vec_A[curr].y * vec_B[curr].x;
            res[3] += vec_A[curr].y * vec_B[curr].y;
        }

        // 切换缓冲区
        ping = 1 - ping;
        buffer_A = sAB + buffer_A_offset + ping * ks * ms;
        buffer_B = sAB + buffer_B_offset + ping * ks * ns;
        
        // 存储预取的数据
        store_A = ((float2*)(buffer_A + load_col_A * ms)) + (load_row_A / 2);
        read_vec_small<delta>(store_A, prefetch_X, prefetch_Y);
        store_vec_B_strided<epsilon>(buffer_B, prefetch_V, prefetch_W, load_row_B, load_col_B, ns);

        __syncthreads();

        // 预取下一轮的第一个 k
        vec_A[0] = *(float2*)(buffer_A + 0 * ms + warp_offset_A + thread_offset_A);
        vec_B[0] = *(float2*)(buffer_B + 0 * ns + warp_offset_B + thread_offset_B);
    }

    // ========== 写回结果 ==========
    int out_m = bx * ms + warp_offset_A + thread_offset_A;
    int out_n = by * ns + warp_offset_B + thread_offset_B;
    
    D += out_m + out_n * ldd;
    write_vec_small<gamma0>(D, res, ldd);

    E += out_m + out_n * lde;
    write_vec_small<gamma1>(E, res, lde);
}

#endif 