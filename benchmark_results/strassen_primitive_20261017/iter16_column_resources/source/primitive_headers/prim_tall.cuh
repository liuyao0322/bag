#ifndef PRIM_TALL_CUH
#define PRIM_TALL_CUH

#include "base_components.cuh"
        
/*
    M = (X + \delta Y)(V + \epsilon W)
    D += \gamma_0 M
    E += \gamma_1 M

    where:
    - X, Y are column major matrices [M, K], ldx/ldy = M
    - V, W are column major matrices [K, N], ldv/ldw = K (优化后)
    - Tall version: optimized for M >> N (128x32 tile)
    
    要求: M % 128 == 0, N % 32 == 0, K % 8 == 0
*/

template<int alpha>
__device__ __forceinline__ void read_vec_tall_A(
    float4* a_ptr, float4 x_vec[2], float4 y_vec[2]);

template<>
__device__ __forceinline__ void read_vec_tall_A<0>(
    float4* a_ptr, float4 x_vec[2], float4 y_vec[2]) {
    a_ptr[0] = x_vec[0];
    a_ptr[1] = x_vec[1];
}

template<>
__device__ __forceinline__ void read_vec_tall_A<1>(
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
__device__ __forceinline__ void read_vec_tall_A<-1>(
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

// 存储 B 矩阵到共享内存 (列主序 [K, N] -> 共享内存 [k, n])
template<int alpha>
__device__ __forceinline__ void store_vec_B_tall(
    float* buffer, float2 v_vec, float2 w_vec, int row, int col, int stride);

template<>
__device__ __forceinline__ void store_vec_B_tall<0>(
    float* buffer, float2 v_vec, float2 w_vec, int row, int col, int stride) {
    buffer[row * stride + col] = v_vec.x;
    buffer[(row + 1) * stride + col] = v_vec.y;
}

template<>
__device__ __forceinline__ void store_vec_B_tall<1>(
    float* buffer, float2 v_vec, float2 w_vec, int row, int col, int stride) {
    buffer[row * stride + col] = v_vec.x + w_vec.x;
    buffer[(row + 1) * stride + col] = v_vec.y + w_vec.y;
}

template<>
__device__ __forceinline__ void store_vec_B_tall<-1>(
    float* buffer, float2 v_vec, float2 w_vec, int row, int col, int stride) {
    buffer[row * stride + col] = v_vec.x - w_vec.x;
    buffer[(row + 1) * stride + col] = v_vec.y - w_vec.y;
}

// write_vec_tall: 使用 atomicAdd 累加结果
template<int gamma>
__device__ __forceinline__ void write_vec_tall(float* C, float* res, int ldc);

template<>
__device__ __forceinline__ void write_vec_tall<0>(float* C, float* res, int ldc) {}

template<>
__device__ __forceinline__ void write_vec_tall<1>(float* C, float* res, int ldc) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            atomicAdd(C + i + (long long)j * ldc, res[i * 4 + j]);
        }
    }
}

template<>
__device__ __forceinline__ void write_vec_tall<-1>(float* C, float* res, int ldc) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            atomicAdd(C + i + (long long)j * ldc, -res[i * 4 + j]);
        }
    }
}
template<>
__device__ __forceinline__ void write_vec_tall<2>(float* C, float* res, int ldc) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        *((float4*)(C + (long long)j * ldc + 0)) = make_float4(res[0 * 4 + j], res[1 * 4 + j], res[2 * 4 + j], res[3 * 4 + j]);
        *((float4*)(C + (long long)j * ldc + 4)) = make_float4(res[4 * 4 + j], res[5 * 4 + j], res[6 * 4 + j], res[7 * 4 + j]);
    }
}
template<>
__device__ __forceinline__ void write_vec_tall<-2>(float* C, float* res, int ldc) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        *((float4*)(C + (long long)j * ldc + 0)) = make_float4(-res[0 * 4 + j], -res[1 * 4 + j], -res[2 * 4 + j], -res[3 * 4 + j]);
        *((float4*)(C + (long long)j * ldc + 4)) = make_float4(-res[4 * 4 + j], -res[5 * 4 + j], -res[6 * 4 + j], -res[7 * 4 + j]);
    }
}

/*
 * primitive_tall: 优化版本 (无边界检查)
 * 
 * Tile 大小: ms=128, ns=32, ks=8
 * 线程数: 128
 * 要求: M % 128 == 0, N % 32 == 0, K % 8 == 0
 */
template <int delta, int epsilon, int gamma0, int gamma1>
__global__ __launch_bounds__(128)
void primitive_tall(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const float* __restrict__ V,
    const float* __restrict__ W,
    float* __restrict__ D,
    float* __restrict__ E,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde
) {
    const int ms = 128, ns = 32, ks = 8;
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;

    // 共享内存布局: buffer_A[2][ks][ms], buffer_B[2][ks][ns]
    __shared__ float sAB[2560];  // 2 * (128*8 + 32*8) = 2 * 1280 = 2560
    const int buffer_A_offset = 0;
    const int buffer_B_offset = 2 * ms * ks;  // 2048

    // 块级偏移
    X += bx * ms;
    Y += bx * ms;
    V += (long long)by * ns * ldv;  // 列主序 [K, N]：偏移 = col * ldv
    W += (long long)by * ns * ldw;

    // ========== X, Y 加载索引（列主序 [M, K]）==========
    // 128 线程加载 128 * 8 = 1024 floats
    // 每线程加载 8 floats (float4 * 2)
    const int load_A_floats = 8;
    const int load_A_threads_per_col = ms / load_A_floats;  // 128 / 8 = 16
    int load_col_A = tx / load_A_threads_per_col;  // K 维度: 0-7
    int load_row_A = (tx % load_A_threads_per_col) * load_A_floats;  // M 维度
    X += load_row_A + (long long)load_col_A * ldx;
    Y += load_row_A + (long long)load_col_A * ldy;

    // ========== V, W 加载索引（列主序 [K, N]）==========
    // 128 线程加载 32 * 8 = 256 floats
    // 每线程加载 2 floats (float2)
    // 每列 K=8 元素，需要 4 个 float2
    // 32 列共需要 128 个 float2
    const int float2_per_col_B = ks / 2;  // 8 / 2 = 4
    int load_col_B = tx / float2_per_col_B;  // N 维度: 0-31
    int load_row_B = (tx % float2_per_col_B) * 2;  // K 维度: 0, 2, 4, 6
    V += load_row_B + (long long)load_col_B * ldv;
    W += load_row_B + (long long)load_col_B * ldw;

    // 第一次预取
    float4 prefetch_X[2], prefetch_Y[2];
    float2 prefetch_V, prefetch_W;
    
    prefetch_X[0] = *((float4*)X);
    prefetch_X[1] = *((float4*)X + 1);
    prefetch_Y[0] = *((float4*)Y);
    prefetch_Y[1] = *((float4*)Y + 1);
    prefetch_V = *((float2*)V);
    prefetch_W = *((float2*)W);

    // 存储到共享内存
    float* buffer_A = sAB + buffer_A_offset;
    float* buffer_B = sAB + buffer_B_offset;
    
    // buffer_A: [k][m] layout
    float4* buffer_A_ptr = (float4*)(buffer_A + load_col_A * ms + load_row_A);
    read_vec_tall_A<delta>(buffer_A_ptr, prefetch_X, prefetch_Y);
    
    // buffer_B: [k][n] layout，列主序输入直接存储
    store_vec_B_tall<epsilon>(buffer_B, prefetch_V, prefetch_W, load_row_B, load_col_B, ns);

    __syncthreads();

    // Warp 和线程级索引
    const int mw = 64, nw = 16, mr = 8, nr = 4;
    int id_warp = tx / 32;  // 0-3
    int idA_warp = id_warp / 2;  // 0 or 1
    int idB_warp = id_warp % 2;  // 0 or 1

    int offset_vec_A_warp = idA_warp * mw;  // 0 or 64
    int offset_vec_B_warp = idB_warp * nw;  // 0 or 16
    
    int lane = tx & 31;
    int idA_thread = lane / (nw / nr);  // lane / 4 = 0-7
    int idB_thread = lane % (nw / nr);  // lane % 4 = 0-3

    int offset_vec_A_thread = idA_thread * mr;  // 0, 8, 16, ..., 56
    int offset_vec_B_thread = idB_thread * nr;  // 0, 4, 8, 12

    // 累加器
    float res[32];
    memset(res, 0, sizeof(res));
    
    float4 vec_A[4], vec_B[2];
    
    // 预取第一个 k
    vec_A[0] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread);
    vec_A[1] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + 4);
    vec_B[0] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread);

    // K 主循环
    for (int k = 0; k < K; k += ks) {
        // 预取下一个 tile
        if (k + ks < K) {
            X += ks * ldx;
            Y += ks * ldy;
            V += ks;  // 列主序：K 维度步进
            W += ks;
            
            prefetch_X[0] = *((float4*)X);
            prefetch_X[1] = *((float4*)X + 1);
            prefetch_Y[0] = *((float4*)Y);
            prefetch_Y[1] = *((float4*)Y + 1);
            prefetch_V = *((float2*)V);
            prefetch_W = *((float2*)W);
        }

        // 计算当前 tile
        #pragma unroll
        for (int kk = 0; kk < ks; ++kk) {
            int reg_cur = kk & 1;
            int reg_next = (kk + 1) & 1;
            int kk_next = (kk + 1) % ks;
            
            // 预取下一个 k 的数据
            vec_A[reg_next * 2] = *(float4*)(buffer_A + kk_next * ms + offset_vec_A_warp + offset_vec_A_thread);
            vec_A[reg_next * 2 + 1] = *(float4*)(buffer_A + kk_next * ms + offset_vec_A_warp + offset_vec_A_thread + 4);
            vec_B[reg_next] = *(float4*)(buffer_B + kk_next * ns + offset_vec_B_warp + offset_vec_B_thread);
            
            // 计算: res[i][j] += A[i] * B[j]
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float a0 = ((float*)&vec_A[reg_cur * 2])[i];
                float a1 = ((float*)&vec_A[reg_cur * 2 + 1])[i];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    float b = ((float*)&vec_B[reg_cur])[j];
                    res[i * 4 + j] = fmaf(a0, b, res[i * 4 + j]);
                    res[(i + 4) * 4 + j] = fmaf(a1, b, res[(i + 4) * 4 + j]);
                }
            }
        }
        
        // 切换缓冲区
        int offset_store = (((k / ks) + 1) & 1);
        buffer_A = sAB + buffer_A_offset + offset_store * ms * ks;
        buffer_B = sAB + buffer_B_offset + offset_store * ns * ks;
        
        // 存储预取的数据
        float4* buffer_A_ptr_next = (float4*)(buffer_A + load_col_A * ms + load_row_A);
        read_vec_tall_A<delta>(buffer_A_ptr_next, prefetch_X, prefetch_Y);
        store_vec_B_tall<epsilon>(buffer_B, prefetch_V, prefetch_W, load_row_B, load_col_B, ns);

        __syncthreads();
        
        // 预取下一个 tile 的第一个 k
        vec_A[0] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread);
        vec_A[1] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + 4);
        vec_B[0] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread);
    }

    // 写回结果
    D += bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    D += (long long)(by * ns + offset_vec_B_warp + offset_vec_B_thread) * ldd;
    write_vec_tall<gamma0>(D, res, ldd);

    E += bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    E += (long long)(by * ns + offset_vec_B_warp + offset_vec_B_thread) * lde;
    write_vec_tall<gamma1>(E, res, lde);
}

#endif 
