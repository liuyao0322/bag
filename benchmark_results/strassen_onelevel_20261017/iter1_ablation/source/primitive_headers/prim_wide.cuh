#ifndef PRIM_WIDE_CUH
#define PRIM_WIDE_CUH

#include "base_components.cuh"
        
/*
    M = (X + \delta Y)(V + \epsilon W)
    D += \gamma_0 M
    E += \gamma_1 M

    where:
    - X, Y are column major matrices [M, K], ldx/ldy = M
    - V, W are column major matrices [K, N], ldv/ldw = K (优化后)
    - Wide version: optimized for N >> M (32x128 tile)
    
    要求: M % 32 == 0, N % 128 == 0, K % 8 == 0
*/

template<int alpha>
__device__ __forceinline__ void read_vec_wide_A(
    float2* a_ptr, float2 x_vec, float2 y_vec);

template<>
__device__ __forceinline__ void read_vec_wide_A<0>(
    float2* a_ptr, float2 x_vec, float2 y_vec) {
    *a_ptr = x_vec;
}

template<>
__device__ __forceinline__ void read_vec_wide_A<1>(
    float2* a_ptr, float2 x_vec, float2 y_vec) {
    a_ptr->x = x_vec.x + y_vec.x;
    a_ptr->y = x_vec.y + y_vec.y;
}

template<>
__device__ __forceinline__ void read_vec_wide_A<-1>(
    float2* a_ptr, float2 x_vec, float2 y_vec) {
    a_ptr->x = x_vec.x - y_vec.x;
    a_ptr->y = x_vec.y - y_vec.y;
}

// 存储 B 矩阵到共享内存 (列主序 [K, N] -> 共享内存 [k, n])
template<int alpha>
__device__ __forceinline__ void store_vec_B_wide(
    float* buffer, float4 v_vec0, float4 v_vec1, float4 w_vec0, float4 w_vec1, int col, int stride);

template<>
__device__ __forceinline__ void store_vec_B_wide<0>(
    float* buffer, float4 v_vec0, float4 v_vec1, float4 w_vec0, float4 w_vec1, int col, int stride) {
    buffer[0 * stride + col] = v_vec0.x;
    buffer[1 * stride + col] = v_vec0.y;
    buffer[2 * stride + col] = v_vec0.z;
    buffer[3 * stride + col] = v_vec0.w;
    buffer[4 * stride + col] = v_vec1.x;
    buffer[5 * stride + col] = v_vec1.y;
    buffer[6 * stride + col] = v_vec1.z;
    buffer[7 * stride + col] = v_vec1.w;
}

template<>
__device__ __forceinline__ void store_vec_B_wide<1>(
    float* buffer, float4 v_vec0, float4 v_vec1, float4 w_vec0, float4 w_vec1, int col, int stride) {
    buffer[0 * stride + col] = v_vec0.x + w_vec0.x;
    buffer[1 * stride + col] = v_vec0.y + w_vec0.y;
    buffer[2 * stride + col] = v_vec0.z + w_vec0.z;
    buffer[3 * stride + col] = v_vec0.w + w_vec0.w;
    buffer[4 * stride + col] = v_vec1.x + w_vec1.x;
    buffer[5 * stride + col] = v_vec1.y + w_vec1.y;
    buffer[6 * stride + col] = v_vec1.z + w_vec1.z;
    buffer[7 * stride + col] = v_vec1.w + w_vec1.w;
}

template<>
__device__ __forceinline__ void store_vec_B_wide<-1>(
    float* buffer, float4 v_vec0, float4 v_vec1, float4 w_vec0, float4 w_vec1, int col, int stride) {
    buffer[0 * stride + col] = v_vec0.x - w_vec0.x;
    buffer[1 * stride + col] = v_vec0.y - w_vec0.y;
    buffer[2 * stride + col] = v_vec0.z - w_vec0.z;
    buffer[3 * stride + col] = v_vec0.w - w_vec0.w;
    buffer[4 * stride + col] = v_vec1.x - w_vec1.x;
    buffer[5 * stride + col] = v_vec1.y - w_vec1.y;
    buffer[6 * stride + col] = v_vec1.z - w_vec1.z;
    buffer[7 * stride + col] = v_vec1.w - w_vec1.w;
}

// write_vec_wide: 使用 atomicAdd 累加结果
template<int gamma>
__device__ __forceinline__ void write_vec_wide(float* C, float* res, int ldc);

template<>
__device__ __forceinline__ void write_vec_wide<0>(float* C, float* res, int ldc) {}

template<>
__device__ __forceinline__ void write_vec_wide<1>(float* C, float* res, int ldc) {
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            atomicAdd(C + i + (long long)j * ldc, res[i * 8 + j]);
        }
    }
}

template<>
__device__ __forceinline__ void write_vec_wide<-1>(float* C, float* res, int ldc) {
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            atomicAdd(C + i + (long long)j * ldc, -res[i * 8 + j]);
        }
    }
}
template<>
__device__ __forceinline__ void write_vec_wide<2>(float* C, float* res, int ldc) {
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        *((float4*)(C + (long long)j * ldc)) = make_float4(res[0 * 8 + j], res[1 * 8 + j], res[2 * 8 + j], res[3 * 8 + j]);
    }
}
template<>
__device__ __forceinline__ void write_vec_wide<-2>(float* C, float* res, int ldc) {
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        *((float4*)(C + (long long)j * ldc)) = make_float4(-res[0 * 8 + j], -res[1 * 8 + j], -res[2 * 8 + j], -res[3 * 8 + j]);
    }
}

/*
 * primitive_wide: 优化版本 (无边界检查)
 * 
 * Tile 大小: ms=32, ns=128, ks=8
 * 线程数: 128
 * 要求: M % 32 == 0, N % 128 == 0, K % 8 == 0
 */
template <int delta, int epsilon, int gamma0, int gamma1>
__global__ __launch_bounds__(128)
void primitive_wide(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const float* __restrict__ V,
    const float* __restrict__ W,
    float* __restrict__ D,
    float* __restrict__ E,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde
) {
    const int ms = 32, ns = 128, ks = 8;
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;

    // 共享内存布局: buffer_A[2][ks][ms], buffer_B[2][ks][ns]
    __shared__ float sAB[2560];  // 2 * (32*8 + 128*8) = 2 * 1280 = 2560
    const int buffer_A_offset = 0;
    const int buffer_B_offset = 2 * ms * ks;  // 512

    // 块级偏移
    X += bx * ms;
    Y += bx * ms;
    V += (long long)by * ns * ldv;  // 列主序 [K, N]
    W += (long long)by * ns * ldw;

    // ========== X, Y 加载索引（列主序 [M, K]）==========
    // 128 线程加载 32 * 8 = 256 floats
    // 每线程加载 2 floats (float2)
    // 每列 M=32 元素，需要 16 个 float2
    // 8 列共需要 128 个 float2
    const int float2_per_col_A = ms / 2;  // 32 / 2 = 16
    int load_col_A = tx / float2_per_col_A;  // K 维度: 0-7
    int load_row_A = (tx % float2_per_col_A) * 2;  // M 维度: 0, 2, 4, ..., 30
    X += load_row_A + (long long)load_col_A * ldx;
    Y += load_row_A + (long long)load_col_A * ldy;

    // ========== V, W 加载索引（列主序 [K, N]）==========
    // 128 线程加载 128 * 8 = 1024 floats
    // 每线程加载 8 floats (2 个 float4)
    // 每列 K=8 元素 = 2 个 float4
    // 每线程处理 1 列
    int load_col_B = tx;  // N 维度: 0-127
    V += (long long)load_col_B * ldv;
    W += (long long)load_col_B * ldw;

    // 第一次预取
    float2 prefetch_X, prefetch_Y;
    float4 prefetch_V[2], prefetch_W[2];
    
    prefetch_X = *((float2*)X);
    prefetch_Y = *((float2*)Y);
    prefetch_V[0] = *((float4*)V);
    prefetch_V[1] = *((float4*)(V + 4));
    prefetch_W[0] = *((float4*)W);
    prefetch_W[1] = *((float4*)(W + 4));

    // 存储到共享内存
    float* buffer_A = sAB + buffer_A_offset;
    float* buffer_B = sAB + buffer_B_offset;
    
    // buffer_A: [k][m] layout
    float2* buffer_A_ptr = (float2*)(buffer_A + load_col_A * ms + load_row_A);
    read_vec_wide_A<delta>(buffer_A_ptr, prefetch_X, prefetch_Y);
    
    // buffer_B: [k][n] layout
    store_vec_B_wide<epsilon>(buffer_B, prefetch_V[0], prefetch_V[1], prefetch_W[0], prefetch_W[1], load_col_B, ns);

    __syncthreads();

    // Warp 和线程级索引
    const int mw = 16, nw = 64, mr = 4, nr = 8;
    int id_warp = tx / 32;  // 0-3
    int idA_warp = id_warp / 2;  // 0 or 1
    int idB_warp = id_warp % 2;  // 0 or 1

    int offset_vec_A_warp = idA_warp * mw;  // 0 or 16
    int offset_vec_B_warp = idB_warp * nw;  // 0 or 64
    
    int lane = tx & 31;
    int idA_thread = lane / (nw / nr);  // lane / 8 = 0-3
    int idB_thread = lane % (nw / nr);  // lane % 8 = 0-7

    int offset_vec_A_thread = idA_thread * mr;  // 0, 4, 8, 12
    int offset_vec_B_thread = idB_thread * nr;  // 0, 8, 16, ..., 56

    // 累加器
    float res[32];
    memset(res, 0, sizeof(res));
    
    float4 vec_A[2], vec_B[4];
    
    // 预取第一个 k
    vec_A[0] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread);
    vec_B[0] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread);
    vec_B[1] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + 4);

    // K 主循环
    for (int k = 0; k < K; k += ks) {
        // 预取下一个 tile
        if (k + ks < K) {
            X += ks * ldx;
            Y += ks * ldy;
            V += ks;  // 列主序：K 维度步进
            W += ks;
            
            prefetch_X = *((float2*)X);
            prefetch_Y = *((float2*)Y);
            prefetch_V[0] = *((float4*)V);
            prefetch_V[1] = *((float4*)(V + 4));
            prefetch_W[0] = *((float4*)W);
            prefetch_W[1] = *((float4*)(W + 4));
        }

        // 计算当前 tile
        #pragma unroll
        for (int kk = 0; kk < ks; ++kk) {
            int reg_cur = kk & 1;
            int reg_next = (kk + 1) & 1;
            int kk_next = (kk + 1) % ks;
            
            // 预取下一个 k 的数据
            vec_A[reg_next] = *(float4*)(buffer_A + kk_next * ms + offset_vec_A_warp + offset_vec_A_thread);
            vec_B[reg_next * 2] = *(float4*)(buffer_B + kk_next * ns + offset_vec_B_warp + offset_vec_B_thread);
            vec_B[reg_next * 2 + 1] = *(float4*)(buffer_B + kk_next * ns + offset_vec_B_warp + offset_vec_B_thread + 4);
            
            // 计算: res[i][j] += A[i] * B[j]
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float a = ((float*)&vec_A[reg_cur])[i];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    res[i * 8 + j] = fmaf(a, ((float*)&vec_B[reg_cur * 2])[j], res[i * 8 + j]);
                    res[i * 8 + j + 4] = fmaf(a, ((float*)&vec_B[reg_cur * 2 + 1])[j], res[i * 8 + j + 4]);
                }
            }
        }
        
        // 切换缓冲区
        int offset_store = (((k / ks) + 1) & 1);
        buffer_A = sAB + buffer_A_offset + offset_store * ms * ks;
        buffer_B = sAB + buffer_B_offset + offset_store * ns * ks;
        
        // 存储预取的数据
        float2* buffer_A_ptr_next = (float2*)(buffer_A + load_col_A * ms + load_row_A);
        read_vec_wide_A<delta>(buffer_A_ptr_next, prefetch_X, prefetch_Y);
        store_vec_B_wide<epsilon>(buffer_B, prefetch_V[0], prefetch_V[1], prefetch_W[0], prefetch_W[1], load_col_B, ns);

        __syncthreads();
        
        // 预取下一个 tile 的第一个 k
        vec_A[0] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread);
        vec_B[0] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread);
        vec_B[1] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + 4);
    }

    // 写回结果
    D += bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    D += (long long)(by * ns + offset_vec_B_warp + offset_vec_B_thread) * ldd;
    write_vec_wide<gamma0>(D, res, ldd);

    E += bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    E += (long long)(by * ns + offset_vec_B_warp + offset_vec_B_thread) * lde;
    write_vec_wide<gamma1>(E, res, lde);
}

#endif 
