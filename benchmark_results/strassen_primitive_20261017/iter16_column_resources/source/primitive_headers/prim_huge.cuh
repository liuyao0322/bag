#ifndef PRIM_HUGE_CUH
#define PRIM_HUGE_CUH

#include "base_components.cuh"

// Vectorized accumulation macro
#define tab(t, a, b) \
    do { \
        (t).x = fmaf((a).x, (b), (t).x); \
        (t).y = fmaf((a).y, (b), (t).y); \
        (t).z = fmaf((a).z, (b), (t).z); \
        (t).w = fmaf((a).w, (b), (t).w); \
    } while (0)
        
/*
    M = (X + \delta Y)(V + \epsilon W)
    D += \gamma_0 M
    E += \gamma_1 M

    where:
    - X, Y, V, W are the input matrices
    - X, Y are column major matrices [M, K], ldx/ldy = M
    - V, W are column major matrices [K, N], ldv/ldw = K
    - Huge version: optimized for very large matrices (128x128 tile)
    
*/

// 融合加载和计算的辅助函数
template<int alpha>
__device__ __forceinline__ float4 read_vec_huge(float4 x_vec, float4 y_vec);

template<>
__device__ __forceinline__ float4 read_vec_huge<0>(float4 x_vec, float4 y_vec) {
    return x_vec;
}

template<>
__device__ __forceinline__ float4 read_vec_huge<1>(float4 x_vec, float4 y_vec) {
    float4 result;
    result.x = x_vec.x + y_vec.x;
    result.y = x_vec.y + y_vec.y;
    result.z = x_vec.z + y_vec.z;
    result.w = x_vec.w + y_vec.w;
    return result;
}

template<>
__device__ __forceinline__ float4 read_vec_huge<-1>(float4 x_vec, float4 y_vec) {
    float4 result;
    result.x = x_vec.x - y_vec.x;
    result.y = x_vec.y - y_vec.y;
    result.z = x_vec.z - y_vec.z;
    result.w = x_vec.w - y_vec.w;
    return result;
}

// write_vec_huge: 使用 atomicAdd 累加 8x8 结果（无边界检查）
template<int gamma>
__device__ __forceinline__ void write_vec_huge(float* C, float4* t, int ldc);
    
template<>
__device__ __forceinline__ void write_vec_huge<0>(float* C, float4* t, int ldc) {
}

template<>
__device__ __forceinline__ void write_vec_huge<1>(float* C, float4* t, int ldc) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float val = (j == 0) ? t[i*2].x : (j == 1 ? t[i*2].y : (j == 2 ? t[i*2].z : t[i*2].w));
            atomicAdd(C + i + (long long)j * ldc, val);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float val = (j == 0) ? t[i*2+1].x : (j == 1 ? t[i*2+1].y : (j == 2 ? t[i*2+1].z : t[i*2+1].w));
            atomicAdd(C + i + (long long)(j + 4) * ldc, val);
        }
    }
}

template<>
__device__ __forceinline__ void write_vec_huge<-1>(float* C, float4* t, int ldc) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float val = (j == 0) ? t[i*2].x : (j == 1 ? t[i*2].y : (j == 2 ? t[i*2].z : t[i*2].w));
            atomicAdd(C + i + (long long)j * ldc, -val);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float val = (j == 0) ? t[i*2+1].x : (j == 1 ? t[i*2+1].y : (j == 2 ? t[i*2+1].z : t[i*2+1].w));
            atomicAdd(C + i + (long long)(j + 4) * ldc, -val);
        }
    }
}
template<>
__device__ __forceinline__ void write_vec_huge<2>(float* C, float4* t, int ldc) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float* col_ptr = C + (long long)i * ldc;
        *((float4*)(col_ptr + 0)) = t[i * 2 + 0];
        *((float4*)(col_ptr + 4)) = t[i * 2 + 1];
    }
}
template<>
__device__ __forceinline__ void write_vec_huge<-2>(float* C, float4* t, int ldc) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float4 neg0, neg1;
        neg0.x = -t[i * 2 + 0].x; neg0.y = -t[i * 2 + 0].y; neg0.z = -t[i * 2 + 0].z; neg0.w = -t[i * 2 + 0].w;
        neg1.x = -t[i * 2 + 1].x; neg1.y = -t[i * 2 + 1].y; neg1.z = -t[i * 2 + 1].z; neg1.w = -t[i * 2 + 1].w;
        float* col_ptr = C + (long long)i * ldc;
        *((float4*)(col_ptr + 0)) = neg0;
        *((float4*)(col_ptr + 4)) = neg1;
    }
}

/*
 * primitive_huge: 优化版本（无边界检查）
 * 
 * Tile 大小: ms=128, ns=128, ks=8
 * 线程数: 256
 * 要求: M % 128 == 0, N % 128 == 0, K % 8 == 0
 */
template <int delta, int epsilon, int gamma0, int gamma1>
__global__ __launch_bounds__(256, 2)
void primitive_huge(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const float* __restrict__ V,
    const float* __restrict__ W,
    float* __restrict__ D,
    float* __restrict__ E,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde
) {
    // Shared memory for double buffering
    __shared__ float shared_A[2][1024];
    __shared__ float shared_B[2][1024];
    float* sa = (float*)shared_A;
    float* sb = (float*)shared_B;
    
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    
    // Thread indexing
    int wid = (tx >> 5);  // warp id (0-7)
    int wid_b = (wid >> 2);  // warp id for B dimension (0-1)
    int wid_a = (wid & 3);   // warp id for A dimension (0-3)
    int inter_warp_id_b = ((tx & 31) >> 2);  // thread id within warp for B (0-7)
    int inter_warp_id_a = ((tx & 31) & 3);   // thread id within warp for A (0-3)
    
    // Output position for this thread
    int i1 = (wid_b << 6) + (inter_warp_id_b << 3) + (bx << 7);  // row index (M)
    int j1 = (wid_a << 5) + (inter_warp_id_a << 3) + (by << 7);  // col index (N)
    
    // X, Y 加载索引（列主序 [M, K]）
    int idx = tx & 31;  // lane id within warp (0-31)
    int idy = tx >> 5;  // warp id (0-7)
    int idx_4 = (idx << 2);  // idx * 4
    
    // V, W 加载索引（列主序 [K, N]）
    int col_B = tx >> 1;     // N 维度: 0-127
    int row_B = (tx & 1) << 2; // K 维度: 0 or 4
    
    // 第一次加载（无边界检查）
    float4 pre_X = *((float4*)(X + idx_4 + (bx << 7) + (long long)idy * ldx));
    float4 pre_Y = *((float4*)(Y + idx_4 + (bx << 7) + (long long)idy * ldy));
    float4 pre_V = *((float4*)(V + row_B + (long long)(by * 128 + col_B) * ldv));
    float4 pre_W = *((float4*)(W + row_B + (long long)(by * 128 + col_B) * ldw));
    
    // Store to shared memory
    ((float4*)sa)[tx] = read_vec_huge<delta>(pre_X, pre_Y);
    
    float4 val_V = read_vec_huge<epsilon>(pre_V, pre_W);
    sb[row_B * 128 + col_B] = val_V.x;
    sb[(row_B + 1) * 128 + col_B] = val_V.y;
    sb[(row_B + 2) * 128 + col_B] = val_V.z;
    sb[(row_B + 3) * 128 + col_B] = val_V.w;
    
    __syncthreads();
    
    // Shared memory offsets for computation
    int offset_vec_A_warp = (wid_b << 6);  // wid_b * 64
    int offset_vec_A_thread = (inter_warp_id_b << 3);  // inter_warp_id_b * 8
    int offset_vec_B_warp = (wid_a << 5);  // wid_a * 32
    int offset_vec_B_thread = (inter_warp_id_a << 3);  // inter_warp_id_a * 8

    // Accumulator and prefetch registers
    float4 t[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        t[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float4 aa[2], bb[2];
    
    // Prefetch first vectors from shared memory
    bb[0] = *(float4*)(sb + offset_vec_B_warp + offset_vec_B_thread);
    bb[1] = *(float4*)(sb + offset_vec_B_warp + offset_vec_B_thread + 4);
    aa[0] = *(float4*)(sa + offset_vec_A_warp + offset_vec_A_thread);
    aa[1] = *(float4*)(sa + offset_vec_A_warp + offset_vec_A_thread + 4);

    // K loop
    for(int k = 0; k < K; k += 8){
        int shared_offset = ((((k >> 3) + 1) & 1) << 10);
        
        // 预取下一个 tile（无边界检查，因为 K 是 8 的倍数）
        if (k + 8 < K) {
            int next_k = k + 8;
            pre_X = *((float4*)(X + idx_4 + (bx << 7) + (long long)(next_k + idy) * ldx));
            pre_Y = *((float4*)(Y + idx_4 + (bx << 7) + (long long)(next_k + idy) * ldy));
            pre_V = *((float4*)(V + (next_k + row_B) + (long long)(by * 128 + col_B) * ldv));
            pre_W = *((float4*)(W + (next_k + row_B) + (long long)(by * 128 + col_B) * ldw));
        }
        
        #pragma unroll
        for(int kk = 0; kk < 8; kk += 1){
            const int kk_ = ((kk + 1) & 7);
            const float4 bb_next0 = *(float4*)(sb + offset_vec_B_warp + offset_vec_B_thread + (kk_ << 7));
            const float4 bb_next1 = *(float4*)(sb + offset_vec_B_warp + offset_vec_B_thread + 4 + (kk_ << 7));
            const float4 aa_next0 = *(float4*)(sa + offset_vec_A_warp + offset_vec_A_thread + (kk_ << 7));
            const float4 aa_next1 = *(float4*)(sa + offset_vec_A_warp + offset_vec_A_thread + 4 + (kk_ << 7));
            
            tab(t[0], bb[0], aa[0].x);
            tab(t[1], bb[1], aa[0].x);
            tab(t[2], bb[0], aa[0].y);
            tab(t[3], bb[1], aa[0].y);
            tab(t[4], bb[0], aa[0].z);
            tab(t[5], bb[1], aa[0].z);
            tab(t[6], bb[0], aa[0].w);
            tab(t[7], bb[1], aa[0].w);

            tab(t[8], bb[0], aa[1].x);
            tab(t[9], bb[1], aa[1].x);
            tab(t[10], bb[0], aa[1].y);
            tab(t[11], bb[1], aa[1].y);
            tab(t[12], bb[0], aa[1].z);
            tab(t[13], bb[1], aa[1].z);
            tab(t[14], bb[0], aa[1].w);
            tab(t[15], bb[1], aa[1].w);

            aa[0] = aa_next0;
            aa[1] = aa_next1;
            bb[0] = bb_next0;
            bb[1] = bb_next1;
        }
        
        sb = (float*)shared_B + shared_offset;
        sa = (float*)shared_A + shared_offset;
        
        ((float4*)sa)[tx] = read_vec_huge<delta>(pre_X, pre_Y);
        
        float4 val_V_next = read_vec_huge<epsilon>(pre_V, pre_W);
        sb[row_B * 128 + col_B] = val_V_next.x;
        sb[(row_B + 1) * 128 + col_B] = val_V_next.y;
        sb[(row_B + 2) * 128 + col_B] = val_V_next.z;
        sb[(row_B + 3) * 128 + col_B] = val_V_next.w;
        
        __syncthreads();
        
        aa[0] = *(float4*)(sa + offset_vec_A_warp + offset_vec_A_thread);
        aa[1] = *(float4*)(sa + offset_vec_A_warp + offset_vec_A_thread + 4);
        bb[0] = *(float4*)(sb + offset_vec_B_warp + offset_vec_B_thread);
        bb[1] = *(float4*)(sb + offset_vec_B_warp + offset_vec_B_thread + 4);
    }

    // 写回结果（无边界检查）
    write_vec_huge<gamma0>(D + i1 + (long long)j1 * ldd, t, ldd);
    write_vec_huge<gamma1>(E + i1 + (long long)j1 * lde, t, lde);
}

#endif 
