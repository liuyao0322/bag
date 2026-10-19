#ifndef PRIM_MEDIUM_CUH
#define PRIM_MEDIUM_CUH

#include "base_components.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void prefetch_global_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}
#else
__device__ __forceinline__ void prefetch_global_l2(const void* ptr) {
    (void)ptr;
}
#endif

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

// 融合加载和计算的辅助函数 - 用于 A (float2 / float4)
template<int alpha>
__device__ __forceinline__ void read_vec_medium2(
    float2* a_ptr, float2 x_vec, float2 y_vec);

template<>
__device__ __forceinline__ void read_vec_medium2<0>(
    float2* a_ptr, float2 x_vec, float2 y_vec) {
    a_ptr->x = x_vec.x;
    a_ptr->y = x_vec.y;
}

template<>
__device__ __forceinline__ void read_vec_medium2<1>(
    float2* a_ptr, float2 x_vec, float2 y_vec) {
    a_ptr->x = x_vec.x + y_vec.x;
    a_ptr->y = x_vec.y + y_vec.y;
}

template<>
__device__ __forceinline__ void read_vec_medium2<-1>(
    float2* a_ptr, float2 x_vec, float2 y_vec) {
    a_ptr->x = x_vec.x - y_vec.x;
    a_ptr->y = x_vec.y - y_vec.y;
}

template<int alpha>
__device__ __forceinline__ void read_vec_medium(
    float4* a_ptr, float4 x_vec, float4 y_vec);

template<>
__device__ __forceinline__ void read_vec_medium<0>(
    float4* a_ptr, float4 x_vec, float4 y_vec) {
    a_ptr->x = x_vec.x;
    a_ptr->y = x_vec.y;
    a_ptr->z = x_vec.z;
    a_ptr->w = x_vec.w;      
}

template<>
__device__ __forceinline__ void read_vec_medium<1>(
    float4* a_ptr, float4 x_vec, float4 y_vec) {
    a_ptr->x = x_vec.x + y_vec.x;
    a_ptr->y = x_vec.y + y_vec.y;
    a_ptr->z = x_vec.z + y_vec.z;
    a_ptr->w = x_vec.w + y_vec.w;
}

template<>
__device__ __forceinline__ void read_vec_medium<-1>(
    float4* a_ptr, float4 x_vec, float4 y_vec) {
    a_ptr->x = x_vec.x - y_vec.x;
    a_ptr->y = x_vec.y - y_vec.y;
    a_ptr->z = x_vec.z - y_vec.z;
    a_ptr->w = x_vec.w - y_vec.w;
}

// 散列存储 (用于 B) - 将 float4 存储到 [k][n] 布局
// buffer[k * stride + col] 布局，一次存储 4 个连续的 k 值
__device__ __forceinline__ int swizzle_b_group_col_medium(int row, int col) {
    const int group = col >> 2;
    const int lane = col & 3;
    const int swizzled_group = group ^ (row & 1);
    return (swizzled_group << 2) + lane;
}

template<int alpha>
__device__ __forceinline__ void store_vec_B_group_swizzle_medium(
    float* buffer, float4 x_vec, float4 y_vec, int row, int col, int stride);

template<>
__device__ __forceinline__ void store_vec_B_group_swizzle_medium<0>(
    float* buffer, float4 x_vec, float4 y_vec, int row, int col, int stride) {
    buffer[row * stride + swizzle_b_group_col_medium(row, col)] = x_vec.x;
    buffer[(row + 1) * stride + swizzle_b_group_col_medium(row + 1, col)] = x_vec.y;
    buffer[(row + 2) * stride + swizzle_b_group_col_medium(row + 2, col)] = x_vec.z;
    buffer[(row + 3) * stride + swizzle_b_group_col_medium(row + 3, col)] = x_vec.w;
}

template<>
__device__ __forceinline__ void store_vec_B_group_swizzle_medium<1>(
    float* buffer, float4 x_vec, float4 y_vec, int row, int col, int stride) {
    buffer[row * stride + swizzle_b_group_col_medium(row, col)] = x_vec.x + y_vec.x;
    buffer[(row + 1) * stride + swizzle_b_group_col_medium(row + 1, col)] = x_vec.y + y_vec.y;
    buffer[(row + 2) * stride + swizzle_b_group_col_medium(row + 2, col)] = x_vec.z + y_vec.z;
    buffer[(row + 3) * stride + swizzle_b_group_col_medium(row + 3, col)] = x_vec.w + y_vec.w;
}

template<>
__device__ __forceinline__ void store_vec_B_group_swizzle_medium<-1>(
    float* buffer, float4 x_vec, float4 y_vec, int row, int col, int stride) {
    buffer[row * stride + swizzle_b_group_col_medium(row, col)] = x_vec.x - y_vec.x;
    buffer[(row + 1) * stride + swizzle_b_group_col_medium(row + 1, col)] = x_vec.y - y_vec.y;
    buffer[(row + 2) * stride + swizzle_b_group_col_medium(row + 2, col)] = x_vec.z - y_vec.z;
    buffer[(row + 3) * stride + swizzle_b_group_col_medium(row + 3, col)] = x_vec.w - y_vec.w;
}

template<int alpha>
__device__ __forceinline__ void store_vec_B_strided_medium(
    float* buffer, float4 x_vec, float4 y_vec, int row, int col, int stride);

template<>
__device__ __forceinline__ void store_vec_B_strided_medium<0>(
    float* buffer, float4 x_vec, float4 y_vec, int row, int col, int stride) {
    buffer[row * stride + col] = x_vec.x;
    buffer[(row + 1) * stride + col] = x_vec.y;
    buffer[(row + 2) * stride + col] = x_vec.z;
    buffer[(row + 3) * stride + col] = x_vec.w;
}

template<>
__device__ __forceinline__ void store_vec_B_strided_medium<1>(
    float* buffer, float4 x_vec, float4 y_vec, int row, int col, int stride) {
    buffer[row * stride + col] = x_vec.x + y_vec.x;
    buffer[(row + 1) * stride + col] = x_vec.y + y_vec.y;
    buffer[(row + 2) * stride + col] = x_vec.z + y_vec.z;
    buffer[(row + 3) * stride + col] = x_vec.w + y_vec.w;
}

template<>
__device__ __forceinline__ void store_vec_B_strided_medium<-1>(
    float* buffer, float4 x_vec, float4 y_vec, int row, int col, int stride) {
    buffer[row * stride + col] = x_vec.x - y_vec.x;
    buffer[(row + 1) * stride + col] = x_vec.y - y_vec.y;
    buffer[(row + 2) * stride + col] = x_vec.z - y_vec.z;
    buffer[(row + 3) * stride + col] = x_vec.w - y_vec.w;
}

template<int alpha>
__device__ __forceinline__ void store_vec_B_strided_medium2(
    float* buffer, float2 x_vec, float2 y_vec, int row, int col, int stride);

template<>
__device__ __forceinline__ void store_vec_B_strided_medium2<0>(
    float* buffer, float2 x_vec, float2 y_vec, int row, int col, int stride) {
    buffer[row * stride + col] = x_vec.x;
    buffer[(row + 1) * stride + col] = x_vec.y;
}

template<>
__device__ __forceinline__ void store_vec_B_strided_medium2<1>(
    float* buffer, float2 x_vec, float2 y_vec, int row, int col, int stride) {
    buffer[row * stride + col] = x_vec.x + y_vec.x;
    buffer[(row + 1) * stride + col] = x_vec.y + y_vec.y;
}

template<>
__device__ __forceinline__ void store_vec_B_strided_medium2<-1>(
    float* buffer, float2 x_vec, float2 y_vec, int row, int col, int stride) {
    buffer[row * stride + col] = x_vec.x - y_vec.x;
    buffer[(row + 1) * stride + col] = x_vec.y - y_vec.y;
}

// write_vec_medium: 使用 atomicAdd 累加结果
template<int gamma>
__device__ __forceinline__ void write_vec_medium(
    float* C, float* res, int ldc);

template<int gamma>
__device__ __forceinline__ void write_vec_medium_vec(
    float* C, const float4* acc, int ldc);
template<>
__device__ __forceinline__ void write_vec_medium<0>(
    float* C, float* res, int ldc) {
}
template<>
__device__ __forceinline__ void write_vec_medium<1>(
    float* C, float* res, int ldc) {
    // 4x4 输出块，按列主序写入
    atomicAdd(C, res[0]);
    atomicAdd(C + ldc, res[1]);
    atomicAdd(C + 2 * ldc, res[2]);
    atomicAdd(C + 3 * ldc, res[3]);
    
    atomicAdd(C + 1, res[4]);
    atomicAdd(C + 1 + ldc, res[5]);
    atomicAdd(C + 1 + 2 * ldc, res[6]);
    atomicAdd(C + 1 + 3 * ldc, res[7]);
    
    atomicAdd(C + 2, res[8]);
    atomicAdd(C + 2 + ldc, res[9]);
    atomicAdd(C + 2 + 2 * ldc, res[10]);
    atomicAdd(C + 2 + 3 * ldc, res[11]);
    
    atomicAdd(C + 3, res[12]);
    atomicAdd(C + 3 + ldc, res[13]);
    atomicAdd(C + 3 + 2 * ldc, res[14]);
    atomicAdd(C + 3 + 3 * ldc, res[15]);
}
template<>
__device__ __forceinline__ void write_vec_medium<-1>(
    float* C, float* res, int ldc) {
    atomicAdd(C, -res[0]);
    atomicAdd(C + ldc, -res[1]);
    atomicAdd(C + 2 * ldc, -res[2]);
    atomicAdd(C + 3 * ldc, -res[3]);
    
    atomicAdd(C + 1, -res[4]);
    atomicAdd(C + 1 + ldc, -res[5]);
    atomicAdd(C + 1 + 2 * ldc, -res[6]);
    atomicAdd(C + 1 + 3 * ldc, -res[7]);
    
    atomicAdd(C + 2, -res[8]);
    atomicAdd(C + 2 + ldc, -res[9]);
    atomicAdd(C + 2 + 2 * ldc, -res[10]);
    atomicAdd(C + 2 + 3 * ldc, -res[11]);
    
    atomicAdd(C + 3, -res[12]);
    atomicAdd(C + 3 + ldc, -res[13]);
    atomicAdd(C + 3 + 2 * ldc, -res[14]);
    atomicAdd(C + 3 + 3 * ldc, -res[15]);
}
// gamma=2: 直接写（覆盖）
template<>
__device__ __forceinline__ void write_vec_medium<2>(
    float* C, float* res, int ldc) {
    *((float4*)(C + 0)) = make_float4(res[0], res[4], res[8], res[12]);
    *((float4*)(C + ldc)) = make_float4(res[1], res[5], res[9], res[13]);
    *((float4*)(C + 2 * ldc)) = make_float4(res[2], res[6], res[10], res[14]);
    *((float4*)(C + 3 * ldc)) = make_float4(res[3], res[7], res[11], res[15]);
}
// gamma=-2: 直接写负值
template<>
__device__ __forceinline__ void write_vec_medium<-2>(
    float* C, float* res, int ldc) {
    *((float4*)(C + 0)) = make_float4(-res[0], -res[4], -res[8], -res[12]);
    *((float4*)(C + ldc)) = make_float4(-res[1], -res[5], -res[9], -res[13]);
    *((float4*)(C + 2 * ldc)) = make_float4(-res[2], -res[6], -res[10], -res[14]);
    *((float4*)(C + 3 * ldc)) = make_float4(-res[3], -res[7], -res[11], -res[15]);
}

template<>
__device__ __forceinline__ void write_vec_medium_vec<0>(
    float* C, const float4* acc, int ldc) {
}

template<>
__device__ __forceinline__ void write_vec_medium_vec<2>(
    float* C, const float4* acc, int ldc) {
    *((float4*)(C + 0)) = acc[0];
    *((float4*)(C + ldc)) = acc[1];
    *((float4*)(C + 2 * ldc)) = acc[2];
    *((float4*)(C + 3 * ldc)) = acc[3];
}

template<>
__device__ __forceinline__ void write_vec_medium_vec<-2>(
    float* C, const float4* acc, int ldc) {
    *((float4*)(C + 0)) = make_float4(-acc[0].x, -acc[0].y, -acc[0].z, -acc[0].w);
    *((float4*)(C + ldc)) = make_float4(-acc[1].x, -acc[1].y, -acc[1].z, -acc[1].w);
    *((float4*)(C + 2 * ldc)) = make_float4(-acc[2].x, -acc[2].y, -acc[2].z, -acc[2].w);
    *((float4*)(C + 3 * ldc)) = make_float4(-acc[3].x, -acc[3].y, -acc[3].z, -acc[3].w);
}

template<>
__device__ __forceinline__ void write_vec_medium_vec<1>(
    float* C, const float4* acc, int ldc) {
    atomicAdd(C + 0, acc[0].x);
    atomicAdd(C + 1, acc[0].y);
    atomicAdd(C + 2, acc[0].z);
    atomicAdd(C + 3, acc[0].w);

    atomicAdd(C + ldc + 0, acc[1].x);
    atomicAdd(C + ldc + 1, acc[1].y);
    atomicAdd(C + ldc + 2, acc[1].z);
    atomicAdd(C + ldc + 3, acc[1].w);

    atomicAdd(C + 2 * ldc + 0, acc[2].x);
    atomicAdd(C + 2 * ldc + 1, acc[2].y);
    atomicAdd(C + 2 * ldc + 2, acc[2].z);
    atomicAdd(C + 2 * ldc + 3, acc[2].w);

    atomicAdd(C + 3 * ldc + 0, acc[3].x);
    atomicAdd(C + 3 * ldc + 1, acc[3].y);
    atomicAdd(C + 3 * ldc + 2, acc[3].z);
    atomicAdd(C + 3 * ldc + 3, acc[3].w);
}

template<>
__device__ __forceinline__ void write_vec_medium_vec<-1>(
    float* C, const float4* acc, int ldc) {
    atomicAdd(C + 0, -acc[0].x);
    atomicAdd(C + 1, -acc[0].y);
    atomicAdd(C + 2, -acc[0].z);
    atomicAdd(C + 3, -acc[0].w);

    atomicAdd(C + ldc + 0, -acc[1].x);
    atomicAdd(C + ldc + 1, -acc[1].y);
    atomicAdd(C + ldc + 2, -acc[1].z);
    atomicAdd(C + ldc + 3, -acc[1].w);

    atomicAdd(C + 2 * ldc + 0, -acc[2].x);
    atomicAdd(C + 2 * ldc + 1, -acc[2].y);
    atomicAdd(C + 2 * ldc + 2, -acc[2].z);
    atomicAdd(C + 2 * ldc + 3, -acc[2].w);

    atomicAdd(C + 3 * ldc + 0, -acc[3].x);
    atomicAdd(C + 3 * ldc + 1, -acc[3].y);
    atomicAdd(C + 3 * ldc + 2, -acc[3].z);
    atomicAdd(C + 3 * ldc + 3, -acc[3].w);
}

template<int gamma>
__device__ __forceinline__ void write_vec_medium_pair(
    float* C, const float4* acc, int ldc);

template<>
__device__ __forceinline__ void write_vec_medium_pair<0>(
    float* C, const float4* acc, int ldc) {
}

template<>
__device__ __forceinline__ void write_vec_medium_pair<2>(
    float* C, const float4* acc, int ldc) {
    *((float4*)(C + 0)) = acc[0];
    *((float4*)(C + ldc)) = acc[1];
}

template<>
__device__ __forceinline__ void write_vec_medium_pair<-2>(
    float* C, const float4* acc, int ldc) {
    *((float4*)(C + 0)) = make_float4(-acc[0].x, -acc[0].y, -acc[0].z, -acc[0].w);
    *((float4*)(C + ldc)) = make_float4(-acc[1].x, -acc[1].y, -acc[1].z, -acc[1].w);
}

template<>
__device__ __forceinline__ void write_vec_medium_pair<1>(
    float* C, const float4* acc, int ldc) {
    atomicAdd(C + 0, acc[0].x);
    atomicAdd(C + 1, acc[0].y);
    atomicAdd(C + 2, acc[0].z);
    atomicAdd(C + 3, acc[0].w);

    atomicAdd(C + ldc + 0, acc[1].x);
    atomicAdd(C + ldc + 1, acc[1].y);
    atomicAdd(C + ldc + 2, acc[1].z);
    atomicAdd(C + ldc + 3, acc[1].w);
}

template<>
__device__ __forceinline__ void write_vec_medium_pair<-1>(
    float* C, const float4* acc, int ldc) {
    atomicAdd(C + 0, -acc[0].x);
    atomicAdd(C + 1, -acc[0].y);
    atomicAdd(C + 2, -acc[0].z);
    atomicAdd(C + 3, -acc[0].w);

    atomicAdd(C + ldc + 0, -acc[1].x);
    atomicAdd(C + ldc + 1, -acc[1].y);
    atomicAdd(C + ldc + 2, -acc[1].z);
    atomicAdd(C + ldc + 3, -acc[1].w);
}

template<int gamma>
__device__ __forceinline__ void write_vec_medium_row4(
    float* C, const float2* acc, int ldc);

template<>
__device__ __forceinline__ void write_vec_medium_row4<0>(
    float* C, const float2* acc, int ldc) {
}

template<>
__device__ __forceinline__ void write_vec_medium_row4<2>(
    float* C, const float2* acc, int ldc) {
    *((float2*)(C + 0 * ldc)) = acc[0];
    *((float2*)(C + 1 * ldc)) = acc[1];
    *((float2*)(C + 2 * ldc)) = acc[2];
    *((float2*)(C + 3 * ldc)) = acc[3];
}

template<>
__device__ __forceinline__ void write_vec_medium_row4<-2>(
    float* C, const float2* acc, int ldc) {
    *((float2*)(C + 0 * ldc)) = make_float2(-acc[0].x, -acc[0].y);
    *((float2*)(C + 1 * ldc)) = make_float2(-acc[1].x, -acc[1].y);
    *((float2*)(C + 2 * ldc)) = make_float2(-acc[2].x, -acc[2].y);
    *((float2*)(C + 3 * ldc)) = make_float2(-acc[3].x, -acc[3].y);
}

template<>
__device__ __forceinline__ void write_vec_medium_row4<1>(
    float* C, const float2* acc, int ldc) {
    atomicAdd(C + 0 * ldc + 0, acc[0].x);
    atomicAdd(C + 0 * ldc + 1, acc[0].y);
    atomicAdd(C + 1 * ldc + 0, acc[1].x);
    atomicAdd(C + 1 * ldc + 1, acc[1].y);
    atomicAdd(C + 2 * ldc + 0, acc[2].x);
    atomicAdd(C + 2 * ldc + 1, acc[2].y);
    atomicAdd(C + 3 * ldc + 0, acc[3].x);
    atomicAdd(C + 3 * ldc + 1, acc[3].y);
}

template<>
__device__ __forceinline__ void write_vec_medium_row4<-1>(
    float* C, const float2* acc, int ldc) {
    atomicAdd(C + 0 * ldc + 0, -acc[0].x);
    atomicAdd(C + 0 * ldc + 1, -acc[0].y);
    atomicAdd(C + 1 * ldc + 0, -acc[1].x);
    atomicAdd(C + 1 * ldc + 1, -acc[1].y);
    atomicAdd(C + 2 * ldc + 0, -acc[2].x);
    atomicAdd(C + 2 * ldc + 1, -acc[2].y);
    atomicAdd(C + 3 * ldc + 0, -acc[3].x);
    atomicAdd(C + 3 * ldc + 1, -acc[3].y);
}

template<int gamma>
__device__ __forceinline__ void write_vec_medium_pair2(
    float* C, const float2* acc, int ldc);

template<>
__device__ __forceinline__ void write_vec_medium_pair2<0>(
    float* C, const float2* acc, int ldc) {
}

template<>
__device__ __forceinline__ void write_vec_medium_pair2<2>(
    float* C, const float2* acc, int ldc) {
    *((float2*)(C + 0)) = acc[0];
    *((float2*)(C + ldc)) = acc[1];
}

template<>
__device__ __forceinline__ void write_vec_medium_pair2<-2>(
    float* C, const float2* acc, int ldc) {
    *((float2*)(C + 0)) = make_float2(-acc[0].x, -acc[0].y);
    *((float2*)(C + ldc)) = make_float2(-acc[1].x, -acc[1].y);
}

template<>
__device__ __forceinline__ void write_vec_medium_pair2<1>(
    float* C, const float2* acc, int ldc) {
    atomicAdd(C + 0, acc[0].x);
    atomicAdd(C + 1, acc[0].y);
    atomicAdd(C + ldc + 0, acc[1].x);
    atomicAdd(C + ldc + 1, acc[1].y);
}

template<>
__device__ __forceinline__ void write_vec_medium_pair2<-1>(
    float* C, const float2* acc, int ldc) {
    atomicAdd(C + 0, -acc[0].x);
    atomicAdd(C + 1, -acc[0].y);
    atomicAdd(C + ldc + 0, -acc[1].x);
    atomicAdd(C + ldc + 1, -acc[1].y);
}

#define ACCUM_MEDIUM_COL2(acc, avec, bval) \
    do { \
        (acc).x = fmaf((avec).x, (bval), (acc).x); \
        (acc).y = fmaf((avec).y, (bval), (acc).y); \
    } while (0)

#define ACCUM_MEDIUM_COL(acc, avec, bval) \
    do { \
        (acc).x = fmaf((avec).x, (bval), (acc).x); \
        (acc).y = fmaf((avec).y, (bval), (acc).y); \
        (acc).z = fmaf((avec).z, (bval), (acc).z); \
        (acc).w = fmaf((avec).w, (bval), (acc).w); \
    } while (0)

template<int alpha>
__device__ __forceinline__ float4 med_load_ab_kernel(const float* X, const float* Y, long long base) {
    float4 xv = *((const float4*)(X + base));
    if (alpha == 0) return xv;
    float4 yv = *((const float4*)(Y + base));
    float4 r;
    r.x = (alpha == 1) ? xv.x + yv.x : xv.x - yv.x;
    r.y = (alpha == 1) ? xv.y + yv.y : xv.y - yv.y;
    r.z = (alpha == 1) ? xv.z + yv.z : xv.z - yv.z;
    r.w = (alpha == 1) ? xv.w + yv.w : xv.w - yv.w;
    return r;
}

template<int gamma>
__device__ __forceinline__ void med_write_kernel(float* C, int idx, float val) {
    if (gamma == 0) return;
    if (gamma ==  1) atomicAdd(C + idx,  val);
    if (gamma == -1) atomicAdd(C + idx, -val);
    if (gamma ==  2) C[idx] =  val;
    if (gamma == -2) C[idx] = -val;
}

template<int alpha>
__device__ __forceinline__ void med_store_b_tile(float sB[32][9], int n_idx, int k_idx, const float4& v) {
    sB[n_idx][k_idx + 0] = v.x;
    sB[n_idx][k_idx + 1] = v.y;
    sB[n_idx][k_idx + 2] = v.z;
    sB[n_idx][k_idx + 3] = v.w;
}

// Fused medium-size primitive with the shared BAG coefficient interface.

template <int delta, int epsilon, int gamma0, int gamma1>
__global__ __launch_bounds__(64, 2)
void primitive_medium(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const float* __restrict__ V,
    const float* __restrict__ W,
    float* __restrict__ D,
    float* __restrict__ E,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde
) {
    __shared__ float sA[8][32];
    __shared__ float sB[32][9];

    int t  = threadIdx.x;
    int tx = t & 7;
    int ty = t >> 3;

    int a_kk = t >> 3;
    int a_m  = (t & 7) * 4;

    int b_n  = t >> 1;
    int b_kk = (t & 1) * 4;

    float acc[4][4] = {};

    for (int k = 0; k < K; k += 8) {
        long long gm = (long long)blockIdx.x * 32 + a_m;
        *((float4*)(&sA[a_kk][a_m])) = med_load_ab_kernel<delta>(X, Y, gm + (long long)(k + a_kk) * ldx);

        long long gn = (long long)blockIdx.y * 32 + b_n;
        med_store_b_tile<epsilon>(sB, b_n, b_kk,
            med_load_ab_kernel<epsilon>(V, W, (k + b_kk) + gn * ldv));

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            float a0 = sA[kk][tx*4+0], a1 = sA[kk][tx*4+1];
            float a2 = sA[kk][tx*4+2], a3 = sA[kk][tx*4+3];
            float b0 = sB[ty*4+0][kk], b1 = sB[ty*4+1][kk];
            float b2 = sB[ty*4+2][kk], b3 = sB[ty*4+3][kk];
            acc[0][0] = fmaf(a0, b0, acc[0][0]); acc[0][1] = fmaf(a0, b1, acc[0][1]);
            acc[0][2] = fmaf(a0, b2, acc[0][2]); acc[0][3] = fmaf(a0, b3, acc[0][3]);
            acc[1][0] = fmaf(a1, b0, acc[1][0]); acc[1][1] = fmaf(a1, b1, acc[1][1]);
            acc[1][2] = fmaf(a1, b2, acc[1][2]); acc[1][3] = fmaf(a1, b3, acc[1][3]);
            acc[2][0] = fmaf(a2, b0, acc[2][0]); acc[2][1] = fmaf(a2, b1, acc[2][1]);
            acc[2][2] = fmaf(a2, b2, acc[2][2]); acc[2][3] = fmaf(a2, b3, acc[2][3]);
            acc[3][0] = fmaf(a3, b0, acc[3][0]); acc[3][1] = fmaf(a3, b1, acc[3][1]);
            acc[3][2] = fmaf(a3, b2, acc[3][2]); acc[3][3] = fmaf(a3, b3, acc[3][3]);
        }

        __syncthreads();
    }

    int row_base = blockIdx.x * 32 + tx * 4;
    int col_base = blockIdx.y * 32 + ty * 4;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        long long col_off_d = (long long)(col_base + j) * ldd;
        long long col_off_e = (long long)(col_base + j) * lde;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float val = acc[i][j];
            med_write_kernel<gamma0>(D, row_base + i + col_off_d, val);
            med_write_kernel<gamma1>(E, row_base + i + col_off_e, val);
        }
    }
}

template <int delta, int epsilon, int gamma0, int gamma1>
__global__ __launch_bounds__(128)
void primitive_medium_batched_pair(
    const PrimBatchedPairArgs args,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde
) {
    const int ms = 32, ns = 32, ks = 8;
    const int mw = 16, nw = 16, mr = 4, nr = 2;

    const int batch_id = blockIdx.z;
    const float* X = (batch_id == 0) ? args.X0 : args.X1;
    const float* Y = (batch_id == 0) ? args.Y0 : args.Y1;
    const float* V = (batch_id == 0) ? args.V0 : args.V1;
    const float* W = (batch_id == 0) ? args.W0 : args.W1;
    float* D = (batch_id == 0) ? args.D0 : args.D1;
    float* E = (batch_id == 0) ? args.E0 : args.E1;

    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;

    __shared__ float sAB[2 * ks * ms + 2 * ks * ns];

    const int buffer_A_offset = 0;
    const int buffer_B_offset = 2 * ks * ms;

    X += bx * ms;
    Y += bx * ms;
    V += by * ns * ldv;
    W += by * ns * ldw;

    const bool loader = tx < 64;
    const int float4_per_col_A = ms / 4;
    const int float4_per_col_B = ks / 4;
    const int load_row_A = (tx % float4_per_col_A) * 4;
    const int load_col_A = tx / float4_per_col_A;
    const int load_row_B = (tx % float4_per_col_B) * 4;
    const int load_col_B = tx / float4_per_col_B;

    const float* tile_X = X;
    const float* tile_Y = Y;
    const float* tile_V = V;
    const float* tile_W = W;

    int ping = 0;
    float* buffer_A = sAB + buffer_A_offset + ping * ks * ms;
    float* buffer_B = sAB + buffer_B_offset + ping * ks * ns;

    float4 prefetch_X = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 prefetch_Y = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 prefetch_V = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 prefetch_W = make_float4(0.f, 0.f, 0.f, 0.f);
    if (loader) {
        const float4* x_ptr = reinterpret_cast<const float4*>(tile_X + load_row_A + load_col_A * ldx);
        const float4* y_ptr = reinterpret_cast<const float4*>(tile_Y + load_row_A + load_col_A * ldy);
        const float4* v_ptr = reinterpret_cast<const float4*>(tile_V + load_row_B + load_col_B * ldv);
        const float4* w_ptr = reinterpret_cast<const float4*>(tile_W + load_row_B + load_col_B * ldw);
        prefetch_X = *x_ptr;
        prefetch_Y = *y_ptr;
        prefetch_V = *v_ptr;
        prefetch_W = *w_ptr;
    }

    if (loader) {
        float4* store_A = reinterpret_cast<float4*>(buffer_A + load_col_A * ms + load_row_A);
        read_vec_medium<delta>(store_A, prefetch_X, prefetch_Y);
        store_vec_B_strided_medium<epsilon>(buffer_B, prefetch_V, prefetch_W, load_row_B, load_col_B, ns);
    }

    __syncthreads();

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

    float4 vec_A[2];
    float2 vec_B[2];
    float4 acc[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float* a_lane_base = buffer_A + warp_offset_A + thread_offset_A;
    float* b_lane_base = buffer_B + warp_offset_B + thread_offset_B;
    vec_A[0] = *(const float4*)a_lane_base;
    vec_B[0] = *(const float2*)b_lane_base;

    for (int k_outer = 0; k_outer < K; k_outer += ks) {
        tile_X += ks * ldx;
        tile_Y += ks * ldy;
        tile_V += ks;
        tile_W += ks;

        if (k_outer + ks < K && loader) {
            prefetch_global_l2(tile_X + ks * ldx);
            prefetch_global_l2(tile_Y + ks * ldy);
            prefetch_global_l2(tile_V + ks);
            prefetch_global_l2(tile_W + ks);
            const float4* x_ptr = reinterpret_cast<const float4*>(tile_X + load_row_A + load_col_A * ldx);
            const float4* y_ptr = reinterpret_cast<const float4*>(tile_Y + load_row_A + load_col_A * ldy);
            const float4* v_ptr = reinterpret_cast<const float4*>(tile_V + load_row_B + load_col_B * ldv);
            const float4* w_ptr = reinterpret_cast<const float4*>(tile_W + load_row_B + load_col_B * ldw);
            prefetch_X = *x_ptr;
            prefetch_Y = *y_ptr;
            prefetch_V = *v_ptr;
            prefetch_W = *w_ptr;
        }

        #pragma unroll
        for (int kk = 0; kk < ks; kk += 2) {
            const float* a_ptr0 = a_lane_base + kk * ms;
            const float* b_ptr0 = b_lane_base + kk * ns;
            const float* a_ptr1 = a_ptr0 + ms;
            const float* b_ptr1 = b_ptr0 + ns;

            vec_A[1] = *(const float4*)a_ptr1;
            vec_B[1] = *(const float2*)b_ptr1;
            ACCUM_MEDIUM_COL(acc[0], vec_A[0], vec_B[0].x);
            ACCUM_MEDIUM_COL(acc[1], vec_A[0], vec_B[0].y);

            ACCUM_MEDIUM_COL(acc[0], vec_A[1], vec_B[1].x);
            ACCUM_MEDIUM_COL(acc[1], vec_A[1], vec_B[1].y);

            if (kk + 2 < ks) {
                vec_A[0] = *(const float4*)(a_ptr1 + ms);
                vec_B[0] = *(const float2*)(b_ptr1 + ns);
            }
        }

        if (k_outer + ks < K) {
            ping = 1 - ping;
            buffer_A = sAB + buffer_A_offset + ping * ks * ms;
            buffer_B = sAB + buffer_B_offset + ping * ks * ns;

            if (loader) {
                float4* store_A = reinterpret_cast<float4*>(buffer_A + load_col_A * ms + load_row_A);
                read_vec_medium<delta>(store_A, prefetch_X, prefetch_Y);
                store_vec_B_strided_medium<epsilon>(buffer_B, prefetch_V, prefetch_W, load_row_B, load_col_B, ns);
            }

            __syncthreads();

            a_lane_base = buffer_A + warp_offset_A + thread_offset_A;
            b_lane_base = buffer_B + warp_offset_B + thread_offset_B;
            vec_A[0] = *(const float4*)a_lane_base;
            vec_B[0] = *(const float2*)b_lane_base;
        }
    }

    int out_m = bx * ms + warp_offset_A + thread_offset_A;
    int out_n = by * ns + warp_offset_B + thread_offset_B;

    D += out_m + out_n * ldd;
    write_vec_medium_pair<gamma0>(D, acc, ldd);

    E += out_m + out_n * lde;
    write_vec_medium_pair<gamma1>(E, acc, lde);
}

inline void prim_inter_batched_pair_same_variant_medium(
    const PrimBatchedPairArgs& args,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde,
    cudaStream_t stream = 0
) {
    dim3 grid((M + 31) / 32, (N + 31) / 32, 2);
    dim3 block(128);
    primitive_medium_batched_pair<-1, -1, 1, 1><<<grid, block, 0, stream>>>(
        args, M, N, K, ldx, ldy, ldv, ldw, ldd, lde);
}

inline void prim_medium(
    const float* X, const float* Y,
    const float* V, const float* W,
    float* D, float* E,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde,
    int delta, int epsilon, int gamma0, int gamma1,
    cudaStream_t stream = 0
) {
    dim3 grid((M + 31) / 32, (N + 31) / 32);
    dim3 block(64);

    if (delta == 1 && epsilon == 0 && gamma0 == 1 && gamma1 == -1) {
        primitive_medium<1, 0, 1, -1><<<grid, block, 0, stream>>>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde);
    } else if (delta == -1 && epsilon == 1 && gamma0 == 0 && gamma1 == 1) {
        primitive_medium<-1, 1, 0, 1><<<grid, block, 0, stream>>>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde);
    } else if (delta == -1 && epsilon == 1 && gamma0 == 1 && gamma1 == 0) {
        primitive_medium<-1, 1, 1, 0><<<grid, block, 0, stream>>>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde);
    } else if (delta == 0 && epsilon == -1 && gamma0 == 1 && gamma1 == 1) {
        primitive_medium<0, -1, 1, 1><<<grid, block, 0, stream>>>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde);
    } else if (delta == 0 && epsilon == 1 && gamma0 == 1 && gamma1 == 1) {
        primitive_medium<0, 1, 1, 1><<<grid, block, 0, stream>>>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde);
    } else if (delta == 1 && epsilon == 1 && gamma0 == 1 && gamma1 == 1) {
        primitive_medium<1, 1, 1, 1><<<grid, block, 0, stream>>>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde);
    } else if (delta == -1 && epsilon == -1 && gamma0 == 1 && gamma1 == 1) {
        primitive_medium<-1, -1, 1, 1><<<grid, block, 0, stream>>>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde);
    } else if (delta == -1 && epsilon == -1 && gamma0 == -1 && gamma1 == -1) {
        primitive_medium<-1, -1, -1, -1><<<grid, block, 0, stream>>>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde);
    } else {
        primitive_medium<1, 1, 1, 1><<<grid, block, 0, stream>>>(X, Y, V, W, D, E, M, N, K, ldx, ldy, ldv, ldw, ldd, lde);
    }
}

#endif 
