#ifndef BASE_COMPONENTS_CUH
#define BASE_COMPONENTS_CUH

#include <cuda_runtime.h>
#include <cstdint>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
#include <cuda_fp16.h>
#endif  

struct PrimBatchedPairArgs {
    const float* X0;
    const float* Y0;
    const float* V0;
    const float* W0;
    float* D0;
    float* E0;
    const float* X1;
    const float* Y1;
    const float* V1;
    const float* W1;
    float* D1;
    float* E1;
};

#define strassen_tab(t, a, b) \
    t.x += a.x * b; \
    t.y += a.y * b; \
    t.z += a.z * b; \
    t.w += a.w * b;

#define strassen_tcab(t, c, alpha, beta) \
    c.x = alpha * t.x + beta * c.x; \
    c.y = alpha * t.y + beta * c.y; \
    c.z = alpha * t.z + beta * c.z; \
    c.w = alpha * t.w + beta * c.w;

// read_vec: Combine two float4 vectors based on template parameter
template<int alpha>
__device__ __forceinline__ void read_vec(
    float4* a_ptr, float4 x_vec, float4 y_vec);

template<>
__device__ __forceinline__ void read_vec<0>(
    float4* a_ptr, float4 x_vec, float4 y_vec) {
    a_ptr->x = x_vec.x;
    a_ptr->y = x_vec.y;
    a_ptr->z = x_vec.z;
    a_ptr->w = x_vec.w;      
}

template<>
__device__ __forceinline__ void read_vec<1>(
    float4* a_ptr, float4 x_vec, float4 y_vec) {
    a_ptr->x = x_vec.x + y_vec.x;
    a_ptr->y = x_vec.y + y_vec.y;
    a_ptr->z = x_vec.z + y_vec.z;
    a_ptr->w = x_vec.w + y_vec.w;
}

template<>
__device__ __forceinline__ void read_vec<-1>(
    float4* a_ptr, float4 x_vec, float4 y_vec) {
    a_ptr->x = x_vec.x - y_vec.x;
    a_ptr->y = x_vec.y - y_vec.y;
    a_ptr->z = x_vec.z - y_vec.z;
    a_ptr->w = x_vec.w - y_vec.w;
}

// write_vec: Write results with atomic operations
template<int gamma>
__device__ __forceinline__ void write_vec(
    float* C, float* res, int ldc);

template<>
__device__ __forceinline__ void write_vec<0>(
    float* C, float* res, int ldc) {
}

template<>
__device__ __forceinline__ void write_vec<1>(
    float* C, float* res, int ldc) {
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
__device__ __forceinline__ void write_vec<-1>(
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

#endif // BASE_COMPONENTS_CUH
