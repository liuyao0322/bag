#pragma once
#include "strassen.cuh"
#include <algorithm>

namespace strassen_cuda {
// Reuse three compact half-size matrices across the seven outer products.
// All writes/reads of this scratch are ordered through the caller stream.
struct HybridWorkspace {
    float *a=nullptr,*b=nullptr,*m=nullptr;
    int n;
    explicit HybridWorkspace(int size):n(size) {
        if(n<=0||n%16)throw std::runtime_error("hybrid requires N divisible by 16");
        const size_t bytes=size_t(n/2)*(n/2)*sizeof(float);
        checked(cudaMalloc(&a,bytes));checked(cudaMalloc(&b,bytes));checked(cudaMalloc(&m,bytes));
    }
    ~HybridWorkspace(){cudaFree(a);cudaFree(b);cudaFree(m);}
    HybridWorkspace(const HybridWorkspace&)=delete;
    HybridWorkspace& operator=(const HybridWorkspace&)=delete;
    size_t bytes()const{return 3*size_t(n/2)*(n/2)*sizeof(float);}
};

template<int Delta,int Epsilon>
__global__ void hybrid_prepare(const float* x,const float* y,const float* v,const float* w,
                               float* a,float* b,float* m,int s,int ld) {
    const size_t count=size_t(s)*s;
    for(size_t q=size_t(blockIdx.x)*blockDim.x+threadIdx.x;q<count;q+=size_t(blockDim.x)*gridDim.x) {
        const size_t src=q%s+(q/s)*ld;
        float av=x[src],bv=v[src];
        if constexpr(Delta!=0)av+=Delta*y[src];
        if constexpr(Epsilon!=0)bv+=Epsilon*w[src];
        a[q]=av;b[q]=bv;m[q]=0.f;
    }
}
template<int GammaD,int GammaE>
__global__ void hybrid_merge(const float* m,float* d,float* e,int s,int ld) {
    const size_t count=size_t(s)*s;
    for(size_t q=size_t(blockIdx.x)*blockDim.x+threadIdx.x;q<count;q+=size_t(blockDim.x)*gridDim.x) {
        const size_t dst=q%s+(q/s)*ld;
        const float value=m[q];
        if constexpr(GammaD!=0)d[dst]+=GammaD*value;
        if constexpr(GammaE!=0)e[dst]+=GammaE*value;
    }
}
template<int Delta,int Epsilon,int GammaD,int GammaE>
inline void hybrid_product(Operand p,Backend backend,Schedule schedule,int n,cudaStream_t caller,
                           Workspace& inner,HybridWorkspace& scratch) {
    const int s=n/2;
    const unsigned blocks=unsigned(std::min<size_t>((size_t(s)*s+255)/256,4096));
    // M must be reset for every product, including repeated top-level calls.
    // This initialization and both transforms are part of the timed algorithm.
    hybrid_prepare<Delta,Epsilon><<<blocks,256,0,caller>>>
        (p.x,p.y,p.v,p.w,scratch.a,scratch.b,scratch.m,s,n);
    one_level(backend,schedule,scratch.a,scratch.b,scratch.m,s,caller,inner);
    hybrid_merge<GammaD,GammaE><<<blocks,256,0,caller>>>(scratch.m,p.d,p.e,s,n);
}
inline void hybrid_two_level(Backend backend,Schedule schedule,const float* a,const float* b,
                             float* c,int n,cudaStream_t caller,Workspace& inner,HybridWorkspace& scratch) {
    if(n!=scratch.n||n%16||!valid_shape(backend,n/4))
        throw std::runtime_error("invalid hybrid workspace or leaf tile shape");
    if(schedule!=Schedule::SevenAtomic&&schedule!=Schedule::FourBatchedAtomic)
        throw std::runtime_error("hybrid supports seven-stream or four-batched inner calls");
    for(int i=0;i<7;++i) {
        const auto p=operands(i,a,b,c,n);
        switch(i) {
        case 0:case 1:hybrid_product<1,0,1,-1>(p,backend,schedule,n,caller,inner,scratch);break;
        case 2:hybrid_product<-1,1,0,1>(p,backend,schedule,n,caller,inner,scratch);break;
        case 3:hybrid_product<-1,1,1,0>(p,backend,schedule,n,caller,inner,scratch);break;
        case 4:case 5:hybrid_product<0,-1,1,1>(p,backend,schedule,n,caller,inner,scratch);break;
        default:hybrid_product<1,1,1,1>(p,backend,schedule,n,caller,inner,scratch);break;
        }
    }
}
} // namespace strassen_cuda
