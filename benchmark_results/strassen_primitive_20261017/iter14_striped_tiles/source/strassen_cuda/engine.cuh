#pragma once
#include "prim_inter.cuh"
#include "large_control.cuh"
#include "huge_pair.cuh"
#include "simt.cuh"
#include "sliced.cuh"
#include <stdexcept>
#include <string>

namespace strassen_cuda {
enum class Backend { Legacy, Large, Medium, Huge, Sync64, Pipe64,
    Pipe64K16, Pipe64R4, Pipe64x32, Pipe128x64, Pipe128,
    Swizzle64, Swizzle64x32, Swizzle32x64, Coalesced64, SwizzleCoalesced64,
    BalancedK8R8, BalancedK16R8, BalancedK8R4, BalancedK32R4, Exact64, ExactSwizzle64,
    Deferred64, DeferredSwizzle64, Sliced1, Sliced2, Sliced4, Striped1, Striped2, Striped4,
    CgExact64, CgDeferred64, ReadonlyDeferred64, StripedTile64, StripedTile128x64, StripedTile64x128 };
struct NamedBackend {const char* name;Backend backend;bool normal;bool paired=false;};
inline const NamedBackend backends[]={
    {"legacy",Backend::Legacy,false},{"large",Backend::Large,true},
    {"medium",Backend::Medium,false},{"huge",Backend::Huge,false,true},
    {"sync64_k8_r8",Backend::Sync64,true},{"pipe64_k8_r8",Backend::Pipe64,true},
    {"pipe64_k16_r8",Backend::Pipe64K16,true},{"pipe64_k16_r4",Backend::Pipe64R4,true},
    {"pipe64x32_k16_r4",Backend::Pipe64x32,true,true},
    {"pipe128x64_k16_r8x4",Backend::Pipe128x64,true},{"pipe128_k16_r8",Backend::Pipe128,true},
    {"swizzle64_k16_r4",Backend::Swizzle64,true},
    {"swizzle64x32_k16_r4",Backend::Swizzle64x32,true},
    {"swizzle32x64_k16_r4",Backend::Swizzle32x64,true},
    {"coalesced64_k16_r4",Backend::Coalesced64,true},
    {"swizzle_coalesced64_k16_r4",Backend::SwizzleCoalesced64,true},
    {"swizzle64_k8_r8x4",Backend::BalancedK8R8,true},
    {"swizzle64_k16_r8x4",Backend::BalancedK16R8,true},
    {"swizzle_coalesced64_k8_r4",Backend::BalancedK8R4,true},
    {"swizzle_coalesced64_k32_r4",Backend::BalancedK32R4,true},
    {"exact_coalesced64_k16_r4",Backend::Exact64,true},
    {"exact_swizzle_coalesced64_k16_r4",Backend::ExactSwizzle64,true},
    {"deferred_exact_coalesced64_k16_r4",Backend::Deferred64,true},
    {"deferred_exact_swizzle_coalesced64_k16_r4",Backend::DeferredSwizzle64,true,true},
    {"sliced1_64x32_k8_r8x4",Backend::Sliced1,true},
    {"sliced2_64x32_k8_r8x4",Backend::Sliced2,true},
    {"sliced4_64x32_k8_r8x4",Backend::Sliced4,true},
    {"striped_sliced1_64x32_k8_r8x4",Backend::Striped1,true},
    {"striped_sliced2_64x32_k8_r8x4",Backend::Striped2,true},
    {"striped_sliced4_64x32_k8_r8x4",Backend::Striped4,true},
    {"cg_exact_swizzle_coalesced64_k16_r4",Backend::CgExact64,true},
    {"cg_deferred_exact_swizzle_coalesced64_k16_r4",Backend::CgDeferred64,true},
    {"ldg_deferred_exact_swizzle_coalesced64_k16_r4",Backend::ReadonlyDeferred64,true},
    {"striped_tile64_k16_r8x4",Backend::StripedTile64,true},
    {"striped_tile128x64_k16_r8x4",Backend::StripedTile128x64,true},
    {"striped_tile64x128_k16_r8x4",Backend::StripedTile64x128,true}};

inline bool valid_shape(Backend backend,int s) {
    if(s<=0||s%4)return false;
    switch(backend) {
    case Backend::Legacy:return s%((s<=384)?16:(s<=1024&&s!=1024)?32:(s<=4096)?64:128)==0;
    case Backend::Large:return s%64==0;
    case Backend::Medium:return s%32==0;
    case Backend::Huge:return s%128==0;
    default:return true;
    }
}

template<int D,int E,int GD,int GE,bool Atomic>
void primitive(Backend backend,const float* x,const float* y,const float* v,const float* w,
               float* d,float* e,int s,int ld,cudaStream_t stream) {
    switch(backend) {
    case Backend::Legacy:
        if constexpr(Atomic) prim_inter_templated<D,E,GD,GE>(x,y,v,w,d,e,s,s,s,ld,ld,ld,ld,ld,ld,stream);
        else throw std::runtime_error("legacy backend requires atomic updates");
        break;
    case Backend::Large:
        primitive_large_control<D,E,GD,GE,Atomic><<<dim3((s+63)/64,(s+63)/64),64,0,stream>>>
            (x,y,v,w,d,e,s,s,s,ld,ld,ld,ld,ld,ld);break;
    case Backend::Medium:
        if constexpr(Atomic) primitive_medium<D,E,GD,GE><<<dim3((s+31)/32,(s+31)/32),64,0,stream>>>
            (x,y,v,w,d,e,s,s,s,ld,ld,ld,ld,ld,ld);
        else throw std::runtime_error("medium backend requires atomic updates");
        break;
    case Backend::Huge:
        if constexpr(Atomic) primitive_huge<D,E,GD,GE><<<dim3((s+127)/128,(s+127)/128),256,0,stream>>>
            (x,y,v,w,d,e,s,s,s,ld,ld,ld,ld,ld,ld);
        else throw std::runtime_error("huge backend requires atomic updates");
        break;
    case Backend::Sync64: launch<64,64,8,8,8,false,Atomic,D,E,GD,GE>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Pipe64: launch<64,64,8,8,8,true,Atomic,D,E,GD,GE>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Pipe64K16: launch<64,64,16,8,8,true,Atomic,D,E,GD,GE>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Pipe64R4: launch<64,64,16,4,4,true,Atomic,D,E,GD,GE>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Pipe64x32: launch<64,32,16,4,4,true,Atomic,D,E,GD,GE>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Pipe128x64: launch<128,64,16,8,4,true,Atomic,D,E,GD,GE>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Pipe128: launch<128,128,16,8,8,true,Atomic,D,E,GD,GE>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Swizzle64: launch_swizzled<64,64,16,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Swizzle64x32: launch_swizzled<64,32,16,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Swizzle32x64: launch_swizzled<32,64,16,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Coalesced64: launch_coalesced<false,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::SwizzleCoalesced64: launch_coalesced<true,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::BalancedK8R8: launch_balanced<8,8,false,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::BalancedK16R8: launch_balanced<16,8,false,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::BalancedK8R4: launch_balanced<8,4,true,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::BalancedK32R4: launch_balanced<32,4,true,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Exact64: launch_exact<false,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::ExactSwizzle64: launch_exact<true,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Deferred64: launch_exact<false,D,E,GD,GE,Atomic,true>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::DeferredSwizzle64: launch_exact<true,D,E,GD,GE,Atomic,true>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Sliced1: launch_sliced<1,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Sliced2: launch_sliced<2,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Sliced4: launch_sliced<4,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Striped1: launch_sliced<1,D,E,GD,GE,Atomic,true>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Striped2: launch_sliced<2,D,E,GD,GE,Atomic,true>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::Striped4: launch_sliced<4,D,E,GD,GE,Atomic,true>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::CgExact64: launch_exact<true,D,E,GD,GE,Atomic,false,1>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::CgDeferred64: launch_exact<true,D,E,GD,GE,Atomic,true,1>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::ReadonlyDeferred64: launch_exact<true,D,E,GD,GE,Atomic,true,2>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::StripedTile64: launch_striped_tile<64,64,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::StripedTile128x64: launch_striped_tile<128,64,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    case Backend::StripedTile64x128: launch_striped_tile<64,128,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);break;
    }
}

struct Operand {const float *x,*y,*v,*w;float *d,*e;};
inline Operand operands(int i,const float* a,const float* b,float* c,int n) {
    int s=n/2;size_t col=size_t(s)*n,diag=col+s;
    switch(i) {
    case 0:return{a+s,a+diag,b,b,c+s,c+diag};
    case 1:return{a,a+col,b+diag,b+diag,c+col,c};
    case 2:return{a+s,a,b,b+col,c+diag,c+diag};
    case 3:return{a+col,a+diag,b+s,b+diag,c,c};
    case 4:return{a,a,b+col,b+diag,c+col,c+diag};
    case 5:return{a+diag,a+diag,b+s,b,c,c+s};
    default:return{a,a+diag,b,b+diag,c,c+diag};
    }
}
template<bool Atomic>
void product(int i,Backend backend,const float* a,const float* b,float* c,int n,cudaStream_t stream) {
    auto p=operands(i,a,b,c,n);int s=n/2;
    switch(i) {
    case 0:case 1:primitive<1,0,1,-1,Atomic>(backend,p.x,p.y,p.v,p.w,p.d,p.e,s,n,stream);break;
    case 2:primitive<-1,1,0,1,Atomic>(backend,p.x,p.y,p.v,p.w,p.d,p.e,s,n,stream);break;
    case 3:primitive<-1,1,1,0,Atomic>(backend,p.x,p.y,p.v,p.w,p.d,p.e,s,n,stream);break;
    case 4:case 5:primitive<0,-1,1,1,Atomic>(backend,p.x,p.y,p.v,p.w,p.d,p.e,s,n,stream);break;
    default:primitive<1,1,1,1,Atomic>(backend,p.x,p.y,p.v,p.w,p.d,p.e,s,n,stream);break;
    }
}
inline void checked(cudaError_t status) {if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
struct Workspace {
    cudaStream_t streams[7];cudaEvent_t start,stage,done[7];
    Workspace() {
        checked(cudaEventCreateWithFlags(&start,cudaEventDisableTiming));
        checked(cudaEventCreateWithFlags(&stage,cudaEventDisableTiming));
        for(int i=0;i<7;++i){checked(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));checked(cudaEventCreateWithFlags(&done[i],cudaEventDisableTiming));}
    }
    ~Workspace(){for(int i=0;i<7;++i){cudaStreamDestroy(streams[i]);cudaEventDestroy(done[i]);}cudaEventDestroy(start);cudaEventDestroy(stage);}
    Workspace(const Workspace&)=delete;Workspace& operator=(const Workspace&)=delete;
};
enum class Schedule {SerialAtomic,SerialNormal,SevenAtomic,DependencyNormal,PairedAtomic,PairedNormal,FourBatchedAtomic};

template<int D,int E,int GD,int GE,bool Atomic>
void pair(Backend backend,Operand a,Operand b,int s,int ld,cudaStream_t stream) {
    PrimBatchedPairArgs args{a.x,a.y,a.v,a.w,a.d,a.e,b.x,b.y,b.v,b.w,b.d,b.e};
    if(backend==Backend::Huge) {
        if constexpr(Atomic) primitive_huge_pair<D,E,GD,GE><<<dim3(s/128,s/128,2),256,0,stream>>>
            (args,s,s,s,ld,ld,ld,ld,ld,ld);
        else throw std::runtime_error("huge pairs require atomic updates");
    } else if(backend==Backend::Pipe64x32) {
        simt_primitive<64,32,16,4,4,true,Atomic,D,E,GD,GE,true>
            <<<dim3((s+63)/64,(s+31)/32,2),128,0,stream>>>
            (a.x,a.y,a.v,a.w,a.d,a.e,s,s,s,ld,args);
    } else if(backend==Backend::DeferredSwizzle64) {
        dim3 grid((s+63)/64,(s+63)/64,2);
        if(s%64==0) {
            simt_primitive<64,64,16,4,4,true,Atomic,D,E,GD,GE,true,true,true,true,true>
                <<<grid,256,0,stream>>>(a.x,a.y,a.v,a.w,a.d,a.e,s,s,s,ld,args);
        } else {
            simt_primitive<64,64,16,4,4,true,Atomic,D,E,GD,GE,true,true,true,false,true>
                <<<grid,256,0,stream>>>(a.x,a.y,a.v,a.w,a.d,a.e,s,s,s,ld,args);
        }
    } else throw std::runtime_error("backend has no paired implementation");
}
template<bool Atomic>
void paired_level(Backend backend,const float* a,const float* b,float* c,int n,cudaStream_t stream) {
    int s=n/2;
    pair<1,0,1,-1,Atomic>(backend,operands(0,a,b,c,n),operands(1,a,b,c,n),s,n,stream);
    auto p2=operands(2,a,b,c,n);p2.d=p2.e;
    pair<-1,1,1,0,Atomic>(backend,p2,operands(3,a,b,c,n),s,n,stream);
    pair<0,-1,1,1,Atomic>(backend,operands(4,a,b,c,n),operands(5,a,b,c,n),s,n,stream);
    product<Atomic>(6,backend,a,b,c,n,stream);
}

// Paper section 3.3: four variants, each submitted on its own stream. Pairs
// share compile-time coefficients, with grid.z selecting the product pointers.
// M3's sole positive output is remapped to D to share the M4 specialization.
inline void batched_variant(int variant,Backend backend,const float* a,const float* b,
                            float* c,int n,cudaStream_t stream) {
    const int s=n/2;
    switch(variant) {
    case 0:pair<1,0,1,-1,true>(backend,operands(0,a,b,c,n),operands(1,a,b,c,n),s,n,stream);break;
    case 1:{auto p=operands(2,a,b,c,n);p.d=p.e;
        pair<-1,1,1,0,true>(backend,p,operands(3,a,b,c,n),s,n,stream);break;}
    case 2:pair<0,-1,1,1,true>(backend,operands(4,a,b,c,n),operands(5,a,b,c,n),s,n,stream);break;
    default:product<true>(6,backend,a,b,c,n,stream);break;
    }
}

// All entry/exit dependencies are enqueued on the caller stream. No C reset:
// this is C <- A B + C, including a nonzero initial C and repeated invocations.
inline void one_level(Backend backend,Schedule schedule,const float* a,const float* b,float* c,
                      int n,cudaStream_t caller,Workspace& workspace) {
    if(n<=0||n%8)throw std::runtime_error("one_level currently requires N divisible by 8");
    if(!valid_shape(backend,n/2))throw std::runtime_error("shape violates original kernel tile constraints");
    bool normal=schedule==Schedule::SerialNormal||schedule==Schedule::DependencyNormal||schedule==Schedule::PairedNormal;
    if(normal&&(backend==Backend::Legacy||backend==Backend::Medium||backend==Backend::Huge))
        throw std::runtime_error("selected backend requires atomic updates");
    if(schedule==Schedule::PairedAtomic||schedule==Schedule::PairedNormal) {
        if(schedule==Schedule::PairedAtomic)paired_level<true>(backend,a,b,c,n,caller);
        else paired_level<false>(backend,a,b,c,n,caller);
        return;
    }
    if(schedule==Schedule::SerialAtomic||schedule==Schedule::SerialNormal) {
        for(int i=0;i<7;++i) {
            if(schedule==Schedule::SerialAtomic)product<true>(i,backend,a,b,c,n,caller);
            else product<false>(i,backend,a,b,c,n,caller);
        }
        return;
    }
    checked(cudaEventRecord(workspace.start,caller));
    if(schedule==Schedule::FourBatchedAtomic) {
        if(backend!=Backend::Huge&&backend!=Backend::Pipe64x32&&backend!=Backend::DeferredSwizzle64)
            throw std::runtime_error("four-variant batching requires a paired backend");
        for(int i=0;i<4;++i) {
            checked(cudaStreamWaitEvent(workspace.streams[i],workspace.start));
            batched_variant(i,backend,a,b,c,n,workspace.streams[i]);
            checked(cudaEventRecord(workspace.done[i],workspace.streams[i]));
            checked(cudaStreamWaitEvent(caller,workspace.done[i]));
        }
    } else if(schedule==Schedule::SevenAtomic) {
        for(int i=0;i<7;++i){
            checked(cudaStreamWaitEvent(workspace.streams[i],workspace.start));
            product<true>(i,backend,a,b,c,n,workspace.streams[i]);
            checked(cudaEventRecord(workspace.done[i],workspace.streams[i]));
            checked(cudaStreamWaitEvent(caller,workspace.done[i]));
        }
    } else {
        for(int i=0;i<2;++i)checked(cudaStreamWaitEvent(workspace.streams[i],workspace.start));
        // Disjoint output sets between streams; within-stream overlapping writes are ordered.
        product<false>(0,backend,a,b,c,n,workspace.streams[0]);
        product<false>(2,backend,a,b,c,n,workspace.streams[0]);
        product<false>(1,backend,a,b,c,n,workspace.streams[1]);
        product<false>(3,backend,a,b,c,n,workspace.streams[1]);
        for(int i=0;i<2;++i){checked(cudaEventRecord(workspace.done[i],workspace.streams[i]));checked(cudaStreamWaitEvent(caller,workspace.done[i]));}
        checked(cudaEventRecord(workspace.stage,caller));
        for(int i=0;i<2;++i)checked(cudaStreamWaitEvent(workspace.streams[i],workspace.stage));
        product<false>(4,backend,a,b,c,n,workspace.streams[0]);
        product<false>(5,backend,a,b,c,n,workspace.streams[1]);
        for(int i=0;i<2;++i){checked(cudaEventRecord(workspace.done[i],workspace.streams[i]));checked(cudaStreamWaitEvent(caller,workspace.done[i]));}
        product<false>(6,backend,a,b,c,n,caller);
    }
}
} // namespace strassen_cuda
