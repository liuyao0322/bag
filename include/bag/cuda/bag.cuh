#pragma once
#include "strassen.cuh"
#include <algorithm>
#include <memory>

namespace bag_cuda {
enum class Basis { Minimal, Full };
enum class Schedule { Serial, SevenStream, Batched };
using Backend = strassen_cuda::Backend;
using Operand = strassen_cuda::Operand;

struct DeviceBuffer {
    float* p = nullptr;
    size_t count;
    explicit DeviceBuffer(size_t elements): count(elements) {
        if(count) strassen_cuda::checked(cudaMalloc(&p, count*sizeof(float)));
    }
    ~DeviceBuffer() { if(p) cudaFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

inline size_t local_elements(int n,int depth,Basis basis) {
    if(depth<1) throw std::runtime_error("BAG depth must be positive");
    int size=n;
    for(int level=0;level<depth;++level,size/=2)
        if(size<=0 || size%8) throw std::runtime_error("each BAG level requires N divisible by 8");
    const size_t count=size_t(n)*n;
    return basis==Basis::Full ? 3*count : count+count/2;
}

// One reusable child handles the seven ordered outer products. Workspaces
// are not reentrant and must outlive all work enqueued on the caller stream.
struct Workspace {
    const int n,depth;
    const Basis basis;
    DeviceBuffer data,child_data;
    strassen_cuda::Workspace workers;
    std::unique_ptr<Workspace> child;
    explicit Workspace(int size,int levels=1,Basis layout=Basis::Minimal):
        n(size),depth(levels),basis(layout),data(local_elements(size,levels,layout)),
        child_data(levels>1 ? 3*size_t(size/2)*(size/2) : 0) {
        if(depth>1) child=std::make_unique<Workspace>(n/2,depth-1,basis);
    }
    float* output() const { return data.p; }
    float* left() const { return data.p+size_t(n)*n; }
    float* right() const {
        return basis==Basis::Full ? left()+size_t(n)*n : left()+n/2;
    }
    size_t bytes() const {
        return (data.count+child_data.count)*sizeof(float)+(child ? child->bytes() : 0);
    }
    Workspace(const Workspace&)=delete;
    Workspace& operator=(const Workspace&)=delete;
};

template<bool Full>
__global__ void prepare(const float* a,const float* b,float* ta,float* tb,float* t,int n) {
    const int s=n/2;
    const size_t col=size_t(s)*n,diag=col+s,count=size_t(s)*s;
    for(size_t q=size_t(blockIdx.x)*blockDim.x+threadIdx.x;q<count;
        q+=size_t(blockDim.x)*gridDim.x) {
        const size_t p=q%s+(q/s)*n;
        const float a12=a[p+col],a21=a[p+s],a22=a[p+diag];
        const float b12=b[p+col],b21=b[p+s],b22=b[p+diag];
        if constexpr(Full) {
            ta[p]=a[p];ta[p+col]=a12-a21+a22;ta[p+s]=a22-a21;ta[p+diag]=a12+a22;
            tb[p]=b[p];tb[p+col]=b12-b21+b22;tb[p+s]=b22-b21;tb[p+diag]=b12+b22;
        } else {
            // SA and SB occupy adjacent row blocks of one N x (N/2) buffer,
            // preserving the primitive's common leading dimension N.
            ta[p]=a12-a21+a22;tb[p]=b12-b21+b22;
        }
        t[p]=0;t[p+col]=0;t[p+s]=0;t[p+diag]=0;
    }
}

template<bool Accumulate>
__global__ void decode(const float* t,float* c,int n) {
    const int s=n/2;
    const size_t col=size_t(s)*n,diag=col+s,count=size_t(s)*s;
    for(size_t q=size_t(blockIdx.x)*blockDim.x+threadIdx.x;q<count;
        q+=size_t(blockDim.x)*gridDim.x) {
        const size_t p=q%s+(q/s)*n;
        const float x=t[p],y=t[p+col],z=t[p+s],w=t[p+diag];
        const float v11=x,v12=y-z,v21=-y+w,v22=(-y+z)+w;
        if constexpr(Accumulate) {
            c[p]+=v11;c[p+col]+=v12;c[p+s]+=v21;c[p+diag]+=v22;
        } else {
            c[p]=v11;c[p+col]=v12;c[p+s]=v21;c[p+diag]=v22;
        }
    }
}

template<bool Full>
inline Operand operands(int i,const float* a,const float* b,const Workspace& ws) {
    const int s=ws.n/2;
    const size_t col=size_t(s)*ws.n,diag=col+s;
    float* t=ws.output();
    const float *ta=ws.left(),*tb=ws.right();
    if constexpr(Full) {
        switch(i) {
        case 0:return{ta+diag,ta+diag,tb+diag,tb+diag,t+diag,t+diag};
        case 1:return{ta+s,ta+s,tb+s,tb+s,t+s,t+s};
        case 2:return{ta+col,ta+col,tb+col,tb+col,t+col,t+col};
        case 3:return{ta,ta,tb,tb,t,t};
        case 4:return{ta+col,ta+s,tb+diag,tb+col,t,t+col};
        case 5:return{ta+col,ta,tb+col,tb+s,t+col,t+diag};
        default:return{ta+diag,ta+col,tb+col,tb,t+col,t+s};
        }
    } else {
        switch(i) {
        case 0:return{a+col,a+diag,b+col,b+diag,t+diag,t+diag};
        case 1:return{a+diag,a+s,b+diag,b+s,t+s,t+s};
        case 2:return{ta,ta,tb,tb,t+col,t+col};
        case 3:return{a,a,b,b,t,t};
        case 4:return{a+col,a+col,b+s,b+s,t,t+col};
        case 5:return{ta,a,b+col,b+col,t+col,t+diag};
        default:return{a+s,a+s,tb,b,t+col,t+s};
        }
    }
}

template<int Delta,int Epsilon>
__global__ void prepare_child(Operand p,float* a,float* b,int s,int ld) {
    const size_t count=size_t(s)*s;
    for(size_t q=size_t(blockIdx.x)*blockDim.x+threadIdx.x;q<count;
        q+=size_t(blockDim.x)*gridDim.x) {
        const size_t src=q%s+(q/s)*ld;
        float av=p.x[src],bv=p.v[src];
        if constexpr(Delta!=0) av+=Delta*p.y[src];
        if constexpr(Epsilon!=0) bv+=Epsilon*p.w[src];
        a[q]=av;b[q]=bv;
    }
}
template<int GD,int GE>
__global__ void merge_child(const float* m,Operand p,int s,int ld) {
    const size_t count=size_t(s)*s;
    for(size_t q=size_t(blockIdx.x)*blockDim.x+threadIdx.x;q<count;
        q+=size_t(blockDim.x)*gridDim.x) {
        const size_t dst=q%s+(q/s)*ld;
        if constexpr(GD!=0) p.d[dst]+=GD*m[q];
        if constexpr(GE!=0) p.e[dst]+=GE*m[q];
    }
}
inline unsigned blocks(int n) {
    return unsigned(std::min<size_t>((size_t(n/2)*(n/2)+255)/256,4096));
}
template<bool Accumulate>
void run(const float*,const float*,float*,Backend,Schedule,cudaStream_t,Workspace&);

template<int D,int E,int GD,int GE>
inline void submit(Operand p,Backend backend,Schedule schedule,cudaStream_t stream,Workspace& ws) {
    const int s=ws.n/2;
    if(ws.depth==1) {
        strassen_cuda::primitive<D,E,GD,GE,true>(backend,p.x,p.y,p.v,p.w,p.d,p.e,s,ws.n,stream);
    } else {
        float *a=ws.child_data.p,*b=a+size_t(s)*s,*m=b+size_t(s)*s;
        prepare_child<D,E><<<blocks(ws.n),256,0,stream>>>(p,a,b,s,ws.n);
        run<false>(a,b,m,backend,schedule,stream,*ws.child);
        merge_child<GD,GE><<<blocks(ws.n),256,0,stream>>>(m,p,s,ws.n);
    }
}

template<bool Full>
inline void product(int i,const float* a,const float* b,Backend backend,Schedule schedule,
                    cudaStream_t stream,Workspace& ws) {
    const auto p=operands<Full>(i,a,b,ws);
    switch(i) {
    case 0:
        if constexpr(Full) submit<0,0,1,0>(p,backend,schedule,stream,ws);
        else submit<1,1,1,0>(p,backend,schedule,stream,ws);
        break;
    case 1:
        if constexpr(Full) submit<0,0,1,0>(p,backend,schedule,stream,ws);
        else submit<-1,-1,1,0>(p,backend,schedule,stream,ws);
        break;
    case 2:case 3:submit<0,0,1,0>(p,backend,schedule,stream,ws);break;
    case 4:
        if constexpr(Full) submit<-1,-1,1,1>(p,backend,schedule,stream,ws);
        else submit<0,0,1,1>(p,backend,schedule,stream,ws);
        break;
    case 5:
        if constexpr(Full) submit<-1,-1,-1,-1>(p,backend,schedule,stream,ws);
        else submit<-1,0,-1,-1>(p,backend,schedule,stream,ws);
        break;
    default:
        if constexpr(Full) submit<-1,-1,1,1>(p,backend,schedule,stream,ws);
        else submit<0,-1,1,1>(p,backend,schedule,stream,ws);
        break;
    }
}

template<bool Full,bool Accumulate>
inline void level(const float* a,const float* b,float* c,Backend backend,Schedule schedule,
                  cudaStream_t caller,Workspace& ws) {
    prepare<Full><<<blocks(ws.n),256,0,caller>>>(a,b,ws.left(),ws.right(),ws.output(),ws.n);
    if(ws.depth>1 || schedule==Schedule::Serial) {
        for(int i=0;i<7;++i) product<Full>(i,a,b,backend,schedule,caller,ws);
    } else {
        auto& w=ws.workers;
        strassen_cuda::checked(cudaEventRecord(w.start,caller));
        auto launch=[&](int worker,auto body) {
            strassen_cuda::checked(cudaStreamWaitEvent(w.streams[worker],w.start));
            body(w.streams[worker]);
            strassen_cuda::checked(cudaEventRecord(w.done[worker],w.streams[worker]));
            strassen_cuda::checked(cudaStreamWaitEvent(caller,w.done[worker]));
        };
        if(schedule==Schedule::SevenStream) {
            for(int i=0;i<7;++i) launch(i,[&](cudaStream_t stream) {
                product<Full>(i,a,b,backend,schedule,stream,ws);
            });
        } else if constexpr(Full) {
            launch(0,[&](cudaStream_t stream) {
                strassen_cuda::pair<0,0,1,0,true>(backend,operands<true>(0,a,b,ws),
                    operands<true>(1,a,b,ws),ws.n/2,ws.n,stream);
            });
            launch(1,[&](cudaStream_t stream) {
                strassen_cuda::pair<0,0,1,0,true>(backend,operands<true>(2,a,b,ws),
                    operands<true>(3,a,b,ws),ws.n/2,ws.n,stream);
            });
            launch(2,[&](cudaStream_t stream) {
                strassen_cuda::pair<-1,-1,1,1,true>(backend,operands<true>(4,a,b,ws),
                    operands<true>(6,a,b,ws),ws.n/2,ws.n,stream);
            });
            launch(3,[&](cudaStream_t stream) {product<true>(5,a,b,backend,schedule,stream,ws);});
        } else {
            launch(0,[&](cudaStream_t stream) {
                strassen_cuda::pair<0,0,1,0,true>(backend,operands<false>(2,a,b,ws),
                    operands<false>(3,a,b,ws),ws.n/2,ws.n,stream);
            });
            constexpr int singles[]={0,1,4,5,6};
            for(int j=0;j<5;++j) launch(j+1,[&](cudaStream_t stream) {
                product<false>(singles[j],a,b,backend,schedule,stream,ws);
            });
        }
    }
    decode<Accumulate><<<blocks(ws.n),256,0,caller>>>(ws.output(),c,ws.n);
}
template<bool Accumulate>
void run(const float* a,const float* b,float* c,Backend backend,Schedule schedule,
         cudaStream_t caller,Workspace& ws) {
    if(ws.basis==Basis::Full) level<true,Accumulate>(a,b,c,backend,schedule,caller,ws);
    else level<false,Accumulate>(a,b,c,backend,schedule,caller,ws);
}

// Enqueue C <- AB+C. A/B/C are distinct column-major N x N FP32 matrices.
// No allocation, host/device synchronization or cuBLAS call occurs here.
inline void matmul(const float* a,const float* b,float* c,int n,cudaStream_t caller,
                   Workspace& ws,Backend backend=Backend::ColumnTile8,
                   Schedule schedule=Schedule::Batched) {
    if(n!=ws.n) throw std::runtime_error("BAG workspace size mismatch");
    int leaf=n;
    for(int level=0;level<ws.depth;++level) leaf/=2;
    if(!strassen_cuda::valid_shape(backend,leaf)) throw std::runtime_error("BAG leaf violates backend tile constraints");
    if(schedule==Schedule::Batched && backend!=Backend::ColumnTile8 && backend!=Backend::Huge &&
       backend!=Backend::Pipe64x32 && backend!=Backend::DeferredSwizzle64)
        throw std::runtime_error("BAG batched schedule requires a paired backend");
    run<true>(a,b,c,backend,schedule,caller,ws);
}
} // namespace bag_cuda
