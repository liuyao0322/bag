#pragma once
#include <cuda_runtime.h>

namespace strassen_cuda {

template<int Coeff,bool Exact=false>
__device__ __forceinline__ float4 load_sum(const float* x,const float* y,int row,int col,int ld,int rows,int cols) {
    float4 a=make_float4(0,0,0,0), b=a;
    if constexpr(Exact) {
        a=*reinterpret_cast<const float4*>(x+row+size_t(col)*ld);
        if constexpr(Coeff!=0) b=*reinterpret_cast<const float4*>(y+row+size_t(col)*ld);
    } else if(col<cols && row+3<rows) {
        a=*reinterpret_cast<const float4*>(x+row+size_t(col)*ld);
        if constexpr(Coeff!=0) b=*reinterpret_cast<const float4*>(y+row+size_t(col)*ld);
    } else if(col<cols) {
        float* av=&a.x; float* bv=&b.x;
        #pragma unroll
        for(int i=0;i<4;++i) if(row+i<rows) {
            av[i]=x[row+i+size_t(col)*ld];
            if constexpr(Coeff!=0) bv[i]=y[row+i+size_t(col)*ld];
        }
    }
    if constexpr(Coeff!=0) {a.x+=Coeff*b.x;a.y+=Coeff*b.y;a.z+=Coeff*b.z;a.w+=Coeff*b.w;}
    return a;
}

// Each warp has eight row groups and four column groups. Padding B breaks
// stride-related bank conflicts; A stays contiguous along its row dimension.
template<int BM,int BN,int BK,int RM,int RN,bool Pipeline,bool Atomic,
         int Delta,int Epsilon,int GD,int GE,bool Pair=false,bool SwizzleB=false,bool Coalesced=false,bool Exact=false>
__global__ __launch_bounds__((BM/RM)*(BN/RN))
void simt_primitive(const float* X,const float* Y,const float* V,const float* W,
                    float* D,float* E,int m,int n,int k,int ld,PrimBatchedPairArgs pairs) {
    if constexpr(Pair) {
        X=blockIdx.z?pairs.X1:pairs.X0;Y=blockIdx.z?pairs.Y1:pairs.Y0;
        V=blockIdx.z?pairs.V1:pairs.V0;W=blockIdx.z?pairs.W1:pairs.W0;
        D=blockIdx.z?pairs.D1:pairs.D0;E=blockIdx.z?pairs.E1:pairs.E0;
    }
    constexpr int NT=(BM/RM)*(BN/RN), ROW_WARPS=(BM/RM)/8;
    constexpr int NA=(BM*BK/4+NT-1)/NT, NB=(BN*BK/4+NT-1)/NT;
    constexpr int STAGES=Pipeline?2:1, BS=SwizzleB?BN:BN+1;
    static_assert(!SwizzleB || (BK<=32 && BN>=32 && RN==4));
    static_assert(BM%RM==0 && BN%RN==0 && BM/RM>=8 && (BM/RM)%8==0 && (BN/RN)%4==0);
    struct Tiles {float sa[STAGES][BK][BM];float sb[STAGES][BK][BS];};
    union Storage {Tiles tiles;float output[Coalesced?BM*BN:1];};
    __shared__ __align__(16) Storage storage;
    auto& sa=storage.tiles.sa;
    auto& sb=storage.tiles.sb;
    int tid=threadIdx.x, warp=tid/32,lane=tid%32;
    int rr=(warp%ROW_WARPS)*8*RM+(lane%8)*RM;
    int cc=(warp/ROW_WARPS)*4*RN+(lane/8)*RN;
    int base_r=blockIdx.x*BM, base_c=blockIdx.y*BN;
    float acc[RN][RM]={};
    float4 pa[NA],pb[NB];
    auto fetch=[&](int kk) {
        #pragma unroll
        for(int q=0;q<NA;++q) {
            int vec=tid+q*NT, r=(vec%(BM/4))*4, col=vec/(BM/4);
            if(vec<BM*BK/4) pa[q]=load_sum<Delta,Exact>(X,Y,base_r+r,kk+col,ld,m,k);
        }
        #pragma unroll
        for(int q=0;q<NB;++q) {
            int vec=tid+q*NT, row=(vec%(BK/4))*4, col=vec/(BK/4);
            if(vec<BN*BK/4) pb[q]=load_sum<Epsilon,Exact>(V,W,kk+row,base_c+col,ld,k,n);
        }
    };
    auto pack=[&](int stage) {
        #pragma unroll
        for(int q=0;q<NA;++q) {
            int vec=tid+q*NT,r=(vec%(BM/4))*4,col=vec/(BM/4);
            if(vec<BM*BK/4) *reinterpret_cast<float4*>(&sa[stage][col][r])=pa[q];
        }
        #pragma unroll
        for(int q=0;q<NB;++q) {
            int vec=tid+q*NT,row=(vec%(BK/4))*4,col=vec/(BK/4);
            if(vec<BN*BK/4) {
                // Map the K-vector groups of one loading warp to disjoint
                // shared-memory banks, preserving float4 alignment in N.
                int dest=SwizzleB?(col^((row/4)*(128/BK))):col;
                sb[stage][row+0][dest]=pb[q].x;sb[stage][row+1][dest]=pb[q].y;
                sb[stage][row+2][dest]=pb[q].z;sb[stage][row+3][dest]=pb[q].w;
            }
        }
    };
    if constexpr(Pipeline) {fetch(0);pack(0);__syncthreads();}
    for(int outer=0,stage=0;outer<k;outer+=BK,stage=Pipeline?(stage^1):0) {
        if constexpr(Pipeline) {if(outer+BK<k)fetch(outer+BK);}
        else {fetch(outer);pack(0);__syncthreads();}
        #pragma unroll
        for(int kk=0;kk<BK;++kk) {
            float ar[RM],br[RN];
            #pragma unroll
            for(int i=0;i<RM;++i) ar[i]=sa[stage][kk][rr+i];
            if constexpr(SwizzleB) {
                float4 bv=*reinterpret_cast<const float4*>(&sb[stage][kk][cc^((kk/4)*(128/BK))]);
                br[0]=bv.x;br[1]=bv.y;br[2]=bv.z;br[3]=bv.w;
            } else {
                #pragma unroll
                for(int j=0;j<RN;++j) br[j]=sb[stage][kk][cc+j];
            }
            #pragma unroll
            for(int j=0;j<RN;++j) {
                #pragma unroll
                for(int i=0;i<RM;++i) acc[j][i]=fmaf(ar[i],br[j],acc[j][i]);
            }
        }
        if constexpr(Pipeline) {if(outer+BK<k)pack(stage^1);}
        __syncthreads();
    }
    if constexpr(Coalesced) {
        static_assert(RM==4);
        // The compute loop's final barrier makes the packing buffers reusable.
        // Vector stores keep register-tile writes efficient; scalar reads then
        // assign each warp 32 adjacent rows for fully coalesced atomic updates.
        #pragma unroll
        for(int j=0;j<RN;++j)
            *reinterpret_cast<float4*>(&storage.output[(cc+j)*BM+rr])=
                make_float4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);
        __syncthreads();
        #pragma unroll
        for(int q=0;q<(BM*BN+NT-1)/NT;++q) {
            int index=tid+q*NT,row=base_r+index%BM,col=base_c+index/BM;
            if(index<BM*BN && (Exact || (row<m && col<n))) {
                float result=storage.output[index];
                size_t offset=row+size_t(col)*ld;
                if constexpr(GD!=0) {
                    if constexpr(Atomic) atomicAdd(D+offset,GD*result);
                    else D[offset]+=GD*result;
                }
                if constexpr(GE!=0) {
                    if constexpr(Atomic) atomicAdd(E+offset,GE*result);
                    else E[offset]+=GE*result;
                }
            }
        }
    } else {
        #pragma unroll
        for(int j=0;j<RN;++j) {
            #pragma unroll
            for(int i=0;i<RM;++i) {
                int row=base_r+rr+i,col=base_c+cc+j;
                if(Exact || (row<m && col<n)) {
                    size_t offset=row+size_t(col)*ld;
                    if constexpr(GD!=0) {
                        if constexpr(Atomic) atomicAdd(D+offset,GD*acc[j][i]);
                        else D[offset]+=GD*acc[j][i];
                    }
                    if constexpr(GE!=0) {
                        if constexpr(Atomic) atomicAdd(E+offset,GE*acc[j][i]);
                        else E[offset]+=GE*acc[j][i];
                    }
                }
            }
        }
    }
}

template<int BM,int BN,int BK,int RM,int RN,bool Pipeline,bool Atomic,int D,int E,int GD,int GE>
void launch(const float* x,const float* y,const float* v,const float* w,float* d,float* e,
            int s,int ld,cudaStream_t stream) {
    constexpr int NT=(BM/RM)*(BN/RN);
    simt_primitive<BM,BN,BK,RM,RN,Pipeline,Atomic,D,E,GD,GE>
        <<<dim3((s+BM-1)/BM,(s+BN-1)/BN),NT,0,stream>>>(x,y,v,w,d,e,s,s,s,ld,{});
}

template<int BM,int BN,int BK,int D,int E,int GD,int GE,bool Atomic>
void launch_swizzled(const float* x,const float* y,const float* v,const float* w,float* d,float* e,
                     int s,int ld,cudaStream_t stream) {
    constexpr int NT=(BM/4)*(BN/4);
    simt_primitive<BM,BN,BK,4,4,true,Atomic,D,E,GD,GE,false,true>
        <<<dim3((s+BM-1)/BM,(s+BN-1)/BN),NT,0,stream>>>(x,y,v,w,d,e,s,s,s,ld,{});
}

template<bool Swizzle,int D,int E,int GD,int GE,bool Atomic>
void launch_coalesced(const float* x,const float* y,const float* v,const float* w,float* d,float* e,
                      int s,int ld,cudaStream_t stream) {
    simt_primitive<64,64,16,4,4,true,Atomic,D,E,GD,GE,false,Swizzle,true>
        <<<dim3((s+63)/64,(s+63)/64),256,0,stream>>>(x,y,v,w,d,e,s,s,s,ld,{});
}

template<int BK,int RM,bool Coalesced,int D,int E,int GD,int GE,bool Atomic>
void launch_balanced(const float* x,const float* y,const float* v,const float* w,float* d,float* e,
                     int s,int ld,cudaStream_t stream) {
    constexpr int NT=(64/RM)*16;
    simt_primitive<64,64,BK,RM,4,true,Atomic,D,E,GD,GE,false,true,Coalesced>
        <<<dim3((s+63)/64,(s+63)/64),NT,0,stream>>>(x,y,v,w,d,e,s,s,s,ld,{});
}

template<bool Swizzle,int D,int E,int GD,int GE,bool Atomic>
void launch_exact(const float* x,const float* y,const float* v,const float* w,float* d,float* e,
                  int s,int ld,cudaStream_t stream) {
    if(s%64==0) {
        simt_primitive<64,64,16,4,4,true,Atomic,D,E,GD,GE,false,Swizzle,true,true>
            <<<dim3(s/64,s/64),256,0,stream>>>(x,y,v,w,d,e,s,s,s,ld,{});
    } else {
        // Partial tiles retain the complete masked primitive; never substitute
        // an ordinary GEMM for any of the seven Strassen products.
        launch_coalesced<Swizzle,D,E,GD,GE,Atomic>(x,y,v,w,d,e,s,ld,stream);
    }
}
} // namespace strassen_cuda
