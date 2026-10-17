#pragma once
// CTA-local K partitioning: no global partial products or extra launches.
namespace strassen_cuda {

template<int Slices,int Delta,int Epsilon,int GD,int GE,bool Atomic,bool Striped=false>
__global__ __launch_bounds__(64*Slices)
void sliced_primitive(const float* X,const float* Y,const float* V,const float* W,
                      float* D,float* E,int s,int ld) {
    static_assert(Slices==1 || Slices==2 || Slices==4);
    constexpr int BM=64,BN=32,BK=8,NT=64*Slices;
    struct Tiles {float a[2][Slices][BK][BM];float b[2][Slices][BK][BN];};
    union Storage {Tiles tile;float partial[Slices>1?Slices-1:1][BN][BM];};
    __shared__ __align__(16) Storage storage;
    int group=threadIdx.x/64,tid=threadIdx.x%64;
    int lane=tid%32,warp=tid/32;
    int rr=(lane%8)*(Striped?4:8),cc=warp*16+(lane/8)*4;
    int base_r=blockIdx.x*BM,base_c=blockIdx.y*BN;
    int group_k=s/Slices,k0=group*group_k;
    float acc[4][8]={};
    float4 ax[2],ay[2],bv,bw;
    auto fetch=[&](int kk) {
        #pragma unroll
        for(int q=0;q<2;++q) {
            int vec=tid+64*q,row=(vec%16)*4,col=vec/16;
            ax[q]=load_sum<0,true>(X,X,base_r+row,k0+kk+col,ld,s,s);
            if constexpr(Delta!=0) ay[q]=load_sum<0,true>(Y,Y,base_r+row,k0+kk+col,ld,s,s);
        }
        int row=(tid%2)*4,col=tid/2;
        bv=load_sum<0,true>(V,V,k0+kk+row,base_c+col,ld,s,s);
        if constexpr(Epsilon!=0) bw=load_sum<0,true>(W,W,k0+kk+row,base_c+col,ld,s,s);
    };
    auto pack=[&](int stage) {
        #pragma unroll
        for(int q=0;q<2;++q) {
            int vec=tid+64*q,row=(vec%16)*4,col=vec/16;
            float4 a=ax[q];
            if constexpr(Delta!=0) {
                a.x+=Delta*ay[q].x;a.y+=Delta*ay[q].y;
                a.z+=Delta*ay[q].z;a.w+=Delta*ay[q].w;
            }
            *reinterpret_cast<float4*>(&storage.tile.a[stage][group][col][row])=a;
        }
        int row=(tid%2)*4,col=(tid/2)^((row/4)*16);
        float4 b=bv;
        if constexpr(Epsilon!=0) {
            b.x+=Epsilon*bw.x;b.y+=Epsilon*bw.y;
            b.z+=Epsilon*bw.z;b.w+=Epsilon*bw.w;
        }
        storage.tile.b[stage][group][row+0][col]=b.x;
        storage.tile.b[stage][group][row+1][col]=b.y;
        storage.tile.b[stage][group][row+2][col]=b.z;
        storage.tile.b[stage][group][row+3][col]=b.w;
    };
    fetch(0);pack(0);__syncthreads();
    for(int outer=0,stage=0;outer<group_k;outer+=BK,stage^=1) {
        if(outer+BK<group_k)fetch(outer+BK);
        #pragma unroll
        for(int kk=0;kk<BK;++kk) {
            float4 a0=*reinterpret_cast<const float4*>(&storage.tile.a[stage][group][kk][rr]);
            float4 a1=*reinterpret_cast<const float4*>(&storage.tile.a[stage][group][kk][rr+(Striped?32:4)]);
            float ar[8]={a0.x,a0.y,a0.z,a0.w,a1.x,a1.y,a1.z,a1.w};
            float4 b=*reinterpret_cast<const float4*>(&storage.tile.b[stage][group][kk][cc^((kk/4)*16)]);
            float br[4]={b.x,b.y,b.z,b.w};
            #pragma unroll
            for(int j=0;j<4;++j) {
                #pragma unroll
                for(int i=0;i<8;++i)acc[j][i]=fmaf(ar[i],br[j],acc[j][i]);
            }
        }
        if(outer+BK<group_k)pack(stage^1);
        __syncthreads();
    }
    // Only the first S-1 groups need temporary partial outputs. The last
    // group retains its register tile, so reduction fits the packing storage.
    if(group<Slices-1) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            #pragma unroll
            for(int i=0;i<8;i+=4)
                *reinterpret_cast<float4*>(&storage.partial[group][cc+j][rr+(Striped?(i/4)*32:i)])=
                    make_float4(acc[j][i],acc[j][i+1],acc[j][i+2],acc[j][i+3]);
        }
    }
    __syncthreads();
    if(group==Slices-1) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            #pragma unroll
            for(int i=0;i<8;++i) {
                int row=rr+(Striped?(i%4+(i/4)*32):i);
                float result=acc[j][i];
                #pragma unroll
                for(int g=0;g<Slices-1;++g)result+=storage.partial[g][cc+j][row];
                // Each output coordinate has one owner; overwriting its own
                // first partial cannot race with another coordinate's reader.
                storage.partial[0][cc+j][row]=result;
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for(int q=0;q<BM*BN/NT;++q) {
        int index=threadIdx.x+q*NT,row=index%BM,col=index/BM;
        float result=storage.partial[0][col][row];
        size_t offset=base_r+row+size_t(base_c+col)*ld;
        if constexpr(GD!=0) {
            if constexpr(Atomic)atomicAdd(D+offset,GD*result);
            else D[offset]+=GD*result;
        }
        if constexpr(GE!=0) {
            if constexpr(Atomic)atomicAdd(E+offset,GE*result);
            else E[offset]+=GE*result;
        }
    }
}

template<int Slices,int D,int E,int GD,int GE,bool Atomic,bool Striped=false>
void launch_sliced(const float* x,const float* y,const float* v,const float* w,float* d,float* e,
                   int s,int ld,cudaStream_t stream) {
    if(s%64==0) {
        sliced_primitive<Slices,D,E,GD,GE,Atomic,Striped><<<dim3(s/64,s/32),64*Slices,0,stream>>>
            (x,y,v,w,d,e,s,ld);
    } else launch_exact<true,D,E,GD,GE,Atomic,true>(x,y,v,w,d,e,s,ld,stream);
}
} // namespace strassen_cuda
