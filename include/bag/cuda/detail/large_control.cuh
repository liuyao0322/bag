#pragma once
// Verbatim primitive_large compute body; only the output-update policy differs.
// Include prim_inter.cuh first for ACCUM_COL and the original atomic writer.
template<int Gamma, bool Atomic>
__device__ __forceinline__ void control_write(float* out,const float4* acc,int ld) {
    if constexpr (Atomic) { write_vec_large_vec<Gamma>(out,acc,ld); }
    else if constexpr (Gamma != 0) {
        #pragma unroll
        for(int col=0;col<8;++col) {
            float4 a=acc[2*col],b=acc[2*col+1];
            float4 old0=*reinterpret_cast<float4*>(out+col*ld);
            float4 old1=*reinterpret_cast<float4*>(out+col*ld+4);
            old0.x+=Gamma*a.x;old0.y+=Gamma*a.y;old0.z+=Gamma*a.z;old0.w+=Gamma*a.w;
            old1.x+=Gamma*b.x;old1.y+=Gamma*b.y;old1.z+=Gamma*b.z;old1.w+=Gamma*b.w;
            *reinterpret_cast<float4*>(out+col*ld)=old0;
            *reinterpret_cast<float4*>(out+col*ld+4)=old1;
        }
    }
}
template <int delta, int epsilon, int gamma0, int gamma1, bool Atomic>
__global__ __launch_bounds__(64)
void primitive_large_control(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const float* __restrict__ V,
    const float* __restrict__ W,
    float* __restrict__ D,
    float* __restrict__ E,
    int M, int N, int K,
    int ldx, int ldy, int ldv, int ldw, int ldd, int lde
) {
    const int ms = 64, ns = 64, ks = 8;
    const int mw = 32, nw = 64, mr = 8, nr = 8;

    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;

    __shared__ float sAB[ks * ms + ks * ns];
    float* buffer_A = sAB;
    float* buffer_B = sAB + ks * ms;

    X += bx * ms;
    Y += bx * ms;
    V += by * ns * ldv;
    W += by * ns * ldw;

    const int floats_per_col_A = ms;
    const int float4x2_per_col_A = floats_per_col_A / 8;
    const int load_row_A = (tx % float4x2_per_col_A) * 8;
    const int load_col_A = tx / float4x2_per_col_A;
    const int load_col_B = tx;

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

    float4 acc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    for (int k_outer = 0; k_outer < K; k_outer += ks) {
        const float* tile_X = X + k_outer * ldx;
        const float* tile_Y = Y + k_outer * ldy;
        const float* tile_V = V + k_outer;
        const float* tile_W = W + k_outer;

        float4 x_vec[2];
        float4 y_vec[2];
        x_vec[0] = *reinterpret_cast<const float4*>(tile_X + load_row_A + load_col_A * ldx);
        x_vec[1] = *reinterpret_cast<const float4*>(tile_X + load_row_A + load_col_A * ldx + 4);
        y_vec[0] = *reinterpret_cast<const float4*>(tile_Y + load_row_A + load_col_A * ldy);
        y_vec[1] = *reinterpret_cast<const float4*>(tile_Y + load_row_A + load_col_A * ldy + 4);
        float4* store_A = reinterpret_cast<float4*>(buffer_A + load_col_A * ms + load_row_A);
        read_vec_large<delta>(store_A, x_vec, y_vec);

        float4 v_vec[2];
        float4 w_vec[2];
        v_vec[0] = *reinterpret_cast<const float4*>(tile_V + load_col_B * ldv);
        v_vec[1] = *reinterpret_cast<const float4*>(tile_V + load_col_B * ldv + 4);
        w_vec[0] = *reinterpret_cast<const float4*>(tile_W + load_col_B * ldw);
        w_vec[1] = *reinterpret_cast<const float4*>(tile_W + load_col_B * ldw + 4);
        store_vec_B_strided_large<epsilon>(buffer_B, v_vec, w_vec, 0, load_col_B, ns);

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < ks; ++kk) {
            const float4 a_lo = *reinterpret_cast<const float4*>(buffer_A + kk * ms + warp_offset_A + thread_offset_A);
            const float4 a_hi = *reinterpret_cast<const float4*>(buffer_A + kk * ms + warp_offset_A + thread_offset_A + 4);
            const int base_n = kk * ns + warp_offset_B + thread_offset_B;
            const float4 b_lo = make_float4(buffer_B[base_n + 0], buffer_B[base_n + 1], buffer_B[base_n + 2], buffer_B[base_n + 3]);
            const float4 b_hi = make_float4(buffer_B[base_n + 4], buffer_B[base_n + 5], buffer_B[base_n + 6], buffer_B[base_n + 7]);

            ACCUM_COL(acc[0],  acc[1],  a_lo, a_hi, b_lo.x);
            ACCUM_COL(acc[2],  acc[3],  a_lo, a_hi, b_lo.y);
            ACCUM_COL(acc[4],  acc[5],  a_lo, a_hi, b_lo.z);
            ACCUM_COL(acc[6],  acc[7],  a_lo, a_hi, b_lo.w);
            ACCUM_COL(acc[8],  acc[9],  a_lo, a_hi, b_hi.x);
            ACCUM_COL(acc[10], acc[11], a_lo, a_hi, b_hi.y);
            ACCUM_COL(acc[12], acc[13], a_lo, a_hi, b_hi.z);
            ACCUM_COL(acc[14], acc[15], a_lo, a_hi, b_hi.w);
        }

        __syncthreads();
    }

    int out_m = bx * ms + warp_offset_A + thread_offset_A;
    int out_n = by * ns + warp_offset_B + thread_offset_B;

    D += out_m + out_n * ldd;
    control_write<gamma0,Atomic>(D, acc, ldd);

    E += out_m + out_n * lde;
    control_write<gamma1,Atomic>(E, acc, lde);
}

