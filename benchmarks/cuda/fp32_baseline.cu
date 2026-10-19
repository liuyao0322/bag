// Standalone cost audit. Existing primitive headers are included without modification.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <bag/cuda/detail/legacy/prim_inter.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t cuda_check_status_ = (x); if (cuda_check_status_ != cudaSuccess) \
    throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(cuda_check_status_)); } while (0)
#define BLAS_CHECK(x) do { cublasStatus_t blas_check_status_ = (x); if (blas_check_status_ != CUBLAS_STATUS_SUCCESS) \
    throw std::runtime_error(std::string(#x) + ": status=" + std::to_string(int(blas_check_status_))); } while (0)

// Identical, exactly representable FP32 input generation on host and device.
__host__ __device__ uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    x *= 0x846ca68bu; return x ^ (x >> 16);
}
__host__ __device__ float value(size_t i, uint32_t seed) {
    return (int(mix(uint32_t(i) ^ seed) & 65535u) - 32768) * (1.0f / 65536.0f);
}
__global__ void initialize(float* p, size_t count, uint32_t seed) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) p[i] = value(i, seed);
}
__global__ void gather(const float* p, const size_t* indices, float* out, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = p[indices[i]];
}

struct Buffer {
    float* p = nullptr;
    explicit Buffer(size_t count) { CUDA_CHECK(cudaMalloc(&p, count * sizeof(float))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Events {
    cudaEvent_t start, stop;
    Events() { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
    ~Events() { cudaEventDestroy(start); cudaEventDestroy(stop); }
};
struct Options {
    std::vector<int> sizes{512,768,1024,1280,1536,1792,2048,3072,4096,6144,
                           7680,8192,10240,12288,16384,20480};
    int rounds = 5, samples = 128;
    double target_ms = 200;
    bool validate_only = false;
    std::string output = "results", only;
};
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() { if (++i == argc) throw std::runtime_error("missing value for " + arg); return std::string(argv[i]); };
        if (arg == "--sizes") {
            o.sizes.clear(); std::istringstream s(next()); std::string item;
            while (std::getline(s, item, ',')) o.sizes.push_back(std::stoi(item));
        } else if (arg == "--rounds") o.rounds = std::stoi(next());
        else if (arg == "--target-ms") o.target_ms = std::stod(next());
        else if (arg == "--samples") o.samples = std::stoi(next());
        else if (arg == "--output") o.output = next();
        else if (arg == "--only") o.only = next();
        else if (arg == "--validate-only") o.validate_only = true;
        else throw std::runtime_error("unknown argument " + arg);
    }
    if (o.sizes.empty() || o.rounds < 1 || o.samples < 16 || o.target_ms <= 0)
        throw std::runtime_error("invalid benchmark configuration");
    for (int n : o.sizes) {
        // Existing square kernels do not mask their tiles. Reject unsupported sizes.
        int s = n / 2;
        int tile = s <= 384 ? 16 : (s == 1024 ? 64 : (s <= 1024 ? 32 : (s <= 4096 ? 64 : 128)));
        if (n < 32 || n % 2 || s % tile || s % 8)
            throw std::runtime_error("size incompatible with existing unmasked primitives: " + std::to_string(n));
    }
    return o;
}

struct Case {
    std::string name, kernel;
    int s, ld, delta = 0, epsilon = 0, gamma_d = 1, gamma_e = 0;
    bool primitive = false;
    float *a, *b, *c;
    std::function<void()> launch;
};
std::string selected_kernel(int s) {
    if (s <= 384) return "primitive_small";
    if (s == 1024) return "primitive_large";
    if (s <= 1024) return "primitive_medium";
    if (s <= 4096) return "primitive_large";
    return "primitive_huge";
}
float elapsed(Events& e, cudaStream_t stream, const Case& c, int repeats) {
    CUDA_CHECK(cudaEventRecord(e.start, stream));
    for (int i = 0; i < repeats; ++i) c.launch();
    CUDA_CHECK(cudaEventRecord(e.stop, stream));
    CUDA_CHECK(cudaEventSynchronize(e.stop));
    CUDA_CHECK(cudaGetLastError());
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e.start, e.stop)); return ms;
}
std::vector<float> read_samples(const float* p, const std::vector<size_t>& indices, cudaStream_t stream) {
    size_t* device_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&device_indices, indices.size() * sizeof(size_t)));
    Buffer device_values(indices.size()); std::vector<float> result(indices.size());
    CUDA_CHECK(cudaMemcpyAsync(device_indices, indices.data(), indices.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    gather<<<(indices.size()+255)/256,256,0,stream>>>(p, device_indices, device_values.p, int(indices.size()));
    CUDA_CHECK(cudaMemcpyAsync(result.data(), device_values.p, result.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); CUDA_CHECK(cudaFree(device_indices)); return result;
}
struct Error { double relative_l2 = 0, max_abs = 0, max_scaled = 0; int count = 0; bool pass = true; };
Error validate(const Case& c, int parent, int requested_samples, int repeats, cudaStream_t stream,
               uint32_t seed_a=11,uint32_t seed_b=23,uint32_t seed_c=37) {
    int s = c.s, ld = c.ld;
    // Exhaustive for small matrices, deterministic samples (including corners) otherwise.
    std::vector<size_t> positions;
    if (s <= 128) {
        for (int col = 0; col < s; ++col) for (int row = 0; row < s; ++row)
            positions.push_back(row + size_t(col)*ld);
    } else {
        positions = {0, size_t(s-1), size_t(s-1)*ld, size_t(s-1)*ld+s-1};
        std::mt19937 rng(20261017 + s);
        while (positions.size() < size_t(requested_samples)) {
            int row = rng() % s, col = rng() % s;
            positions.push_back(row + size_t(col)*ld);
        }
        std::sort(positions.begin(), positions.end());
        positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
    }
    size_t offset = c.primitive ? size_t(s) * (ld + 1) : 0;
    std::vector<size_t> indices = positions;
    if (c.primitive) for (auto p : positions) indices.push_back(p + offset);
    // Sentinel samples in untouched quadrants catch stray output writes.
    if (ld == parent && s < parent) {
        for (int i = 0; i < 16; ++i) {
            int r = i * (s-1) / 15;
            indices.push_back(r + size_t(s+r)*ld);
            indices.push_back(s+r + size_t(r)*ld);
        }
        if (!c.primitive) for (int i = 0; i < 16; ++i) {
            int r = i * (s-1) / 15; indices.push_back(s+r+size_t(s+r)*ld);
        }
    }
    auto got = read_samples(c.c, indices, stream);
    std::vector<double> refs, dots; refs.reserve(indices.size());
    for (size_t q = 0; q < positions.size(); ++q) {
        int row = positions[q] % ld, col = positions[q] / ld;
        double dot = 0;
        for (int k = 0; k < s; ++k) {
            float x = value(row + size_t(k)*ld, seed_a);
            float v = value(k + size_t(col)*ld, seed_b);
            if (c.primitive) {
                x += c.delta * value(row + size_t(k)*ld + offset, seed_a);
                v += c.epsilon * value(k + size_t(col)*ld + offset, seed_b);
            }
            dot += double(x) * double(v);
        }
        dots.push_back(dot);
        refs.push_back(double(value(positions[q],seed_c)) + repeats*c.gamma_d*dot);
    }
    if (c.primitive) {
        for (size_t q = 0; q < positions.size(); ++q) {
            refs.push_back(double(value(positions[q]+offset,seed_c)) + repeats*c.gamma_e*dots[q]);
        }
    }
    while (refs.size() < indices.size()) refs.push_back(value(indices[refs.size()],seed_c));
    Error e; double err2 = 0, ref2 = 0, max_ref = 0;
    for (size_t q = 0; q < got.size(); ++q) {
        double d = std::abs(double(got[q])-refs[q]);
        if (!std::isfinite(got[q])) e.pass = false;
        bool inactive = q >= positions.size() * (c.primitive ? 2 : 1)
            || (q < positions.size() && c.gamma_d == 0)
            || (c.primitive && q >= positions.size() && q < 2*positions.size() && c.gamma_e == 0);
        if (inactive && double(got[q]) != refs[q]) e.pass = false;
        err2 += d*d; ref2 += refs[q]*refs[q];
        e.max_abs = std::max(e.max_abs,d); max_ref = std::max(max_ref,std::abs(refs[q]));
    }
    e.relative_l2 = std::sqrt(err2/std::max(ref2,1e-300));
    e.max_scaled = e.max_abs/(1+max_ref); e.count = int(got.size());
    e.pass = e.pass && e.relative_l2 <= 3e-5 && e.max_scaled <= 1e-4;
    return e;
}

int main(int argc, char** argv) try {
    Options o = parse(argc, argv);
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    if (prop.major != 8) throw std::runtime_error("this build targets Ampere sm_80");
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cublasHandle_t handle; BLAS_CHECK(cublasCreate(&handle));
    BLAS_CHECK(cublasSetStream(handle,stream));
    BLAS_CHECK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
    BLAS_CHECK(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
    cublasMath_t math_mode; BLAS_CHECK(cublasGetMathMode(handle,&math_mode));
    if (math_mode != CUBLAS_PEDANTIC_MATH) throw std::runtime_error("FP32 math mode not set");
    int runtime, driver, blas_version;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
    BLAS_CHECK(cublasGetVersion(handle,&blas_version));
    std::ofstream meta(o.output+".metadata.json");
    if (!meta) throw std::runtime_error("cannot open metadata output");
    meta << "{\n  \"gpu\": \"" << prop.name << "\",\n  \"compute_capability\": \"" << prop.major << "." << prop.minor
         << "\",\n  \"global_memory_bytes\": " << prop.totalGlobalMem << ",\n  \"cuda_runtime\": " << runtime
         << ",\n  \"cuda_driver\": " << driver << ",\n  \"cublas_version\": " << blas_version
         << ",\n  \"math_mode\": \"CUBLAS_PEDANTIC_MATH\",\n  \"gemm_api\": \"cublasSgemm\",\n"
         << "  \"alpha\": 1, \"beta\": 1, \"layout\": \"column-major\",\n"
         << "  \"rounds\": " << o.rounds << ", \"target_ms\": " << o.target_ms
         << ", \"requested_samples\": " << o.samples << ",\n"
         << "  \"correctness_relative_l2_limit\": 0.00003, \"correctness_max_scaled_limit\": 0.0001,\n"
         << "  \"sizes\": [";
    for (size_t i=0;i<o.sizes.size();++i) meta << (i ? "," : "") << o.sizes[i];
    meta << "]\n}\n"; meta.close();
    std::ofstream raw(o.output+".rounds.csv"), checks(o.output+".correctness.csv");
    if (!raw || !checks) throw std::runtime_error("cannot open csv outputs");
    raw << "parent_n,case,kernel,m,n,k,ld,delta,epsilon,gamma_d,gamma_e,round,repeats,batch_ms,latency_ms,gemm_equivalent_tflops\n";
    checks << "parent_n,case,repeats,checked_values,relative_l2_error,max_absolute_error,max_scaled_error,pass\n";
    raw << std::setprecision(12); checks << std::setprecision(12);
    const float one = 1;
    auto gemm = [&](float* a,float* b,float* c,int s,int ld) {
        BLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,s,s,s,
            &one,a,ld,b,ld,&one,c,ld));
    };
    Events events;
    for (int n : o.sizes) {
        int s = n/2; size_t count = size_t(n)*n, leaf_count = size_t(s)*s;
        Buffer a(count), b(count), c(count), la(leaf_count), lb(leaf_count), lc(leaf_count);
        auto init = [&](float* p,size_t length,int seed) {
            initialize<<<std::min<size_t>((length+255)/256,4096),256,0,stream>>>(p,length,seed);
        };
        init(a.p,count,11); init(b.p,count,23); init(la.p,leaf_count,11); init(lb.p,leaf_count,23);
        init(c.p,count,37); init(lc.p,leaf_count,37);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<Case> cases;
        auto add_gemm = [&](const std::string& name,float* ap,float* bp,float* cp,int dim,int ld) {
            Case x; x.name=name; x.kernel="cublas_sgemm_pedantic"; x.s=dim; x.ld=ld; x.a=ap;x.b=bp;x.c=cp;
            x.launch=[=,&gemm](){gemm(ap,bp,cp,dim,ld);}; cases.push_back(x);
        };
        add_gemm("cublas_full",a.p,b.p,c.p,n,n);
        add_gemm("cublas_leaf_compact",la.p,lb.p,lc.p,s,s);
        add_gemm("cublas_leaf_parent_ld",a.p,b.p,c.p,s,n);
        struct Variant { const char* name; int d,e,gd,ge; };
        const Variant variants[] = {
            {"primitive_gemm",0,0,1,0},
            {"primitive_m1",1,0,1,-1}, {"primitive_m2",1,0,1,-1},
            {"primitive_m3",-1,1,0,1}, {"primitive_m4",-1,1,1,0},
            {"primitive_m5",0,-1,1,1}, {"primitive_m6",0,-1,1,1},
            {"primitive_m7",1,1,1,1}};
        for (auto v : variants) {
            Case x; x.name=v.name; x.kernel=selected_kernel(s); x.s=s;x.ld=n;
            x.delta=v.d; x.epsilon=v.e; x.gamma_d=v.gd; x.gamma_e=v.ge;
            x.a=a.p;x.b=b.p;x.c=c.p;x.primitive=true;
            float *ap=a.p,*bp=b.p,*cp=c.p; size_t off=size_t(s)*(n+1);
            x.launch=[=](){prim_inter(ap,ap+off,bp,bp+off,cp,cp+off,
                s,s,s,n,n,n,n,n,n,v.d,v.e,v.gd,v.ge,stream);}; cases.push_back(x);
        }
        if (!o.only.empty()) {
            cases.erase(std::remove_if(cases.begin(),cases.end(),[&](const Case& x){return x.name!=o.only;}),cases.end());
            if (cases.empty()) throw std::runtime_error("unknown --only case");
        }
        auto reset = [&](const Case& x) { init(x.c,x.c==lc.p ? leaf_count : count,37); };
        for (const auto& x : cases) {
            for (int repeats : {1,3}) {
                reset(x); for(int i=0;i<repeats;++i)x.launch();
                CUDA_CHECK(cudaStreamSynchronize(stream));
                Error e=validate(x,n,o.samples,repeats,stream);
                checks << n << ',' << x.name << ',' << repeats << ',' << e.count << ',' << e.relative_l2 << ','
                       << e.max_abs << ',' << e.max_scaled << ',' << e.pass << '\n'; checks.flush();
                if (!e.pass) throw std::runtime_error("FP64 validation failed: N="+std::to_string(n)+" "+x.name);
            }
        }
        std::cout << "N=" << n << " FP64 validation PASS (" << cases.size() << " cases, 1/3 accumulated calls)" << std::endl;
        if (o.validate_only) continue;
        // Warm every implementation before adaptive timing; no allocations in timed intervals.
        for (const auto& x : cases) {
            reset(x); float warm_probe=elapsed(events,stream,x,3)/3;
            int warm_reps=std::max(3,std::min(10000,int(std::ceil(20/std::max(double(warm_probe),0.0001)))));
            elapsed(events,stream,x,warm_reps);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<int> order(cases.size()); std::iota(order.begin(),order.end(),0);
        std::mt19937 rng(20261017+n);
        for (int round=0; round<o.rounds; ++round) {
            std::shuffle(order.begin(),order.end(),rng);
            for (int index : order) {
                const auto& x=cases[index]; reset(x);
                float probe=elapsed(events,stream,x,3)/3;
                int reps=std::max(3,std::min(10000,int(std::ceil(o.target_ms/std::max(double(probe),0.0001)))));
                reset(x); float batch=elapsed(events,stream,x,reps); double ms=double(batch)/reps;
                double tflops=2.0*x.s*x.s*x.s/(ms*1e9);
                raw << n << ',' << x.name << ',' << x.kernel << ',' << x.s << ',' << x.s << ',' << x.s << ',' << x.ld << ','
                    << x.delta << ',' << x.epsilon << ',' << x.gamma_d << ',' << x.gamma_e << ',' << round << ','
                    << reps << ',' << batch << ',' << ms << ',' << tflops << '\n'; raw.flush();
                std::cout << "N=" << n << " round=" << round << ' ' << x.name << " ms=" << ms << std::endl;
            }
        }
    }
    BLAS_CHECK(cublasDestroy(handle)); CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "COMPLETE " << o.output << std::endl; return 0;
} catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << std::endl; return 1; }
