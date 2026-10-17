// Reuse the established input generator, FP64 checks, error limits and timing.
#define main standalone_cost_main
#include "benchmark_strassen_fp32.cu"
#undef main
#include "engine.cuh"

using strassen_cuda::Backend;
using strassen_cuda::Schedule;
struct ExperimentOptions {
    Options base;
    std::string mode="ablation",backend_filter,case_filter;
};
ExperimentOptions experiment_options(int argc,char** argv) {
    ExperimentOptions o;o.base.sizes={2048,8192,20480};o.base.rounds=3;o.base.target_ms=100;
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        auto next=[&](){if(++i==argc)throw std::runtime_error("missing option value");return std::string(argv[i]);};
        if(arg=="--mode")o.mode=next();
        else if(arg=="--backends")o.backend_filter=","+next()+",";
        else if(arg=="--only")o.case_filter=next();
        else if(arg=="--sizes"){o.base.sizes.clear();std::istringstream s(next());std::string t;while(std::getline(s,t,','))o.base.sizes.push_back(std::stoi(t));}
        else if(arg=="--rounds")o.base.rounds=std::stoi(next());
        else if(arg=="--target-ms")o.base.target_ms=std::stod(next());
        else if(arg=="--samples")o.base.samples=std::stoi(next());
        else if(arg=="--output")o.base.output=next();
        else if(arg=="--validate-only")o.base.validate_only=true;
        else throw std::runtime_error("unknown option "+arg);
    }
    if(o.mode!="ablation"&&o.mode!="endtoend")throw std::runtime_error("invalid mode");
    for(int n:o.base.sizes)if(n<32||n%8)throw std::runtime_error("N must be divisible by 8");
    if(o.base.rounds<1||o.base.samples<16||o.base.target_ms<=0)throw std::runtime_error("invalid options");
    return o;
}
int main(int argc,char** argv)try {
    auto options=experiment_options(argc,argv);auto o=options.base;
    cudaDeviceProp prop;CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    strassen_cuda::Workspace workspace;
    cublasHandle_t handle;BLAS_CHECK(cublasCreate(&handle));BLAS_CHECK(cublasSetStream(handle,stream));
    BLAS_CHECK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));BLAS_CHECK(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
    int blas_version,runtime;BLAS_CHECK(cublasGetVersion(handle,&blas_version));CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::ofstream meta(o.output+".metadata.json");
    if(!meta)throw std::runtime_error("cannot write metadata");
    meta<<"{\"gpu\":\""<<prop.name<<"\",\"mode\":\""<<options.mode<<"\",\"alpha\":1,\"beta\":1,"
        <<"\"math_mode\":\"CUBLAS_PEDANTIC_MATH\",\"rounds\":"<<o.rounds<<",\"target_ms\":"<<o.target_ms
        <<",\"samples\":"<<o.samples<<",\"cuda_runtime\":"<<runtime<<",\"cublas_version\":"<<blas_version
        <<",\"backend_filter\":\""<<options.backend_filter<<"\",\"case_filter\":\""<<options.case_filter<<"\",\"sizes\":[";
    for(size_t i=0;i<o.sizes.size();++i)meta<<(i?",":"")<<o.sizes[i];meta<<"]}\n";meta.close();
    std::ofstream raw(o.output+".rounds.csv"),checks(o.output+".correctness.csv");
    raw<<"parent_n,case,mode,round,repeats,batch_ms,latency_ms,equivalent_tflops\n";
    checks<<"parent_n,case,repeats,checked_values,relative_l2_error,max_absolute_error,max_scaled_error,pass\n";
    raw<<std::setprecision(12);checks<<std::setprecision(12);
    Events events;const float one=1;
    for(int n:o.sizes) {
        int s=n/2;size_t count=size_t(n)*n;Buffer a(count),b(count),c(count);
        auto init=[&](float* p,int seed){initialize<<<std::min<size_t>((count+255)/256,4096),256,0,stream>>>(p,count,seed);};
        init(a.p,11);init(b.p,23);init(c.p,37);CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<Case> cases;
        auto add=[&](std::string name,int size,bool primitive,int delta,int epsilon,int gd,int ge,std::function<void()> launch){
            Case x;x.name=name;x.s=size;x.ld=n;x.a=a.p;x.b=b.p;x.c=c.p;x.primitive=primitive;
            x.delta=delta;x.epsilon=epsilon;x.gamma_d=gd;x.gamma_e=ge;x.launch=launch;cases.push_back(x);
        };
        auto* ap=a.p;auto* bp=b.p;auto* cp=c.p;
        auto blas=[=,&one](int size){BLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,size,size,size,&one,ap,n,bp,n,&one,cp,n));};
        add("cublas_full",n,false,0,0,1,0,[=](){blas(n);});
        if(options.mode=="ablation")add("cublas_leaf",s,false,0,0,1,0,[=](){blas(s);});
        for(auto config:strassen_cuda::backends) {
            if(!options.backend_filter.empty()&&options.backend_filter.find(","+std::string(config.name)+",")==std::string::npos)continue;
            // Original unmasked kernels are only legal at their tile boundaries.
            if(!strassen_cuda::valid_shape(config.backend,s))continue;
            auto backend=config.backend;
            if(options.mode=="ablation") {
                size_t off=size_t(s)*(n+1);
                for(bool atomic:{true,false}) {
                    if(!atomic&&!config.normal)continue;
                    std::string suffix=atomic?"_atomic":"_normal";
                    add(std::string(config.name)+"_gemm"+suffix,s,true,0,0,1,0,[=](){
                        if(atomic)strassen_cuda::primitive<0,0,1,0,true>(backend,ap,ap+off,bp,bp+off,cp,cp+off,s,n,stream);
                        else strassen_cuda::primitive<0,0,1,0,false>(backend,ap,ap+off,bp,bp+off,cp,cp+off,s,n,stream);
                    });
                    add(std::string(config.name)+"_m7"+suffix,s,true,1,1,1,1,[=](){
                        if(atomic)strassen_cuda::primitive<1,1,1,1,true>(backend,ap,ap+off,bp,bp+off,cp,cp+off,s,n,stream);
                        else strassen_cuda::primitive<1,1,1,1,false>(backend,ap,ap+off,bp,bp+off,cp,cp+off,s,n,stream);
                    });
                }
            } else {
                struct Plan {const char* name;Schedule schedule;bool normal;bool paired=false;};
                for(auto plan:{Plan{"serial_atomic",Schedule::SerialAtomic,false},Plan{"serial_normal",Schedule::SerialNormal,true},
                               Plan{"seven_atomic",Schedule::SevenAtomic,false},Plan{"dependency_normal",Schedule::DependencyNormal,true},
                               Plan{"paired_atomic",Schedule::PairedAtomic,false,true},Plan{"paired_normal",Schedule::PairedNormal,true,true}}) {
                    if(plan.normal&&!config.normal)continue;
                    if(plan.paired&&!config.paired)continue;
                    add(std::string(config.name)+"_"+plan.name,n,false,0,0,1,0,
                        [=,&workspace](){strassen_cuda::one_level(backend,plan.schedule,ap,bp,cp,n,stream,workspace);});
                }
            }
        }
        if(!options.case_filter.empty())cases.erase(std::remove_if(cases.begin(),cases.end(),[&](const Case& x){return x.name!=options.case_filter;}),cases.end());
        if(cases.empty())throw std::runtime_error("no selected cases");
        for(const auto& x:cases) {
            for(int repeats:{1,3}) {
                init(cp,37);for(int t=0;t<repeats;++t)x.launch();CUDA_CHECK(cudaStreamSynchronize(stream));
                auto e=validate(x,n,o.samples,repeats,stream);
                checks<<n<<','<<x.name<<','<<repeats<<','<<e.count<<','<<e.relative_l2<<','<<e.max_abs<<','<<e.max_scaled<<','<<e.pass<<'\n';checks.flush();
                if(!e.pass)throw std::runtime_error("FP64 validation failed: "+std::to_string(n)+" "+x.name);
            }
        }
        std::cout<<"N="<<n<<" correctness PASS cases="<<cases.size()<<std::endl;
        if(o.validate_only)continue;
        for(const auto& x:cases){init(cp,37);float p=elapsed(events,stream,x,3)/3;
            elapsed(events,stream,x,std::max(3,std::min(10000,int(std::ceil(20/std::max(double(p),.0001))))));}
        std::vector<int> order(cases.size());std::iota(order.begin(),order.end(),0);std::mt19937 rng(20261017+n);
        for(int round=0;round<o.rounds;++round) {
            std::shuffle(order.begin(),order.end(),rng);
            for(int index:order) {
                const auto& x=cases[index];init(cp,37);float p=elapsed(events,stream,x,3)/3;
                int reps=std::max(3,std::min(10000,int(std::ceil(o.target_ms/std::max(double(p),.0001)))));
                init(cp,37);double batch=elapsed(events,stream,x,reps),ms=batch/reps;
                raw<<n<<','<<x.name<<','<<options.mode<<','<<round<<','<<reps<<','<<batch<<','<<ms<<','<<2.*x.s*x.s*x.s/(ms*1e9)<<'\n';raw.flush();
                std::cout<<n<<" round="<<round<<' '<<x.name<<" ms="<<ms<<std::endl;
            }
        }
    }
    BLAS_CHECK(cublasDestroy(handle));CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout<<"COMPLETE "<<o.output<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1;}
