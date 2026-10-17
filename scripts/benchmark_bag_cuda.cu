// Reuse the established deterministic inputs, FP64 reference and CUDA events.
#define main standalone_cost_main
#include "benchmark_strassen_fp32.cu"
#undef main
#include "bag.cuh"

struct BagOptions {
    Options base;
    std::vector<int> depths{1};
    std::string basis="both",schedule="all",backend="column",only;
    uint32_t seed=0;
};
std::vector<int> integers(const std::string& text) {
    std::istringstream input(text);std::string token;std::vector<int> result;
    while(std::getline(input,token,','))result.push_back(std::stoi(token));
    if(result.empty())throw std::runtime_error("empty integer list");
    return result;
}
BagOptions bag_options(int argc,char** argv) {
    BagOptions o;o.base.sizes={1280,1536,1792,2048};
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        auto next=[&](){if(++i==argc)throw std::runtime_error("missing "+arg);return std::string(argv[i]);};
        if(arg=="--sizes")o.base.sizes=integers(next());
        else if(arg=="--depths")o.depths=integers(next());
        else if(arg=="--basis")o.basis=next();
        else if(arg=="--schedule")o.schedule=next();
        else if(arg=="--backend")o.backend=next();
        else if(arg=="--only")o.only=next();
        else if(arg=="--seed-offset")o.seed=std::stoul(next());
        else if(arg=="--rounds")o.base.rounds=std::stoi(next());
        else if(arg=="--target-ms")o.base.target_ms=std::stod(next());
        else if(arg=="--samples")o.base.samples=std::stoi(next());
        else if(arg=="--output")o.base.output=next();
        else if(arg=="--validate-only")o.base.validate_only=true;
        else throw std::runtime_error("unknown option "+arg);
    }
    for(int n:o.base.sizes)if(n<8||n%8)throw std::runtime_error("N must be positive and divisible by8");
    for(int d:o.depths)if(d<1||d>8)throw std::runtime_error("depth must be1..8");
    if(o.basis!="both"&&o.basis!="minimal"&&o.basis!="full")throw std::runtime_error("invalid basis");
    if(o.schedule!="all"&&o.schedule!="serial"&&o.schedule!="seven"&&o.schedule!="batched")throw std::runtime_error("invalid schedule");
    if(o.backend!="column"&&o.backend!="huge")throw std::runtime_error("invalid backend");
    if(o.base.rounds<1||o.base.target_ms<=0||o.base.samples<16)throw std::runtime_error("invalid measurement options");
    return o;
}
int main(int argc,char** argv)try {
    const auto options=bag_options(argc,argv);const auto& o=options.base;
    const auto backend=options.backend=="column"?bag_cuda::Backend::ColumnTile8:bag_cuda::Backend::Huge;
    cudaDeviceProp prop;CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    cudaStream_t caller;CUDA_CHECK(cudaStreamCreateWithFlags(&caller,cudaStreamNonBlocking));
    strassen_cuda::Workspace strassen_workers;
    cublasHandle_t handle;BLAS_CHECK(cublasCreate(&handle));BLAS_CHECK(cublasSetStream(handle,caller));
    BLAS_CHECK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
    BLAS_CHECK(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
    int runtime,blas_version;CUDA_CHECK(cudaRuntimeGetVersion(&runtime));BLAS_CHECK(cublasGetVersion(handle,&blas_version));
    std::ofstream meta(o.output+".metadata.json"),raw(o.output+".rounds.csv"),checks(o.output+".correctness.csv");
    if(!meta||!raw||!checks)throw std::runtime_error("cannot create output files");
    meta<<"{\"gpu\":\""<<prop.name<<"\",\"alpha\":1,\"beta\":1,\"math_mode\":\"CUBLAS_PEDANTIC_MATH\","
        <<"\"seed_offset\":"<<options.seed<<",\"rounds\":"<<o.rounds<<",\"target_ms\":"<<o.target_ms
        <<",\"samples\":"<<o.samples<<",\"backend\":\""<<options.backend<<"\",\"basis\":\""<<options.basis
        <<"\",\"schedule\":\""<<options.schedule<<"\",\"cuda_runtime\":"<<runtime<<",\"cublas_version\":"<<blas_version<<",\"sizes\":[";
    for(size_t i=0;i<o.sizes.size();++i)meta<<(i?",":"")<<o.sizes[i];
    meta<<"],\"depths\":[";for(size_t i=0;i<options.depths.size();++i)meta<<(i?",":"")<<options.depths[i];
    meta<<"]}\n";meta.close();
    raw<<"parent_n,case,mode,round,repeats,batch_ms,latency_ms,equivalent_tflops\n"<<std::setprecision(12);
    checks<<"parent_n,case,repeats,checked_values,relative_l2_error,max_absolute_error,max_scaled_error,pass\n"<<std::setprecision(12);
    std::ofstream workspaces(o.output+".workspace.csv");workspaces<<"parent_n,depth,basis,bytes\n";
    const float one=1;Events events;
    for(int n:o.sizes) {
        const size_t count=size_t(n)*n;Buffer a(count),b(count),c(count);
        const auto init=[&](float* p,int seed){initialize<<<std::min<size_t>((count+255)/256,4096),256,0,caller>>>(p,count,uint32_t(seed)+options.seed);};
        init(a.p,11);init(b.p,23);init(c.p,37);CUDA_CHECK(cudaStreamSynchronize(caller));
        std::vector<Case> cases;std::vector<std::unique_ptr<bag_cuda::Workspace>> storage;
        auto add=[&](std::string name,std::function<void()> launch) {
            Case x;x.name=name;x.s=x.ld=n;x.a=a.p;x.b=b.p;x.c=c.p;x.launch=launch;cases.push_back(x);
        };
        auto* ap=a.p;auto* bp=b.p;auto* cp=c.p;
        add("cublas_full",[&](){BLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&one,ap,n,bp,n,&one,cp,n));});
        if(strassen_cuda::valid_shape(backend,n/2))
            add("strassen_"+options.backend+"_four_batched",[&](){strassen_cuda::one_level(backend,
                strassen_cuda::Schedule::FourBatchedAtomic,ap,bp,cp,n,caller,strassen_workers);});
        for(int depth:options.depths) {
            int leaf=n;bool valid=true;
            for(int level=0;level<depth;++level,leaf/=2)if(leaf%8||leaf<=0)valid=false;
            if(!valid||!strassen_cuda::valid_shape(backend,leaf)) {
                std::cout<<"SKIP N="<<n<<" depth="<<depth<<" invalid leaf shape\n";continue;
            }
            for(auto basis:{bag_cuda::Basis::Minimal,bag_cuda::Basis::Full}) {
                const std::string label=basis==bag_cuda::Basis::Minimal?"minimal":"full";
                if(options.basis!="both"&&options.basis!=label)continue;
                storage.emplace_back(std::make_unique<bag_cuda::Workspace>(n,depth,basis));
                auto* ws=storage.back().get();
                workspaces<<n<<','<<depth<<','<<label<<','<<ws->bytes()<<'\n';workspaces.flush();
                for(auto schedule:{bag_cuda::Schedule::Serial,bag_cuda::Schedule::SevenStream,bag_cuda::Schedule::Batched}) {
                    const std::string plan=schedule==bag_cuda::Schedule::Serial?"serial":schedule==bag_cuda::Schedule::SevenStream?"seven":"batched";
                    if(options.schedule!="all"&&options.schedule!=plan)continue;
                    add("bag_"+options.backend+"_"+label+"_d"+std::to_string(depth)+"_"+plan,
                        [=](){bag_cuda::matmul(ap,bp,cp,n,caller,*ws,backend,schedule);});
                }
            }
        }
        if(!options.only.empty())cases.erase(std::remove_if(cases.begin(),cases.end(),
            [&](const Case& x){return x.name!=options.only&&x.name!="cublas_full";}),cases.end());
        if(cases.size()<2)throw std::runtime_error("no legal BAG/Strassen case selected");
        for(const auto& x:cases)for(int repeats:{1,3}) {
            init(cp,37);for(int r=0;r<repeats;++r)x.launch();CUDA_CHECK(cudaStreamSynchronize(caller));
            const auto e=validate(x,n,o.samples,repeats,caller,11+options.seed,23+options.seed,37+options.seed);
            checks<<n<<','<<x.name<<','<<repeats<<','<<e.count<<','<<e.relative_l2<<','<<e.max_abs<<','<<e.max_scaled<<','<<e.pass<<'\n';checks.flush();
            if(!e.pass)throw std::runtime_error("FP64 validation failed N="+std::to_string(n)+" "+x.name);
        }
        std::cout<<"N="<<n<<" FP64 PASS cases="<<cases.size()<<std::endl;
        if(o.validate_only)continue;
        for(const auto& x:cases) {
            init(cp,37);const float p=elapsed(events,caller,x,3)/3;
            elapsed(events,caller,x,std::max(3,std::min(10000,int(std::ceil(20/std::max(double(p),.0001))))));
        }
        std::vector<int> order(cases.size());std::iota(order.begin(),order.end(),0);std::mt19937 rng(20261017+n);
        for(int round=0;round<o.rounds;++round) {
            std::shuffle(order.begin(),order.end(),rng);
            for(int index:order) {
                const auto& x=cases[index];init(cp,37);const float p=elapsed(events,caller,x,3)/3;
                const int repeats=std::max(3,std::min(10000,int(std::ceil(o.target_ms/std::max(double(p),.0001)))));
                init(cp,37);const double batch=elapsed(events,caller,x,repeats),ms=batch/repeats;
                raw<<n<<','<<x.name<<",endtoend,"<<round<<','<<repeats<<','<<batch<<','<<ms<<','<<2.*n*n*n/(ms*1e9)<<'\n';raw.flush();
                std::cout<<n<<" round="<<round<<' '<<x.name<<" ms="<<ms<<std::endl;
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(caller));
    }
    BLAS_CHECK(cublasDestroy(handle));CUDA_CHECK(cudaStreamDestroy(caller));
    std::cout<<"COMPLETE "<<o.output<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1;}
