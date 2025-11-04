#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#ifdef _WIN32
#include <windows.h>
#endif

typedef int64_t fx_t;
#define FX_ONE ((fx_t)1LL << 61)
#ifndef m
#define m 28
#endif
#ifndef N
#define N (1ULL << m)
#endif
#ifndef P
#define P 2
#endif
#define CUDA_OK(msg) do{cudaError_t _e=cudaGetLastError();if(_e!=cudaSuccess){exit(1);} _e=cudaDeviceSynchronize();if(_e!=cudaSuccess){exit(1);}}while(0)

__global__ void worker_step2_kernel(fx_t* a){
    int64_t i=blockIdx.x;if(i>=P)return;if(threadIdx.x!=0)return;
    const int64_t mid=(int64_t)(N/2),L=(int64_t)(N/(2*P));
    int64_t start=((i*(int64_t)N)/(2*P))+mid+1,startb=mid-((i*(int64_t)N)/(2*P))-1;
    if(start<=(int64_t)N)a[start]>>=(m-1);
    if(startb>=0)a[startb]>>=(m-1);
    for(int64_t j=1;j<L;++j){
        if(start+j<=(int64_t)N){a[start+j]>>=(m-1);a[start+j]+=a[start+j-1];}
        if(startb-j>=0){a[startb-j]>>=(m-1);a[startb-j]+=a[startb-j+1];}
    }
}
__global__ void compute_g_kernel(const fx_t* a,fx_t* g){
    if(blockIdx.x||threadIdx.x)return;
    const int64_t mid=(int64_t)(N/2),L=(int64_t)(N/(2*P));
    g[P-1]=FX_ONE;
    for(int64_t j=1;j<P;++j){
        int64_t idxR=mid+j*L;fx_t bRj=(idxR>=0&&(uint64_t)idxR<=N)?a[idxR]:0;
        g[P-1+j]=g[P+j-2]+bRj;
    }
    for(int64_t j=1;j<P;++j){
        int64_t idxL=mid-j*L;fx_t bLj=(idxL>=0&&(uint64_t)idxL<=N)?a[idxL]:0;
        g[P-1-j]=g[P-j]-bLj;
    }
}
__global__ void worker_step4_kernel(fx_t* a,const fx_t* g){
    int64_t i=blockIdx.x;if(i>=P)return;if(threadIdx.x!=0)return;
    const int64_t mid=(int64_t)(N/2),L=(int64_t)(N/(2*P));
    int64_t start=((i*(int64_t)N)/(2*P))+mid+1,startb=mid-((i*(int64_t)N)/(2*P))-1;
    for(int64_t j=0;j<L;++j){
        if(start+j<=(int64_t)N)a[start+j]+=g[P-1+i];
        if(startb-j>=0)a[startb-j]=g[P-1-i]-a[startb-j];
    }
}
static double wall_seconds_now(){
#ifdef _WIN32
    static LARGE_INTEGER f={0};LARGE_INTEGER c;if(!f.QuadPart)QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);return(double)c.QuadPart/(double)f.QuadPart;
#else
    timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return(double)ts.tv_sec+(double)ts.tv_nsec*1e-9;
#endif
}
int main(){
    const uint64_t points=N+1,bytesA=(points)*sizeof(fx_t),bytesG=(2*P-1)*sizeof(fx_t);
    fx_t*d_a=nullptr,*d_g=nullptr;cudaMalloc(&d_a,bytesA);cudaMalloc(&d_g,bytesG);
    cudaMemset(d_a,0,bytesA);fx_t one=FX_ONE;cudaMemcpy(d_a+(N/2),&one,sizeof(fx_t),cudaMemcpyHostToDevice);
    dim3 blocksP(P),oneThread(1);double t0=wall_seconds_now();
    for(int iter=0;iter<m;++iter){
        worker_step2_kernel<<<blocksP,oneThread>>>(d_a);CUDA_OK("step2");
        compute_g_kernel<<<1,1>>>(d_a,d_g);CUDA_OK("compute_g");
        worker_step4_kernel<<<blocksP,oneThread>>>(d_a,d_g);CUDA_OK("step4");
    }
    double t1=wall_seconds_now();
    std::vector<fx_t>h_a(points);cudaMemcpy(h_a.data(),d_a,bytesA,cudaMemcpyDeviceToHost);
    double max_err=0.0;
    for(uint64_t i=0;i<=N;++i){
        double x=-1.0+2.0*(double)i/(double)N;
        double approx=(double)h_a[i]/(double)FX_ONE;
        double ref=exp(x);
        double e=fabs(approx-ref);
        if(e>max_err)max_err=e;
    }
    std::cout<<std::fixed<<"Elasped Times:"<<std::setprecision(6)<<(t1-t0)<<" second\nMaximum Error:"<<std::scientific<<max_err<<"\n";
    cudaFree(d_g);cudaFree(d_a);
}
