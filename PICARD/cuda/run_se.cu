#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>

typedef int64_t fx_t;
#define FX_ONE ((fx_t)1LL << 61)

#ifndef m
#define m 26
#endif
#ifndef N
#define N (1u << (m+2))
#endif

static inline double fx_to_double(fx_t v){ return double(v) / double(FX_ONE); }

__global__ void solve_kernel(fx_t* result, fx_t* temp, int M, uint64_t NN){
  for(uint64_t i=0;i<=NN;i++) result[i]=0;
  for(int iter=0; iter<M; ++iter){
    temp[NN/2] = FX_ONE;
    for(uint64_t i=NN/2+1;i<=NN;i++){
      fx_t prev = temp[i-1];
      fx_t term = (result[i] >> (M+1));
      temp[i] = prev + term;
    }
    for(int64_t i=(int64_t)NN/2-1;i>=0;i--){
      fx_t next = temp[i+1];
      fx_t term = (result[i] >> (M+1));
      temp[i] = next - term;
    }
    for(uint64_t i=0;i<=NN;i++) result[i]=temp[i];
  }
}

int main(){
  using clk=std::chrono::steady_clock;

  uint64_t NN = N;
  fx_t *d_res=nullptr, *d_tmp=nullptr;
  cudaMalloc(&d_res, (NN+1)*sizeof(fx_t));
  cudaMalloc(&d_tmp, (NN+1)*sizeof(fx_t));

  auto t0 = clk::now();
  solve_kernel<<<1,1>>>(d_res, d_tmp, m, NN);
  cudaDeviceSynchronize();
  auto t1 = clk::now();

  std::vector<fx_t> h(NN+1);
  cudaMemcpy(h.data(), d_res, (NN+1)*sizeof(fx_t), cudaMemcpyDeviceToHost);

  double max_err=0.0, max_x=0.0;
  for(uint64_t i=0;i<=NN;i++){
    double x = -1.0 + 2.0*double(i)/double(NN);
    double e = std::fabs(fx_to_double(h[i]) - std::exp(x));
    if(e>max_err){ max_err=e; max_x=x; }
  }

  double secs = std::chrono::duration<double>(t1-t0).count();
  std::cout<<"Starting sequential CUDA kernel...\n";
  std::cout<<"Parameters: m="<<m<<", N="<<NN<<"\n";
  std::cout<<"Time: "<<secs<<" s\n";
  std::cout<<"Maximum error: "<<std::scientific<<max_err<<" at x="<<max_x<<"\n";

  cudaFree(d_res); cudaFree(d_tmp);
  return 0;
}
