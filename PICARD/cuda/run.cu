#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

using fx_t = std::int64_t;

constexpr fx_t FX_ONE = static_cast<fx_t>(1ULL << 61);

#ifndef m
#define m 20
#endif

#ifndef N
#define N (1ULL << (m + 2))
#endif

#ifndef P
#define P 2
#endif

inline void cuda_check(const char* label) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cerr << "CUDA error after " << label << ": "
                  << cudaGetErrorString(e) << std::endl;
        std::exit(1);
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::cerr << "CUDA sync error after " << label << ": "
                  << cudaGetErrorString(e) << std::endl;
        std::exit(1);
    }
}

__global__ void worker_step2_kernel(fx_t* a) {
    std::int64_t i = blockIdx.x;
    if (i >= P) return;
    if (threadIdx.x != 0) return;

    const std::int64_t mid = static_cast<std::int64_t>(N / 2);
    const std::int64_t L   = static_cast<std::int64_t>(N / (2 * P));

    std::int64_t start  = (i * static_cast<std::int64_t>(N)) / (2 * P) + mid + 1;
    std::int64_t startb = mid - (i * static_cast<std::int64_t>(N)) / (2 * P) - 1;

    if (start <= static_cast<std::int64_t>(N))
        a[start] >>= (m + 1);
    if (startb >= 0)
        a[startb] >>= (m + 1);

    for (std::int64_t j = 1; j < L; ++j) {
        if (start + j <= static_cast<std::int64_t>(N)) {
            a[start + j] >>= (m + 1);
            a[start + j] += a[start + j - 1];
        }
        if (startb - j >= 0) {
            a[startb - j] >>= (m + 1);
            a[startb - j] += a[startb - j + 1];
        }
    }
}

__global__ void compute_g_kernel(const fx_t* a, fx_t* g) {
    if (blockIdx.x || threadIdx.x) return;

    const std::int64_t mid = static_cast<std::int64_t>(N / 2);
    const std::int64_t L   = static_cast<std::int64_t>(N / (2 * P));

    g[P - 1] = FX_ONE;

    for (std::int64_t j = 1; j < P; ++j) {
        std::int64_t idxR = mid + j * L;
        fx_t bRj = (idxR >= 0 && static_cast<std::uint64_t>(idxR) <= N) ? a[idxR] : 0;
        g[P - 1 + j] = g[P + j - 2] + bRj;
    }

    for (std::int64_t j = 1; j < P; ++j) {
        std::int64_t idxL = mid - j * L;
        fx_t bLj = (idxL >= 0 && static_cast<std::uint64_t>(idxL) <= N) ? a[idxL] : 0;
        g[P - 1 - j] = g[P - j] - bLj;
    }
}

__global__ void worker_step4_kernel(fx_t* a, const fx_t* g) {
    std::int64_t i = blockIdx.x;
    if (i >= P) return;
    if (threadIdx.x != 0) return;

    const std::int64_t mid = static_cast<std::int64_t>(N / 2);
    const std::int64_t L   = static_cast<std::int64_t>(N / (2 * P));

    std::int64_t start  = (i * static_cast<std::int64_t>(N)) / (2 * P) + mid + 1;
    std::int64_t startb = mid - (i * static_cast<std::int64_t>(N)) / (2 * P) - 1;

    for (std::int64_t j = 0; j < L; ++j) {
        if (start + j <= static_cast<std::int64_t>(N))
            a[start + j] += g[P - 1 + i];
        if (startb - j >= 0)
            a[startb - j] = g[P - 1 - i] - a[startb - j];
    }
}

static double wall_seconds_now() {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (!freq.QuadPart) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    return static_cast<double>(counter.QuadPart) /
           static_cast<double>(freq.QuadPart);
#else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) +
           static_cast<double>(ts.tv_nsec) * 1e-9;
#endif
}

int main() {
    const std::uint64_t points = N + 1;
    const std::size_t bytesA = points * sizeof(fx_t);
    const std::size_t bytesG = (2 * P - 1) * sizeof(fx_t);

    fx_t* d_a = nullptr;
    fx_t* d_g = nullptr;

    cudaMalloc(&d_a, bytesA);
    cudaMalloc(&d_g, bytesG);

    cudaMemset(d_a, 0, bytesA);

    fx_t one = FX_ONE;
    cudaMemcpy(d_a + (N / 2), &one, sizeof(fx_t), cudaMemcpyHostToDevice);

    dim3 blocksP(P);
    dim3 oneThread(1);

    double t0 = wall_seconds_now();

    for (int iter = 0; iter < m; ++iter) {
        worker_step2_kernel<<<blocksP, oneThread>>>(d_a);
        cuda_check("worker_step2_kernel");

        compute_g_kernel<<<1, 1>>>(d_a, d_g);
        cuda_check("compute_g_kernel");

        worker_step4_kernel<<<blocksP, oneThread>>>(d_a, d_g);
        cuda_check("worker_step4_kernel");
    }

    double t1 = wall_seconds_now();

    std::vector<fx_t> h_a(points);
    cudaMemcpy(h_a.data(), d_a, bytesA, cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (std::uint64_t i = 0; i <= N; ++i) {
        double x      = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(N);
        double approx = static_cast<double>(h_a[i]) / static_cast<double>(FX_ONE);
        double ref    = std::exp(x);
        double e      = std::fabs(approx - ref);
        if (e > max_err) max_err = e;
    }

    std::cout << std::fixed
              << "Elapsed Time: " << std::setprecision(6) << (t1 - t0) << " seconds\n"
              << "Maximum Error: " << std::scientific << max_err << "\n";

    cudaFree(d_g);
    cudaFree(d_a);

    return 0;
}

