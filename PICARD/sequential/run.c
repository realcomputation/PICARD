#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
  #include <windows.h>
  #include <io.h>
  #include <fcntl.h>
#else
  #include <time.h>
  #include <unistd.h>
#endif

#ifndef m
#define m 20
#endif

#ifndef N
#define N (1 << (m+2))
#endif

static double wall_seconds_now(void){
#ifdef _WIN32
    static LARGE_INTEGER f = {0};
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9;
#endif
}

typedef int64_t fx_t;

fx_t fx_add(fx_t a, fx_t b) {
    return a + b;
}

fx_t fx_sub(fx_t a, fx_t b) {
    return a - b;
}

fx_t calculate_riemann_interval(fx_t x, fx_t _m){
    return (x >> (_m + 1));
}

double fx_to_double(fx_t fx_val) {
    return (double)fx_val / (1LL << (61));
}

fx_t double_to_fx(double val) {
    return (fx_t)(val * (1LL << (61)));
}

void solve_integral_equation(fx_t *result) {
    for (int i = 0; i <= N; i++) {
        result[i] = 0;
    }
    fx_t *temp = (fx_t*) malloc((N + 1) * sizeof(fx_t));
    for (int iter = 0; iter < m; iter++) {
        printf("%d\n",iter);
        temp[N/2] = 1LL << (61); 
        for (int i = N/2 + 1; i <= N; i++) {
            fx_t prev_val = temp[i-1]; 
            fx_t term = ((result[i]) >> (m + 1)); 
            temp[i] = fx_add(prev_val, term);
        }
        for (int i = N/2 - 1; i >= 0; i--) {
            fx_t next_val = temp[i+1];  
            fx_t term = ((result[i]) >> (m + 1)); 
            temp[i] = fx_sub(next_val, term);
        }
        for (int i = 0; i <= N; i++) {
            result[i] = temp[i];
        }
        
    }
    free(temp);
}
void compare_with_exponential(fx_t *result) {   
    double max_error = 0.0;
    double max_error_x = 0.0;
    
    for (int i = 0; i <= N; i++) {
        double x = -1.0 + (2.0 * i) / N;
        double computed_val = fx_to_double(result[i]);
        double exact_val = exp(x);
        double error = fabs(computed_val - exact_val);
        if (error > max_error) {
            max_error = error;
            max_error_x = x;
        }
    }
    
    
    printf("Maximum error: %.10f at x = %.6f\n", max_error, max_error_x);
}

int main() {
    printf("Starting recursive integral equation solver...\n");
    printf("Parameters: m = %d, N = %d\n", m, N);
    printf("Interval [-1, 1] divided into %d subintervals\n", N);
    printf("N = %d (should be %d)\n", N, 1 << (m));
    fx_t *result = (fx_t*) malloc((N + 1) * sizeof(fx_t));
    
    if (result == NULL) {
        printf("Error: Memory allocation failed!\n");
        return 1;
    }
    double t0 = wall_seconds_now();
    solve_integral_equation(result);
    double t1 = wall_seconds_now();
    double solve_time = t1 - t0;
    
    printf("Time taken for solve_integral_equation: %.6f seconds\n", solve_time);
    compare_with_exponential(result);
    free(result);
    return 0;

}
