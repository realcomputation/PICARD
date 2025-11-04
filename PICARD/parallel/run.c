#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <math.h>

#ifdef _WIN32
  #include <windows.h>
  #include <io.h>
  #include <fcntl.h>
#else
  #include <time.h>
  #include <unistd.h>
#endif

typedef int64_t fx_t;
#define FX_ONE ((fx_t)1LL << 61)

static inline fx_t fx_add(fx_t a, fx_t b){ return a+b; }
static inline fx_t fx_sub(fx_t a, fx_t b){ return a-b; }
static inline double fx_to_double(fx_t x){ return (double)x / (double)FX_ONE; }

#ifndef m
#define m 30
#endif

#ifndef N
#define N (1ULL << m)
#endif

#ifndef P
#define P 2
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

fx_t *array;
fx_t *g;

typedef struct {
    int64_t tid;
} ThreadArg;

void* worker_step2(void *arg){
    ThreadArg *ta = (ThreadArg*)arg;
    int64_t i = ta->tid;

    int64_t start  = ((i * N) / (2 * P)) + N/2 + 1;
    int64_t startb = (N / 2) - ((i * N) / (2 * P)) - 1;

    array[start]  = array[start]  >> (m-1);
    array[startb] = array[startb] >> (m-1);

    for(int64_t j=1;j<(N/(2*P));j++){
        array[start + j]  = array[start + j]  >> (m-1);
        array[startb - j] = array[startb - j] >> (m-1);
        array[start + j]  = fx_add(array[start + j], array[start + j - 1]);
        array[startb - j] = fx_add(array[startb - j], array[startb - j + 1]);
    }
    return NULL;
}

void* worker_step4(void *arg){
    ThreadArg *ta = (ThreadArg*)arg;
    int64_t i = ta->tid;

    int64_t start  = ((i * N) / (2 * P)) + N/2 + 1;
    int64_t startb = (N / 2) - ((i * N) / (2 * P)) - 1;

    for(int64_t j=0;j<(N/(2*P));j++){
        array[start + j]  = fx_add(array[start + j], g[P-1+i]);
        array[startb - j] = fx_sub(g[P-1-i], array[startb - j]);
    }
    return NULL;
}

void parallel_prefix_sum(){
    pthread_t th[P];
    ThreadArg args[P];

    for(int64_t i=0;i<P;i++){
        args[i].tid=i;
        pthread_create(&th[i],NULL,worker_step2,&args[i]);
    }
    for(int64_t i=0;i<P;i++) pthread_join(th[i],NULL);

    g = malloc(sizeof(fx_t)*(2*P-1));
    g[P-1] = FX_ONE;
    for(int64_t j=1;j<P;j++){
        g[P-1-j] = fx_sub(g[P-j], array[N/2 - (j*N)/(2*P)]);
        g[P-1+j] = fx_add(g[P+j-2], array[N/2 + (j*N)/(2*P)]);
    }

    for(int64_t i=0;i<P;i++){
        args[i].tid=i;
        pthread_create(&th[i],NULL,worker_step4,&args[i]);
    }
    for(int64_t i=0;i<P;i++) pthread_join(th[i],NULL);

    free(g);
}

double compute_max_error(){
    double max_err=0.0;
    for(size_t i=0;i<=N;i++){
        double x=-1.0+2.0*(double)i/(double)N;
        double f_num=fx_to_double(array[i]);
        double f_sol=exp(x);
        double e=fabs(f_num-f_sol);
        if(e>max_err) max_err=e;
    }
    return max_err;
}

int main(){
    array = calloc(N+1,sizeof(fx_t));
    array[N/2] = FX_ONE;

    double t0 = wall_seconds_now();
    for (int64_t i = 0; i < m; i ++){
        parallel_prefix_sum();
    }
    double t1 = wall_seconds_now();

    double max_err=compute_max_error();

    printf("Done. N=%d (points=%d)  P=%d  m=%d\n",N,N+1,P,m);
    printf("Elapsed: %.6f s\n", t1-t0);
    printf("Max error vs exp(x): %.12e\n", max_err);

    free(array);
    return 0;
}
