#include "kernels.h"

__global__
void fast_add_4to3_single(float* s, float*d, float mul){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d[i*3] = d[i*3] + s[i*4] * mul;
    d[i*3 + 1] = d[i*3 + 1] + s[i*4 + 1] * mul;
    d[i*3 + 2] = d[i*3 + 2] + s[i*4 + 2] * mul;

}

__global__
void fast_add_3to3_single(float *s, float *d, float mul){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d[i*3] = d[i*3] + s[i*3] * mul;
    d[i*3 + 1] = d[i*3 + 1] + s[i*3 + 1] * mul;
    d[i*3 + 2] = d[i*3 + 2] + s[i*3 + 2] * mul;

}

__global__
void fast_add_4to3_double(double* s, double*d, double mul){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d[i*3] = d[i*3] + s[i*4] * mul;
    d[i*3 + 1] = d[i*3 + 1] + s[i*4 + 1] * mul;
    d[i*3 + 2] = d[i*3 + 2] + s[i*4 + 2] * mul;

}

__global__
void fast_add_3to3_double(double *s, double *d, double mul){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d[i*3] = d[i*3] + s[i*3] * mul;
    d[i*3 + 1] = d[i*3 + 1] + s[i*3 + 1] * mul;
    d[i*3 + 2] = d[i*3 + 2] + s[i*3 + 2] * mul;

}