#include <cuda_fp16.h>

extern __global__ void fast_add_4to3_single(float* s, float*d, float mul);

extern __global__ void fast_add_3to3_single(float* s, float*d, float mul);

extern __global__ void fast_add_4to3_double(double* s, double*d, double mul);

extern __global__ void fast_add_3to3_double(double* s, double*d, double mul);

extern __global__ void force_solve_single(float* pos, float* mass, float* acc_phi, float G, float eps, int n_particles);