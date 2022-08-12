#include <cuda_fp16.h>

extern __global__ void fast_add_4to3_single(float* s, float*d, float mul);

extern __global__ void fast_add_3to3_single(float* s, float*d, float mul);

extern __global__ void fast_add_4to3_double(double* s, double*d, double mul);

extern __global__ void fast_add_3to3_double(double* s, double*d, double mul);

extern __global__ void force_solve_single(float* pos, float* mass, float* acc_phi, float G, float eps, int n_particles);

extern __global__ void float_array_2_half2_array(float* s, half2* d);

extern __global__ void half_force_solver(half2* part_pos, float mass, half2 eps, float G, float* output, int n_particles);
