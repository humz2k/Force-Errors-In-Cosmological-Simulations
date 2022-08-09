#include <cuda_fp16.h>

extern __global__ void double_distances(double* eval_pos, double* part_pos, double* output, int n_particles, int n_evals);

extern __global__ void single_distances(float* eval_pos, float* part_pos, float* output, int n_particles, int n_evals);

extern __global__ void half_distances(half2* eval_pos, half2* part_pos, half2* output, int n_particles, int n_evals);

extern __global__ void half2half2_array(half* input, half2* output);

extern __global__ void half_phis(half2* eval_pos, half2* part_pos, half2* masses, half2* G, half2* eps, half2* output, int n_particles, int n_evals);