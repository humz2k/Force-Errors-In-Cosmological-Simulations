//kernels.h
#include <cuda_fp16.h>

extern __global__ void single_force_solver(float* eval_pos, float* part_pos, float* mass, float* output, float G, float eps, int n_particles, int n_evals);

extern __global__ void double_force_solver(double* eval_pos, double* part_pos, double* mass, double* output, double G, double eps, int n_particles, int n_evals);

extern __global__ void half_force_solver(half* eval_pos, half* part_pos, half* mass, float* output, float G, float eps, int n_particles, int n_evals);

extern __global__ void single_force_solver_shared_mem_cuda(float* eval_pos, float* part_pos, float* mass, float* output, float G, float eps, int n_particles, int n_evals);

extern __global__ void half_force_solver_shared_mem_cuda(half* eval_pos, half* part_pos, half* mass, float* output, float G, float eps, int n_particles, int n_evals);