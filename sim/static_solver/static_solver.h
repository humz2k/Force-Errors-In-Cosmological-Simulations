#include <cuda_fp16.h>

extern "C" { void half_precision(half* h_eval_pos, half* h_part_pos, half* h_mass, float* h_output, int n_evals, int n_particles, float G, float eps, unsigned long long* timer); }

extern "C" { void single_precision(float* h_eval_pos, float* h_pos, float* h_mass, float* h_output, int n_evals, int n_particles, float G, float eps, unsigned long long* timer); }

extern "C" { void double_precision(double* h_eval_pos, double* h_part_pos, double* h_mass, double* h_output, int n_evals, int n_particles, double G, double eps, unsigned long long* timer); }

extern "C" { void single_precision_shared_mem_cuda(float* h_eval_pos, float* h_part_pos, float* h_mass, float* h_output, int n_evals, int n_particles, float G, float eps, unsigned long long* timer); }

extern "C" { void half_precision_shared_mem_cuda(half* h_eval_pos, half* h_part_pos, half* h_mass, float* h_output, int n_evals, int n_particles, float G, float eps, unsigned long long* timer); }