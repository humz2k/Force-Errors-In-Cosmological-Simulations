#include <cuda_fp16.h>

extern "C" { void single_precision(float* input_pos, float* input_vel, float* input_mass, int n_particles, int steps, float G, float eps, float dt, unsigned long long* timer); }

extern "C" {void half_precision(float* input_pos, float* input_vel, float particle_mass, int n_particles, int steps, float G, half2* eps_array, float dt, unsigned long long* timer); }