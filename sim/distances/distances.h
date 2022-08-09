#include <cuda_fp16.h>

extern "C" {

    void double_precision(double* h_eval_pos, double* h_part_pos, double* h_output, int n_evals, int n_particles);

}

extern "C" {

    void single_precision(float* h_eval_pos, float* h_part_pos, float* h_output, int n_evals, int n_particles);

}

extern "C" {

    void half_precision(half2* h_eval_pos,half2* h_part_pos,half2* h_output,int n_evals,int n_particles);

}

extern "C" {

    void half_precision_phis(half2* h_eval_pos, half2* h_part_pos, half2* h_masses, half2* G, half2* eps, half2* h_output,int n_evals,int n_particles);

}