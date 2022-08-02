#include <stdio.h>
#include <stdlib.h>
#include "static_solver.h"
#include "kernels/kernels.h"
#include <cuda_fp16.h>

#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

using namespace std;

unsigned long long CPUTimer(unsigned long long start=0){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

extern "C" {
    void half_precision(half* h_eval_pos, half* h_part_pos, half* h_mass, float* h_output, int n_evals, int n_particles, float G, float eps, unsigned long long* timer){

        cudaFree(0);
        cudaDeviceSynchronize();

        unsigned long long start,end;

        int blockSize = 128;
        int numBlocks = (n_evals + blockSize - 1) / blockSize;

        half *d_eval_pos, *d_part_pos, *d_mass;
        float *d_output;

        cudaMalloc(&d_part_pos,n_particles * 3 * sizeof(half));
        cudaMalloc(&d_mass,n_particles * sizeof(half));

        cudaMalloc(&d_eval_pos,n_evals * 3 * sizeof(half));
        cudaMalloc(&d_output,n_evals * sizeof(float));

        cudaMemcpy(d_part_pos,h_part_pos,n_particles * 3 * sizeof(half),cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,h_mass,n_particles * sizeof(half),cudaMemcpyHostToDevice);

        cudaMemcpy(d_eval_pos,h_eval_pos,n_evals * 3 * sizeof(half),cudaMemcpyHostToDevice);

        start = CPUTimer();
        half_force_solver<<<numBlocks,blockSize>>>(d_eval_pos,d_part_pos,d_mass,d_output,G,eps,n_particles,n_evals);
        cudaDeviceSynchronize();
        end = CPUTimer();

        *timer = end-start;

        cudaMemcpy(h_output,d_output,n_evals * sizeof(float),cudaMemcpyDeviceToHost);

        cudaFree(d_part_pos);
        cudaFree(d_mass);
        cudaFree(d_output);
        cudaFree(d_eval_pos);
    }
}

extern "C" {
    void single_precision(float* h_eval_pos, float* h_part_pos, float* h_mass, float* h_output, int n_evals, int n_particles, float G, float eps, unsigned long long* timer){

        cudaFree(0);
        cudaDeviceSynchronize();

        unsigned long long start,end;

        int blockSize = 128;
        int numBlocks = (n_evals + blockSize - 1) / blockSize;

        float *d_eval_pos, *d_part_pos, *d_mass, *d_output;

        cudaMalloc(&d_part_pos,n_particles * 3 * sizeof(float));
        cudaMalloc(&d_mass,n_particles * sizeof(float));

        cudaMalloc(&d_eval_pos,n_evals * 3 * sizeof(float));
        cudaMalloc(&d_output,n_evals * sizeof(float));

        cudaMemcpy(d_part_pos,h_part_pos,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,h_mass,n_particles * sizeof(float),cudaMemcpyHostToDevice);

        cudaMemcpy(d_eval_pos,h_eval_pos,n_evals * 3 * sizeof(float),cudaMemcpyHostToDevice);

        start = CPUTimer();
        single_force_solver<<<numBlocks,blockSize>>>(d_eval_pos,d_part_pos,d_mass,d_output,G,eps,n_particles,n_evals);
        cudaDeviceSynchronize();
        end = CPUTimer();

        *timer = end-start;

        cudaMemcpy(h_output,d_output,n_evals * sizeof(float),cudaMemcpyDeviceToHost);

        cudaFree(d_part_pos);
        cudaFree(d_mass);
        cudaFree(d_output);
        cudaFree(d_eval_pos);
    }
}

extern "C" {
    void double_precision(double* h_eval_pos, double* h_part_pos, double* h_mass, double* h_output, int n_evals, int n_particles, double G, double eps, unsigned long long* timer){

        cudaFree(0);
        cudaDeviceSynchronize();

        unsigned long long start,end;

        int blockSize = 128;
        int numBlocks = (n_evals + blockSize - 1) / blockSize;

        double *d_eval_pos, *d_part_pos, *d_mass, *d_output;

        cudaMalloc(&d_part_pos,n_particles * 3 * sizeof(double));
        cudaMalloc(&d_mass,n_particles * sizeof(double));

        cudaMalloc(&d_eval_pos,n_evals * 3 * sizeof(double));
        cudaMalloc(&d_output,n_evals * sizeof(double));

        cudaMemcpy(d_part_pos,h_part_pos,n_particles * 3 * sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,h_mass,n_particles * sizeof(double),cudaMemcpyHostToDevice);

        cudaMemcpy(d_eval_pos,h_eval_pos,n_evals * 3 * sizeof(double),cudaMemcpyHostToDevice);

        start = CPUTimer();
        double_force_solver<<<numBlocks,blockSize>>>(d_eval_pos,d_part_pos,d_mass,d_output,G,eps,n_particles,n_evals);
        cudaDeviceSynchronize();
        end = CPUTimer();

        *timer = end-start;

        cudaMemcpy(h_output,d_output,n_evals * sizeof(double),cudaMemcpyDeviceToHost);

        cudaFree(d_part_pos);
        cudaFree(d_mass);
        cudaFree(d_output);
        cudaFree(d_eval_pos);
    }
}

extern "C" {
    void single_precision_shared_mem_cuda(float* h_eval_pos, float* h_part_pos, float* h_mass, float* h_output, int n_evals, int n_particles, float G, float eps, unsigned long long* timer){

        cudaFree(0);
        cudaDeviceSynchronize();

        unsigned long long start,end;

        int blockSize = 128;
        int numBlocks = (n_evals + blockSize - 1) / blockSize;

        size_t shared_mem_size = blockSize * 4 * sizeof(float);

        float *d_eval_pos, *d_part_pos, *d_mass, *d_output;

        cudaMalloc(&d_part_pos,n_particles * 3 * sizeof(float));
        cudaMalloc(&d_mass,n_particles * sizeof(float));

        cudaMalloc(&d_eval_pos,n_evals * 3 * sizeof(float));
        cudaMalloc(&d_output,n_evals * sizeof(float));

        cudaMemcpy(d_part_pos,h_part_pos,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,h_mass,n_particles * sizeof(float),cudaMemcpyHostToDevice);

        cudaMemcpy(d_eval_pos,h_eval_pos,n_evals * 3 * sizeof(float),cudaMemcpyHostToDevice);

        start = CPUTimer();
        single_force_solver_shared_mem_cuda<<<numBlocks,blockSize,shared_mem_size>>>(d_eval_pos,d_part_pos,d_mass,d_output,G,eps,n_particles,n_evals);
        cudaDeviceSynchronize();
        end = CPUTimer();

        *timer = end-start;

        cudaMemcpy(h_output,d_output,n_evals * sizeof(float),cudaMemcpyDeviceToHost);

        cudaFree(d_part_pos);
        cudaFree(d_mass);
        cudaFree(d_output);
        cudaFree(d_eval_pos);
    }
}

extern "C" {
    void half_precision_shared_mem_cuda(half* h_eval_pos, half* h_part_pos, half* h_mass, float* h_output, int n_evals, int n_particles, float G, float eps, unsigned long long* timer){

        cudaFree(0);
        cudaDeviceSynchronize();

        unsigned long long start,end;

        int blockSize = 128;
        int numBlocks = (n_evals + blockSize - 1) / blockSize;

        size_t shared_mem_size = blockSize * 2 * sizeof(half2);

        half *d_eval_pos, *d_part_pos, *d_mass;
        float *d_output;

        cudaMalloc(&d_part_pos,n_particles * 3 * sizeof(half));
        cudaMalloc(&d_mass,n_particles * sizeof(half));

        cudaMalloc(&d_eval_pos,n_evals * 3 * sizeof(half));
        cudaMalloc(&d_output,n_evals * sizeof(float));

        cudaMemcpy(d_part_pos,h_part_pos,n_particles * 3 * sizeof(half),cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,h_mass,n_particles * sizeof(half),cudaMemcpyHostToDevice);

        cudaMemcpy(d_eval_pos,h_eval_pos,n_evals * 3 * sizeof(half),cudaMemcpyHostToDevice);

        start = CPUTimer();
        half_force_solver_shared_mem_cuda<<<numBlocks,blockSize,shared_mem_size>>>(d_eval_pos,d_part_pos,d_mass,d_output,G,eps,n_particles,n_evals);
        cudaDeviceSynchronize();
        end = CPUTimer();

        *timer = end-start;

        cudaMemcpy(h_output,d_output,n_evals * sizeof(float),cudaMemcpyDeviceToHost);

        cudaFree(d_part_pos);
        cudaFree(d_mass);
        cudaFree(d_output);
        cudaFree(d_eval_pos);
    }
}