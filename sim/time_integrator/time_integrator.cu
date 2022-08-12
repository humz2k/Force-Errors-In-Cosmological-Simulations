#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include "time_integrator.h"
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

void save_single(float* pos, float* vel, float* phi_acc, int n_particles, std::ofstream &out){

    for (int i = 0; i < n_particles; i++){

        out.write( reinterpret_cast<char*>( &pos[i*3]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &pos[i*3 + 1]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &pos[i*3 + 2]), sizeof( float ));

        out.write( reinterpret_cast<char*>( &vel[i*3]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &vel[i*3 + 1]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &vel[i*3 + 2]), sizeof( float ));

        out.write( reinterpret_cast<char*>( &phi_acc[i*4]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &phi_acc[i*4 + 1]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &phi_acc[i*4 + 2]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &phi_acc[i*4 + 3]), sizeof( float ));

    }

}

void save_double(double* pos, double* vel, double* phi_acc, int n_particles, std::ofstream &out){

    for (int i = 0; i < n_particles; i++){

        out.write( reinterpret_cast<char*>( &pos[i*3]), sizeof( double ));
        out.write( reinterpret_cast<char*>( &pos[i*3 + 1]), sizeof( double ));
        out.write( reinterpret_cast<char*>( &pos[i*3 + 2]), sizeof( double ));

        out.write( reinterpret_cast<char*>( &vel[i*3]), sizeof( double ));
        out.write( reinterpret_cast<char*>( &vel[i*3 + 1]), sizeof( double ));
        out.write( reinterpret_cast<char*>( &vel[i*3 + 2]), sizeof( double ));

        out.write( reinterpret_cast<char*>( &phi_acc[i*4]), sizeof( double ));
        out.write( reinterpret_cast<char*>( &phi_acc[i*4 + 1]), sizeof( double ));
        out.write( reinterpret_cast<char*>( &phi_acc[i*4 + 2]), sizeof( double ));
        out.write( reinterpret_cast<char*>( &phi_acc[i*4 + 3]), sizeof( double ));

    }

}

extern "C" { 
    void single_precision(float* input_pos, float* input_vel, float* input_mass, int n_particles, int steps, float G, float eps, float dt, unsigned long long* timer){

        cudaFree(0);

        std::ofstream out;
        out.open( "out.dat", std::ios::out | std::ios::binary);
        std::ofstream &fp = out;

        int blockSize = 128;

        int numBlocks = (n_particles + blockSize - 1) / blockSize;

        float *h_pos = (float*) malloc(n_particles * 3 * sizeof(float));
        float *h_acc_phi = (float*) malloc(n_particles * 4 * sizeof(float));
        float *h_vel = (float*) malloc(n_particles * 3 * sizeof(float));

        float *d_pos,*d_acc_phi,*d_vel,*d_mass;

        cudaMalloc(&d_pos,n_particles * 3 * sizeof(float));
        cudaMalloc(&d_acc_phi,n_particles * 4 * sizeof(float));
        cudaMalloc(&d_vel,n_particles * 3 * sizeof(float));
        cudaMalloc(&d_mass,n_particles * sizeof(float));

        cudaMemcpy(d_pos,input_pos,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel,input_vel,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,input_mass,n_particles * sizeof(float),cudaMemcpyHostToDevice);

        force_solve_single<<<numBlocks,blockSize>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);

        cudaDeviceSynchronize();

        cudaMemcpy(h_pos,d_pos,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vel,d_vel,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * 4 * sizeof(float),cudaMemcpyDeviceToHost);


        for (int step = 0; step < steps; step++){

            fast_add_4to3_single<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5 * dt);
            fast_add_3to3_single<<<numBlocks,blockSize>>>(d_vel,d_pos,1 * dt);

            force_solve_single<<<numBlocks,blockSize>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);

            fast_add_4to3_single<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5 * dt);

            save_single(h_pos,h_vel,h_acc_phi,n_particles,fp);

            cudaDeviceSynchronize();

            cudaMemcpy(h_pos,d_pos,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vel,d_vel,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * 4 * sizeof(float),cudaMemcpyDeviceToHost);

        }

        save_single(h_pos,h_vel,h_acc_phi,n_particles,fp);

        cudaFree(d_pos);
        cudaFree(d_vel);
        cudaFree(d_acc_phi);
        cudaFree(d_mass);

        free(h_pos);
        free(h_vel);
        free(h_acc_phi);
    }
}

extern "C" { 
    void half_precision(float* input_pos, float* input_vel, float particle_mass, int n_particles, int steps, float G, half2* eps_array, float dt, unsigned long long* timer){

        cudaFree(0);

        std::ofstream out;
        out.open( "out.dat", std::ios::out | std::ios::binary);
        std::ofstream &fp = out;

        int blockSize = 128;

        int numBlocks = ((n_particles/2) + blockSize - 1) / blockSize;

        float *h_pos = (float*) malloc(n_particles * 3 * sizeof(float));
        float *h_acc_phi = (float*) malloc(n_particles * 4 * sizeof(float));
        float *h_vel = (float*) malloc(n_particles * 3 * sizeof(float));

        float *d_pos,*d_acc_phi,*d_vel;

        cudaMalloc(&d_pos,n_particles * 3 * sizeof(float));
        cudaMalloc(&d_acc_phi,n_particles * 4 * sizeof(float));
        cudaMalloc(&d_vel,n_particles * 3 * sizeof(float));

        half2 *d_pos_h2;
        cudaMalloc(&d_pos_h2,(n_particles/2) * 3 * sizeof(half2));

        cudaMemcpy(d_pos,input_pos,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel,input_vel,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);

        float_array_2_half2_array<<<numBlocks,blockSize>>>(d_pos,d_pos_h2);

        half_force_solver<<<numBlocks,blockSize,(blockSize) * 3 * sizeof(half2)>>>(d_pos_h2, particle_mass, eps_array[0], G, d_acc_phi, n_particles);

        //cudaDeviceSynchronize();

        cudaMemcpy(h_pos,d_pos,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
        //cudaMemcpy(h_vel,d_vel,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * 4 * sizeof(float),cudaMemcpyDeviceToHost);

        save_single(h_pos,input_vel,h_acc_phi,n_particles,fp);

        cudaFree(d_pos);
        cudaFree(d_vel);
        cudaFree(d_acc_phi);

        free(h_pos);
        free(h_vel);
        free(h_acc_phi);
    }
}