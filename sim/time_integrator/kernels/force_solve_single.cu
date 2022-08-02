#include "kernels.h"

__global__
void force_solve_single(float* pos, float* mass, float* acc_phi, float G, float eps, int n_particles){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;
    
    float pos_ix = pos[i*3];
    float pos_iy = pos[i*3 + 1];
    float pos_iz = pos[i*3 + 2];

    float diffx;
    float diffy;
    float diffz;

    float pos_jx;
    float pos_jy;
    float pos_jz;

    float ax = 0;
    float ay = 0;
    float az = 0;
    float gpe = 0;

    float mass_j = 0;

    float dist;
    float dist_sqr;
    float acc_mul;

    if (i < n_particles){

        for (j = 0; j < n_particles; j++){

            if (j != i) {

                pos_jx = pos[j*3];
                pos_jy = pos[j*3 + 1];
                pos_jz = pos[j*3 + 2];

                mass_j = mass[j];
                
                diffx = pos_jx - pos_ix;
                diffy = pos_jy - pos_iy;
                diffz = pos_jz - pos_iz;

                dist_sqr = (diffx*diffx) + (diffy*diffy) + (diffz*diffz) + eps;
                dist = sqrt(dist_sqr);

                acc_mul = mass_j / (dist_sqr * dist);
                ax = ax + diffx * acc_mul;
                ay = ay + diffy * acc_mul;
                az = az + diffz * acc_mul;

                gpe = gpe + mass_j / dist;
            }

        }
        
        acc_phi[i*4] = ax * G;
        acc_phi[i*4 + 1] = ay * G;
        acc_phi[i*4 + 2] = az * G;
        acc_phi[i*4 + 3] = (-1) * G * mass[i] * gpe;
    }
    
}