#include "kernels.h"

__global__
void single_force_solver_shared_mem_cuda(float* eval_pos, float* part_pos, float* mass, float* output, float G, float eps, int n_particles, int n_evals){    
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ float s[];

    int n_repeats = (n_particles + blockDim.x - 1) / blockDim.x;

    float phi = 0;

    float eval_posx;
    float eval_posy;
    float eval_posz;

    if (i < n_evals){

        eval_posx = eval_pos[i * 3];
        eval_posy = eval_pos[i * 3 + 1];
        eval_posz = eval_pos[i * 3 + 2];
        

    } else{

        eval_posx = 0;
        eval_posy = 0;
        eval_posz = 0;

    }

    float part_posx;
    float part_posy;
    float part_posz;
    float part_mass;

    float diffx;
    float diffy;
    float diffz;

    float dist;
    
    float temp0;

    for (int j = 0; j < n_repeats; j++){

        int offset = blockDim.x * j;
        int myIdx = offset + threadIdx.x;

        if (myIdx < n_particles){
            s[threadIdx.x * 4] = part_pos[myIdx * 3];
            s[threadIdx.x * 4 + 1] = part_pos[myIdx * 3 + 1];
            s[threadIdx.x * 4 + 2] = part_pos[myIdx * 3 + 2];
            s[threadIdx.x * 4 + 3] = mass[myIdx];
        } else {
            s[threadIdx.x * 4 + 3] = 0;
        }


        for (int k = 0; k < blockDim.x; k++){
            
            part_posx = s[k * 4];
            part_posy = s[k * 4 + 1];
            part_posz = s[k * 4 + 2];
            part_mass = s[k * 4 + 3];

            if (part_mass != 0){

                if ((part_posx != eval_posx) || (part_posy != eval_posy) || (part_posy != eval_posy)){

                    diffx = eval_posx - part_posx;
                    diffy = eval_posy - part_posy;
                    diffz = eval_posz - part_posz;

                    dist = sqrt((diffx*diffx) + (diffy*diffy) + (diffz*diffz) + eps);

                    temp0 = (part_mass) / dist;

                    phi = phi + temp0;

                }

            }

        }


    }

    if (i < n_evals){

        output[i] = (-1) * G * phi;

    }

}