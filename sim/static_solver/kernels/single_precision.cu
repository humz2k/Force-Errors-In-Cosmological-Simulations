#include "kernels.h"

__global__
void single_force_solver(float* eval_pos, float* part_pos, float* mass, float* output, float G, float eps, int n_particles, int n_evals){    
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n_evals){
    
        float eval_posx = eval_pos[i*3];
        float eval_posy = eval_pos[i*3 + 1];
        float eval_posz = eval_pos[i*3 + 2];

        float part_posx;
        float part_posy;
        float part_posz;
        float part_mass;

        float diffx;
        float diffy;
        float diffz;

        float dist;

        float temp0;

        float gpe = 0;

        for (int j = 0; j < n_particles; j++){

            part_posx = part_pos[j*3];
            part_posy = part_pos[j*3 + 1];
            part_posz = part_pos[j*3 + 2];

            if ((part_posx != eval_posx) || (part_posy != eval_posy) || (part_posz != eval_posz)){

                part_mass = mass[j];

                diffx = eval_posx - part_posx;
                diffy = eval_posy - part_posy;
                diffz = eval_posz - part_posz;

                dist = sqrt((diffx*diffx) + (diffy*diffy) + (diffz*diffz) + eps);

                temp0 = (part_mass) / dist;

                gpe = gpe + temp0;

            }

        }

        output[i] = (-1) * G * gpe;
    }

}