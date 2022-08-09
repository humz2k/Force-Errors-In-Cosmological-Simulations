#include "kernels.h"

__global__
void single_distances(float* eval_pos, float* part_pos, float* output, int n_particles, int n_evals){    
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n_evals){
    
        float eval_posx = eval_pos[i*3];
        float eval_posy = eval_pos[i*3 + 1];
        float eval_posz = eval_pos[i*3 + 2];

        float part_posx;
        float part_posy;
        float part_posz;

        float diffx;
        float diffy;
        float diffz;

        float dist;

        for (int j = 0; j < n_particles; j++){

            part_posx = part_pos[j*3];
            part_posy = part_pos[j*3 + 1];
            part_posz = part_pos[j*3 + 2];

            diffx = eval_posx - part_posx;
            diffy = eval_posy - part_posy;
            diffz = eval_posz - part_posz;

            dist = sqrt((diffx*diffx) + (diffy*diffy) + (diffz*diffz));

            output[i * n_particles + j] = dist;

        }

    }

}