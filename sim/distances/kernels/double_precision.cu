#include "kernels.h"

__global__
void double_distances(double* eval_pos, double* part_pos, double* output, int n_particles, int n_evals){    
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n_evals){
    
        double eval_posx = eval_pos[i*3];
        double eval_posy = eval_pos[i*3 + 1];
        double eval_posz = eval_pos[i*3 + 2];

        double part_posx;
        double part_posy;
        double part_posz;

        double diffx;
        double diffy;
        double diffz;

        double dist;

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