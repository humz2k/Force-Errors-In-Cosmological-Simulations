#include "kernels.h"

__global__
void double_force_solver(double* eval_pos, double* part_pos, double* mass, double* output, double G, double eps, int n_particles, int n_evals){    
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n_evals){
    
        double eval_posx = eval_pos[i*3];
        double eval_posy = eval_pos[i*3 + 1];
        double eval_posz = eval_pos[i*3 + 2];

        double part_posx;
        double part_posy;
        double part_posz;
        double part_mass;

        double diffx;
        double diffy;
        double diffz;

        double dist;

        double temp0;

        double gpe = 0;

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