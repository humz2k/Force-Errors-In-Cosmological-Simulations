#include "kernels.h"
#include <cuda_fp16.h>

__global__
void half_force_solver(half* eval_pos, half* part_pos, half* mass, float* output, float G, float eps, int n_particles, int n_evals){    
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n_evals){
        
        __half heps = __float2half(eps);
        __half hG = __float2half(G);

        __half eval_posx = eval_pos[i*3];
        __half eval_posy = eval_pos[i*3 + 1];
        __half eval_posz = eval_pos[i*3 + 2];

        __half part_posx;
        __half part_posy;
        __half part_posz;
        __half part_mass;

        __half diffx;
        __half diffy;
        __half diffz;

        __half temp0;
        __half temp1;

        __half dist;

        float gpe = 0;

        for (int j = 0; j < n_particles; j++){

            part_posx = part_pos[j*3 ];
            part_posy = part_pos[j*3 + 1];
            part_posz = part_pos[j*3 + 2];

            if (__hne(part_posx,eval_posx) || __hne(part_posy,eval_posy) || __hne(part_posz,eval_posz)){

                part_mass = __float2half(mass[j]);

                diffx = __hsub(eval_posx,part_posx);
                diffy = __hsub(eval_posy,part_posy);
                diffz = __hsub(eval_posz,part_posz);

                temp0 = __hfma(diffx,diffx,heps);
                temp1 = __hfma(diffy,diffy,temp0);
                temp0 = __hfma(diffz,diffz,temp1);

                dist = hsqrt(temp0);

                temp1 = __hdiv(__hmul(hG,part_mass),dist);
                gpe = gpe + __half2float(temp1);
            }

        }

        output[i] = (-1) * gpe;
    }

}