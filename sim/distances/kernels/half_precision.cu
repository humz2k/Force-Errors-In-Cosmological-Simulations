#include <cuda_fp16.h>
#include "kernels.h"

__global__
void half_distances(half2* eval_pos, half2* part_pos, half2* output, int n_particles, int n_evals){    
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < (n_evals/2)){
    
        __half2 eval_posx = eval_pos[i*3];
        __half2 eval_posy = eval_pos[i*3 + 1];
        __half2 eval_posz = eval_pos[i*3 + 2];

        __half2 part_posx;
        __half2 part_posy;
        __half2 part_posz;

        __half2 diffx;
        __half2 diffy;
        __half2 diffz;

        __half2 dist;

        for (int j = 0; j < (n_particles/2); j++){

            part_posx = part_pos[j*3];
            part_posy = part_pos[j*3 + 1];
            part_posz = part_pos[j*3 + 2];

            diffx = __hsub2(eval_posx,part_posx);
            diffy = __hsub2(eval_posy,part_posy);
            diffz = __hsub2(eval_posz,part_posz);
            
            dist = h2sqrt(__hfma2(diffz,diffz,__hfma2(diffy,diffy,__hmul2(diffx,diffx))));

            output[i * (n_particles) + (j*2)] = dist;

            part_posx = __lowhigh2highlow(part_posx);
            part_posy = __lowhigh2highlow(part_posy);
            part_posz = __lowhigh2highlow(part_posz);

            diffx = __hsub2(eval_posx,part_posx);
            diffy = __hsub2(eval_posy,part_posy);
            diffz = __hsub2(eval_posz,part_posz);

            dist = h2sqrt(__hfma2(diffz,diffz,__hfma2(diffy,diffy,__hmul2(diffx,diffx))));

            output[i * (n_particles) + (j*2+1)] = dist;

        }
    }

}

__global__
void half_phis(half2* eval_pos, half2* part_pos, half2* masses, half2 G, half2 eps, half2* output, int n_particles, int n_evals){    
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < (n_evals/2)){
    
        __half2 eval_posx = eval_pos[i*3];
        __half2 eval_posy = eval_pos[i*3 + 1];
        __half2 eval_posz = eval_pos[i*3 + 2];

        __half2 part_posx;
        __half2 part_posy;
        __half2 part_posz;

        __half2 diffx;
        __half2 diffy;
        __half2 diffz;

        __half2 dist;
        __half2 mass;

        //__half2 G = d_G[0];
        //__half2 eps = d_eps[0];

        for (int j = 0; j < (n_particles/2); j++){

            part_posx = part_pos[j*3];
            part_posy = part_pos[j*3 + 1];
            part_posz = part_pos[j*3 + 2];

            mass = masses[j];

            diffx = __hsub2(eval_posx,part_posx);
            diffy = __hsub2(eval_posy,part_posy);
            diffz = __hsub2(eval_posz,part_posz);
            
            dist = h2sqrt(__hfma2(diffz,diffz,__hfma2(diffy,diffy,__hfma2(diffx,diffx,eps))));

            output[i * n_particles + (j*2)] = __h2div(__hmul2(G,mass),dist);

            part_posx = __lowhigh2highlow(part_posx);
            part_posy = __lowhigh2highlow(part_posy);
            part_posz = __lowhigh2highlow(part_posz);
            mass = __lowhigh2highlow(mass);

            diffx = __hsub2(eval_posx,part_posx);
            diffy = __hsub2(eval_posy,part_posy);
            diffz = __hsub2(eval_posz,part_posz);

            dist = h2sqrt(__hfma2(diffz,diffz,__hfma2(diffy,diffy,__hmul2(diffx,diffx))));

            output[i * n_particles + (j*2+1)] = __h2div(__hmul2(G,mass),dist);

        }
    }

}