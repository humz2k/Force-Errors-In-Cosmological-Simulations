#include "kernels.h"
#include <cuda_fp16.h>

__global__
void half_force_solver_shared_mem_cuda(half* eval_pos, half* part_pos, half* mass, float* output, float G, float eps, int n_particles, int n_evals){    
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ __half2 s[];

    int n_repeats = (n_particles + blockDim.x - 1) / blockDim.x;

    __half2 eval_posxy;
    __half eval_posz;

    __half2 part_posxy;
    __half part_posz;
    __half part_mass;

    __half2 diffxy;
    __half diffz;

    __half2 temp;

    __half temp1;
    __half temp2;
    __half temp3;

    float gpe = 0;

    if (i < n_evals){

        eval_posxy = __halves2half2(eval_pos[i * 3], eval_pos[i * 3 + 1]);
        eval_posz = eval_pos[i * 3 + 2];

    }

    for (int j = 0; j < n_repeats; j++){

        int offset = blockDim.x * j;
        int myIdx = offset + threadIdx.x;

        s[threadIdx.x * 2] = __halves2half2(part_pos[myIdx * 3],part_pos[myIdx * 3 + 1]);
        s[threadIdx.x * 2 + 1] = __halves2half2(part_pos[myIdx * 3 + 2],mass[myIdx]);

        for (int k = 0; k < blockDim.x; k++){

            if ((k + offset) < n_particles){

                part_posxy = s[k * 2];
                part_posz = __low2half(s[k * 2 + 1]);
                part_mass = __high2half(s[k * 2 + 1]);

                if (__hne(part_posz,eval_posz) || (!(__hbeq2(part_posxy,eval_posxy)))){

                    diffxy = __hsub2(eval_posxy,part_posxy);
                    diffz = __hsub(eval_posz,part_posz);

                    temp = __hmul2(diffxy,diffxy);
                    temp1 = __hfma(diffz,diffz,eps);
                    temp2 = __hadd(__low2half(temp),__high2half(temp));
                    temp3 = __hadd(temp1,temp2);

                    temp1 = hsqrt(temp3);
                    temp2 = __hdiv(part_mass,temp1);

                    gpe = gpe + __half2float(temp2);
                }

            }
        }

    }

    if (i < n_evals){

        output[i] = (-1) * G * gpe;

    }

}