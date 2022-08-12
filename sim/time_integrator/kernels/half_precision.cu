#include <cuda_fp16.h>
#include "kernels.h"

__global__
void float_array_2_half2_array(float* s, half2* d){    
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int s_index = i*2;
    d[i*3] = __floats2half2_rn(s[s_index*3],s[(s_index+1)*3]);
    d[i*3 + 1] = __floats2half2_rn(s[s_index*3 + 1],s[(s_index+1)*3 + 1]);
    d[i*3 + 2] = __floats2half2_rn(s[s_index*3 + 2],s[(s_index+1)*3 + 2]);

}

__global__
void half_force_solver(half2* part_pos, float mass, half2 eps, float G, float* output, int n_particles){    
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ __half2 s[];

    int n_repeats = ((n_particles/2) + blockDim.x - 1) / blockDim.x;

    float2 out_phi = make_float2(0,0);
    float2 out_ax = make_float2(0,0);
    float2 out_ay = make_float2(0,0);
    float2 out_az = make_float2(0,0);

    __half2 eval_posx = part_pos[i * 3];
    __half2 eval_posy = part_pos[i * 3 + 1];
    __half2 eval_posz = part_pos[i * 3 + 2];

    __half2 part_posx;
    __half2 part_posy;
    __half2 part_posz;

    __half2 diffx;
    __half2 diffy;
    __half2 diffz;

    float2 phi;
    __half2 acc_mul;

    __half2 dist;
    __half2 dist_cube;

    float2 ax;
    float2 ay;
    float2 az;

    float temp2;
    float temp1;

    __half2 temp;

    for (int j = 0; j < n_repeats; j++){

        int offset = blockDim.x * j;
        int myIdx = offset + threadIdx.x;

        s[threadIdx.x * 3] = part_pos[myIdx * 3];
        s[threadIdx.x * 3 + 1] = part_pos[myIdx * 3 + 1];
        s[threadIdx.x * 3 + 2] = part_pos[myIdx * 3 + 2];

        __syncthreads();

        for (int k = 0; k < blockDim.x; k++){
            
            part_posx = s[k * 3];
            part_posy = s[k * 3 + 1];
            part_posz = s[k * 3 + 2];

                if ((__hbne2(part_posx,eval_posx)) || (__hbne2(part_posy,eval_posy)) || (__hbne2(part_posz,eval_posz))){

                    diffx = __hsub2(eval_posx,part_posx);
                    diffy = __hsub2(eval_posy,part_posy);
                    diffz = __hsub2(eval_posz,part_posz);

                    dist_cube = __hfma2(diffz,diffz,__hfma2(diffy,diffy,__hfma2(diffx,diffx,eps)));

                    temp = h2sqrt(dist_cube);

                    acc_mul = h2rcp(__hmul2(temp,dist_cube));

                    phi = __half22float2(h2rcp(temp));

                    ax = __half22float2(__hmul2(acc_mul,diffx));
                    ay = __half22float2(__hmul2(acc_mul,diffy));
                    az = __half22float2(__hmul2(acc_mul,diffz));

                    if (!(isnan(ax.x) || isnan(ay.x) || isnan(az.x))){
                        out_phi.x += phi.x;
                        out_ax.x += ax.x;
                        out_ay.x += ay.x;
                        out_az.x += az.x;

                        if (!(isnan(ax.y) || isnan(ay.y) || isnan(az.y))){
                            out_phi.y += phi.y;
                            out_ax.y += ax.y;
                            out_ay.y += ay.y;
                            out_az.y += az.y;
                        }
                    }


                }

                part_posx = __lowhigh2highlow(part_posx);
                part_posy = __lowhigh2highlow(part_posy);
                part_posz = __lowhigh2highlow(part_posz);

                diffx = __hsub2(eval_posx,part_posx);
                diffy = __hsub2(eval_posy,part_posy);
                diffz = __hsub2(eval_posz,part_posz);

                dist_cube = __hfma2(diffz,diffz,__hfma2(diffy,diffy,__hfma2(diffx,diffx,eps)));

                temp = h2sqrt(dist_cube);

                acc_mul = h2rcp(__hmul2(temp,dist_cube));

                phi = __half22float2(h2rcp(temp));

                ax = __half22float2(__hmul2(acc_mul,diffx));
                ay = __half22float2(__hmul2(acc_mul,diffy));
                az = __half22float2(__hmul2(acc_mul,diffz));

                if (!(isnan(ax.x) || isnan(ay.x) || isnan(az.x))){
                    out_phi.x += phi.x;
                    out_ax.x += ax.x;
                    out_ay.x += ay.x;
                    out_az.x += az.x;

                    if (!(isnan(ax.y) || isnan(ay.y) || isnan(az.y))){
                        out_phi.y += phi.y;
                        out_ax.y += ax.y;
                        out_ay.y += ay.y;
                        out_az.y += az.y;
                    }
                }
            
            

            }
        }

    temp2 = G * mass * mass;
    temp1 = (-1) * G * mass;

    output[(i*2)*4] = out_ax.x * temp2;
    output[(i*2)*4 + 1] = out_ay.x * temp2;
    output[(i*2)*4 + 2] = out_az.x * temp2;
    output[(i*2)*4 + 3] = out_phi.x * temp1;

    output[(i*2 + 1)*4] = out_ax.y * temp2;
    output[(i*2 + 1)*4 + 1] = out_ay.y * temp2;
    output[(i*2 + 1)*4 + 2] = out_az.y * temp2;
    output[(i*2 + 1)*4 + 3] = out_phi.y * temp1;

}