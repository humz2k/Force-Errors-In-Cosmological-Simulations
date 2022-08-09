#include <cuda_fp16.h>

extern "C" {

    void double_precision(double* h_eval_pos, double* h_part_pos, double* h_output, int n_evals, int n_particles);

}

extern "C" {

    void single_precision(float* h_eval_pos, float* h_part_pos, float* h_output, int n_evals, int n_particles);

}

extern "C" {

    void half_precision(half2* h_eval_pos,half2* h_part_pos,half2* h_output,int n_evals,int n_particles);

}

extern "C" {

    void half_precision_phis(half2* h_eval_pos, half2* h_part_pos, half2* h_masses, half2* G, half2* eps, half2* h_output,int n_evals,int n_particles){

        half2 *d_eval_pos,*d_part_pos,*d_masses,*d_output,*d_G,*d_eps;
        //half *d_eval_pos_half,*d_part_pos_half;

        int blockSize = 128;
        int numBlocks = ((n_evals/2) + blockSize - 1) / blockSize;

        cudaMalloc(&d_part_pos,(n_particles/2) * 3 * sizeof(half2));
        cudaMalloc(&d_eval_pos,(n_evals/2) * 3 * sizeof(half2));
        cudaMalloc(&d_masses,(n_particles/2) * sizeof(half2));

        cudaMalloc(&d_G,sizeof(half2));
        cudaMalloc(&d_eps,sizeof(half2));

        cudaMalloc(&d_output,((n_particles * n_evals)/2) * sizeof(half2));

        cudaMemcpy(d_G,G,sizeof(half2),cudaMemcpyHostToDevice);
        cudaMemcpy(d_eps,eps,sizeof(half2),cudaMemcpyHostToDevice);

        cudaMemcpy(d_part_pos,h_part_pos,(n_particles/2) * 3 * sizeof(half2),cudaMemcpyHostToDevice);
        cudaMemcpy(d_eval_pos,h_eval_pos,(n_evals/2) * 3 * sizeof(half2),cudaMemcpyHostToDevice);
        cudaMemcpy(d_masses,h_masses,(n_particles/2) * sizeof(half2),cudaMemcpyHostToDevice);

        //half_phis<<<numBlocks,blockSize>>>(d_eval_pos,d_part_pos,d_masses,G[0],eps[0],d_output,n_particles,n_evals);

        cudaMemcpy(h_output,d_output,((n_particles * n_evals)/2) * sizeof(half2), cudaMemcpyDeviceToHost);

        cudaFree(d_part_pos);
        cudaFree(d_eval_pos);
        cudaFree(d_output);
        cudaFree(d_masses);
        cudaFree(d_G);
        cudaFree(d_eps);

    }

}