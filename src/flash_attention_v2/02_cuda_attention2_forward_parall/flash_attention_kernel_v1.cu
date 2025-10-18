#include "../../util/cuda_shim.h"
#include <stdio.h>
#include <math.h>
#include <cfloat>

__device__ int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

#define WARP_SIZE 2  // For testing with small N


// Kernel for Flash Attention algorithm
__global__ void flash_attention_kernel(
    const float* Q,  // Query matrix (N x d)
    const float* K,  // Key matrix (N x d)
    const float* V,  // Value matrix (N x d)
    float* O,        // Output matrix (N x d)
    float* L,        // Loss vector (N)
    int N,           // Sequence length
    int d,           // Embedding dimension
    int Br,          // Block rows
    int Bc,          // Block cols
    float scale      // Softmax scale (1/sqrt(d) typically)
) {
    /*
    size_t shared_mem_size = sizeof(float) * (
        Br * d +      // Qi
        Bc * d +      // Kj
        Bc * d +      // Vj
        Br * Bc +     // S
        Br * Bc +     // P
        Br * d +      // Oi 
        Br * 1 +      // li 
        Br * 1 +      // mi 
        Br * 1 +      // li_pre 
        Br * 1        // mi_pre 

    );
    */
    // Shared memory for tiles
    extern __shared__ float shared_mem[];
    int mem_offset = 0;
    float* Qi = shared_mem;                   // Br x d
    float* Kj = &shared_mem[mem_offset += Br * d]; // Bc x d
    float* Vj = &shared_mem[mem_offset += Bc * d]; // Bc x d
    float* S = &shared_mem[mem_offset += Bc * d]; // Br x Bc
    float* P = &shared_mem[mem_offset += Br * Bc]; // Br x Bc
    float* Oi = &shared_mem[mem_offset += Br * Bc]; // Br x d
    float* li = &shared_mem[mem_offset += Br * d]; // Br * 1
    float* mi = &shared_mem[mem_offset += Br * 1]; // Br * 1
    float* li_pre = &shared_mem[mem_offset += Br * 1]; // Br * 1
    float* mi_pre = &shared_mem[mem_offset += Br * 1]; // Br * 1

    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int tnum = blockDim.x; // Number of threads per block (assuming 1D for simplicity)

//    int Tr = (N + Br - 1) / Br; // Total number of row tiles
    int Tc = (N + Bc - 1) / Bc; // Total number of col tiles

    //line 4: load Qi , init Oi
    int i_start = bid * Br;
    int i_end = i_start + Br < N ? i_start + Br : N;
    int tile_size_i = i_end - i_start; // Actual size of the tile (may be smaller at edges)
    int total_Qi = tile_size_i * d;

    
    for(int idx = tid; idx < total_Qi; idx += tnum ) {
        int local_row = idx / d;        // Which row in the tile
        int col = idx % d;              // Which column 
        int global_row = i_start + local_row;  // Global row index
        Qi[idx] = Q[global_row * d + col];     // Correct 2D indexing
        Oi[idx] = 0.0f;                      // Initialize Oi to zero
    }

    // Initialize mi to -inf and li to 0
    for ( int k = tid ; k < Br ; k += tnum ) {
        mi[k] = -FLT_MAX; // Initialize mi to -inf
        li[k] = 0.0f;     // Initialize li to 0
    }

    __syncthreads(); // Ensure all threads have loaded Qi before proceeding
    //line 6 
    for(int tile = 0; tile < Tc; tile++) { // Loop over tiles of K,V
        // each thread load one row of K,V
        // Algorithm Line 10: Load Kj, Vj from HBM to on-chip SRAM
        int j_start = tile * Bc;
        int j_end = j_start + Bc < N ? j_start + Bc : N;
        int tile_size_j = j_end - j_start; // Actual size of the tile (may be smaller at edges)

        for(int idx = tid; idx < tile_size_j * d; idx += tnum) {
            int local_row = idx / d;        // Which row in the tile
            int col = idx % d;              // Which column 
            int global_row = j_start + local_row;  // Global row index
            Kj[idx] = K[global_row * d + col];     // Correct 2D indexing
            Vj[idx] = V[global_row * d + col];     // Correct 2D indexing
        }
        __syncthreads(); // Ensure all threads have loaded Kj, Vj before proceeding

        // line 8 compute Sij = QiKj^T / sqrt(d). mi    size S is Br x Bc
        // one thread handles one element of Sij
        for (int step = 0; step < div_ceil(tile_size_i * tile_size_j, tnum); step++) {
            int index = step * tnum + tid;
            if (index >= tile_size_i * tile_size_j) {
                continue;
            }
            int row = index / tile_size_j; // Row index in S (0 to Br-1)
            int col = index % tile_size_j; // Col index in S (0 to Bc-1)


            S[index] = 0.0f; // Initialize S to zero
            for (int k = 0; k < d; k++) {
                S[index] += Qi[row * d + k] * Kj[col * d + k]; // Dot product
            }
            S[index] *= scale; // Scale

            //store mi_pre
            if (col == 0) {
                mi_pre[row] = mi[row];
            }
            //mi[row] = fmaxf(mi[row], S[index]); 
            // Update mi (max), there will be conflict here, we need to use warp reduction to update mi
            // warp_reduce_max(&mi[row], S[index]);
            // Warp reduce max for mi[row]
            // *all threads in a warp must have the same row index*  which is ensured by tile_size_j % WARP_SIZE == 0
            // if we cannot ensure all threads in a warp have the same row index, we should use atomic max to update mi[row]
            // atomic max for float is not supported in CUDA, we can use atomicCAS to implement it
            if (tile_size_j % WARP_SIZE != 0) {
                // Use atomicCAS to implement atomic max for float
                float old = mi[row];
                float assumed;
                do {
                    assumed = old;
                    #ifndef __INTELLISENSE__
                    old = atomicCAS((unsigned int*)&mi[row], __float_as_uint(assumed), __float_as_uint(fmaxf(assumed, S[index])));
                    #endif
                } while (assumed != old);
                continue;
            }

            // normally we use warp reduce max
            float val = S[index];
            unsigned mask = 0xffffffff;
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                float other = __shfl_down_sync(mask, val, offset);
                val = fmaxf(val, other);
            }
            if ((index % WARP_SIZE) == 0) {
                mi[row] = fmaxf(mi[row], val);
            }

        }
        __syncthreads(); // Ensure all threads have computed S before proceeding

        // line 9 : Pij = exp(Sij - mi); li;
        for (int step = 0; step < div_ceil(tile_size_i * tile_size_j, tnum); step++) {
            int index = step * tnum + tid;
            if (index >= tile_size_i * tile_size_j) {
                continue;
            }
            int row = index / tile_size_j; // Row index in S (0 to Br-1)
            int col = index % tile_size_j; // Col index in S (0 to Bc-1)
            float mi_val = mi[row];
            float Sij = S[index];
            float Pij = expf(Sij - mi_val); // Compute Pij
            P[index] = Pij; // Store Pij

            // Update li , li = exp(mi_pre - mi) * li + Pij
            // Use warp reduction to update li[row] += Pij
            // warp_reduce_sum(&li[row], Pij);
            if(col == 0) {
                li[row] = li[row] * expf(mi_pre[row] - mi_val); // Scale old li
            }

            //if we cannot ensure all threads in a warp have the same row index, we cannot use warp reduce sum
            // so we use warp reduce sum with atomic add to update li[row]
            if (tile_size_j % WARP_SIZE != 0 ) {
                atomicAdd(&li[row], Pij);
                continue;
            }
            // normally we use warp reduce sum
            unsigned mask = 0xffffffff;
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                float other = __shfl_down_sync(mask, Pij, offset);
                Pij += other;
            }
            if ((index % WARP_SIZE) == 0) {
                li[row] += Pij;
            }
        }
        __syncthreads(); // Ensure all threads have updated li before proceeding

        //line 10: Oi = Oi*exp(mi_pre - mi) + Pij Vj
        //one thread handles one element of Oi
        for(int idx = tid; idx < tile_size_i * d; idx += tnum) {
            int row = idx / d; // Row index in Oi (0 to Br-1)
            int col = idx % d; // Col index in Oi (0 to d-1)
            float sum = 0.0f;
            for(int j = 0; j < tile_size_j; j++) {
                sum += P[row * tile_size_j + j] * Vj[j * d + col];
            }
            // Scale Oi by exp(mi_pre - mi)
            Oi[idx] = Oi[idx] * expf(mi_pre[row] - mi[row]) + sum;
        }
    }
    __syncthreads(); // Ensure all threads have updated Oi before proceeding
    //12 compute Oi, Oi / li; 
    for(int idx = tid; idx < tile_size_i * d; idx += tnum){
        int row = idx / d; // Row index in Oi (0 to Br-1)
        Oi[idx] = Oi[idx] / li[row]; // Normalize Oi by li
        // Write back Oi to global memory
        O[(i_start + row) * d + (idx % d)] = Oi[idx];
    }
    //compute L, L = log(li) + mi; write back L to global memory
    for (int idx = tid; idx < tile_size_i; idx += tnum)
    {
        if (idx < tile_size_i) {
            L[i_start + idx] = logf(li[idx]) + mi[idx];
        }
    }
    __syncthreads();

}   
