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
    float* mi_pre = &shared_mem[mem_offset += Br * 1]; // Br * 1

    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int tnum = blockDim.x * blockDim.y; // Number of threads per block

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

    __syncthreads(); // Ensure all threads have loaded Qi before proceeding
    // Initialize mi to -inf and li to 0
    for ( int row = tid ; row < Br ; row += tnum ) {
        mi[row] = -FLT_MAX; // Initialize mi to -inf
        li[row] = 0.0f;     // Initialize li to 0
    }



    //line 6 
    for(int tile = 0; tile < Tc; tile++) { // Loop over tiles of K,V
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
        // one thread handles one row of Sij
        for (int row = tid; row < tile_size_i; row += tnum) {

            mi_pre[row] = mi[row]; // Store previous mi
            for (int col = 0; col < tile_size_j; col++) {
                float dot_product = 0.0f;
                for (int k = 0; k < d; k++) {
                    dot_product += Qi[row * d + k] * Kj[col * d + k];
                }
                float Sij = dot_product * scale; // Scale by 1/sqrt(d)
                S[row * Bc + col] = Sij; // Store Sij

                // Update mi[row] = max(mi[row], Sij)
                mi[row] = fmaxf(mi[row], Sij);
            }
        }
        // __syncthreads(); //  we don't need this because each thread only writes to its own mi[row], and S is local to this tile

        // line 9 : Pij = exp(Sij - mi); li;
        for (int row = tid; row < tile_size_i; row += tnum) {
            li[row] = li[row] * exp(mi_pre[row] - mi[row]); // update li[row]
            for (int col = 0; col < tile_size_j; col++) {
                float Sij = S[row * Bc + col];
                float pij = expf(Sij - mi[row]); // Compute Pij
                P[row * Bc + col] = pij; // Store Pij

                // Update li[row] += pij
                li[row] += pij;
            }
        }
        // __syncthreads(); //  we don't need this because each thread only writes to its own li[row], Pij is local to this tile

        //line 10: Oi = Oi*exp(mi_pre - mi) + Pij Vj
        //one thread handles one row of Oi
        for(int row = tid; row < tile_size_i; row += tnum) {
            // Scale existing Oi by exp(mi_pre - mi)
            float scale_Oi = expf(mi_pre[row] - mi[row]);
            for(int col = 0; col < d; col++) {
                Oi[row * d + col] *= scale_Oi;
            }
            // Accumulate Pij * Vj
            for(int j_col = 0; j_col < tile_size_j; j_col++) {
                float pij = P[row * Bc + j_col];
                for(int v_col = 0; v_col < d; v_col++) {
                    Oi[row * d + v_col] += pij * Vj[j_col * d + v_col];
                }
            }
        }
    }
    // __syncthreads(); //we don't need this because each thread only writes to its own li[row], Pij is local 
    
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

}   
