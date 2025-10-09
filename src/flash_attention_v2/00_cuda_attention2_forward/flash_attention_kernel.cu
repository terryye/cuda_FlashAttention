#include "../../util/cuda_shim.h"
#include <stdio.h>
#include <math.h>
#include <cfloat>

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
        Br * d +      // Oi (in SRAM)
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
    

    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_threads = blockDim.x; // equal to Br
//    int Tr = (N + Br - 1) / Br; // Total number of row tiles
    int Tc = (N + Bc - 1) / Bc; // Total number of col tiles

    //line 4: load Qi , init Oi
    for (int col = 0; col < d; col++)
    {
        //check boundary
        if (bid * Br + tid >= N) break;
        //Qi[tid][col] = Q[bid * Br + tid][col]; // Load from global memory
        Qi[tid * d + col] = Q[ (bid * Br + tid) * d + col]; // Initialize Qi to zero
        Oi[tid * d + col] = 0.0f; // Initialize Oi to zero
    }
    //initialize li/mi 
    float li = 0.0f; // Initialize li to zero
    float mi = -FLT_MAX; // Initialize mi to -inf

    __syncthreads(); // Ensure all threads have loaded Qi before proceeding

    //line 6 
    for(int tile = 0; tile < Tc; tile++) { // Loop over tiles of K,V
        // each thread load one row of K,V
        // Algorithm Line 10: Load Kj, Vj from HBM to on-chip SRAM
        int j_start = tile * Bc;
        int j_end = j_start + Bc < N ? j_start + Bc : N;
        int tile_size_j = j_end - j_start; // Actual size of the tile (may be smaller at edges)

        for (int j = 0; j < tile_size_j; j++) {
            for (int t = 0; t < (d + Br - 1) / Br; t++)
            {
                int k = t * num_threads + tid;
                if (k >= d) continue; // Guard against out-of-bounds
                //kj[j][k] = K[(j_start + j)][k]; // Load from global memory
                Kj[j * d + k] = K[(j_start + j) * d + k]; // Load from global memory
                Vj[j * d + k] = V[(j_start + j) * d + k]; // Load from global memory
            }
        }
        //check boundary for Oi, after loading Kj,Vj, we don't need the thread any more
        if (bid * Br + tid >= N) continue;

        __syncthreads(); // Ensure all threads have loaded Kj, Vj before proceeding

        float mi_old = mi; // Reset mij for the new tile
        // line 8 compute Sij = QiKj^T / sqrt(d).    size S is Br x Bc
        for (int j = 0; j < tile_size_j; j++) {
            //we only compute one row(row tid) of S per thread
            int sid = tid * Bc + j;
            S[sid] = 0.0f;
            for (int k = 0; k < d; k++) {
                S[sid] += Qi[tid * d + k] * Kj[j * d + k]; // Dot product
            }

            S[sid] *= scale; // Scale
            //line 9 compute mi.
            mi = fmaxf(mi, S[sid]); // Update mi (max)
        }
        
        // line 9 : Pij = exp(Sij - mi); lij = sum_j Pij
        float lij = 0; // Initialize lij to zero
        for (int j = 0; j < tile_size_j; j++) {
            int sid = tid * Bc + j;
            P[sid] = expf(S[sid] - mi); // Subtract mi for numerical stability
            lij += P[sid]; // Update lij (sum)  
        }
        li = li == 0 ? lij : li * expf(mi_old - mi) + lij; // Update li

        //line 10: Oi = Oi*exp(mi_old - mi) + Pij Vj
        float scale_Oi = (li == lij) ? 1.0f : expf(mi_old - mi); // Scale factor for Oi
        for (int col = 0; col < d; col++) {
            Oi[tid * d + col] = Oi[tid * d + col] * scale_Oi; // Scale Oi
            for (int j = 0; j < tile_size_j; j++) {
                Oi[tid * d + col] += P[tid * Bc + j] * Vj[j * d + col]; // Update Oi (don't divide by li here)
            }   
        }
        __syncthreads(); 
    }

    if (bid * Br + tid >= N) return; // Boundary check
    //12 compute Oi write back to HBM
    for(int col = 0; col < d; col++)
    {
        O[(bid * Br + tid) * d + col] = Oi[tid * d + col] / li; // Final scale of Oi
    }
    L[bid * Br + tid] = mi + logf(li); // Write back li to global memory
}   
