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
    float* l,        // Row sum vector (N)
    float* m,        // Row max vector (N)
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
        Br +          // li (in SRAM)
        Br            // mi (in SRAM)
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
    float* li = &shared_mem[mem_offset +=  Br * d]; // Br
    float* mi = &shared_mem[mem_offset +=  Br]; // Br
    



    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_threads = blockDim.x;


    int Tc = (N + Br - 1) / Br; // Total number of row tiles
    for (int tile = 0; tile < Tc; tile++) {
        //line 6 liad K,V to SRAM
        // Algorithm Line 10: Load Kj, Vj from HBM to on-chip SRAM
        int j_start = tile * Bc;
        int j_end = j_start + Bc < N ? j_start + Bc : N;
        int tile_size_j = j_end - j_start; // Actual size of the tile (may be smaller at edges)

        for (int j = 0; j < tile_size_j; j++) {
            for (int t = 0; t < (d + num_threads - 1) / num_threads; t++)
            {
                int k = t * num_threads + tid;
                if (k >= d) continue; // Guard against out-of-bounds
                //kj[j][k] = K[(j_start + j)][k]; // Load from global memory
                Kj[j * d + k] = K[(j_start + j) * d + k]; // Load from global memory
                Vj[j * d + k] = V[(j_start + j) * d + k]; // Load from global memory
            }
        }

        // Algorithm Line 5: for 1 ≤ i ≤ Tr do : this is handled by launching multiple threads

        // if tid is out of bounds, return
        int row = bid * blockDim.x + tid;
        if (row >= N) {
            return;
        }

        // line 8: load Br rows of Qi, Oi, li, mi
        for (int col = 0; col < d; col++) {
            Qi[tid * d + col] = Q[row * d + col]; // Load from global memory.  todo: seems put Qi in register is better
            Oi[tid * d + col] = O[row * d + col];  // Load from global memory
        }
        li[tid] = l[row];  // Load from global memory
        mi[tid] = m[row];  // Load from global memory

        __syncthreads();

        //line 9: compute Sij = QiKj^T / sqrt(d).    size S is Br x Bc
        for (int j = 0; j < tile_size_j; j++) {
            //we only compute one row( row tid) of S per thread
            S[tid * Bc + j] = 0.0f;
            for (int k = 0; k < d; k++) {
                S[tid * Bc + j] += Qi[tid * d + k] * Kj[j * d + k]; // Dot product
            }
            S[tid * Bc + j] *= scale; // Scale
        }

        // line 10: compute mij, Pij, lij; for current thread row tid, we only need to compute one row
        float mij = -FLT_MAX;
        for (int j = 0; j < tile_size_j; j++) {
            if (S[tid * Bc + j] > mij) {
                mij = S[tid * Bc + j]; // Find max
            }
        }
        // Pij = exp(Sij - mij); lij = lij + sum_j Pij
        float lij = 0.0f;
        for (int j = 0; j < tile_size_j; j++) {
            P[tid * Bc + j] = expf(S[tid * Bc + j] - mij); // Subtract max for numerical stability
            lij += P[tid * Bc + j];
        }

        //line 11: mi_new, li_new
        float mi_new = fmaxf(mi[tid], mij);
        float li_new = li[tid] * expf(mi[tid] - mi_new) + lij * expf(mij - mi_new);

        /*
        __syncthreads();
        //print debug info 
        //print Sij

        if (tid == 0){
            printf("Tile %d, Block %d, Thread %d, tile_size_j %d: Pij = [", tile, bid, tid, tile_size_j);
            for (int tid_ = 0; tid_ < 2; tid_++)
            {
                for(int j = 0; j < tile_size_j; j++){
                    printf("%f, ", P[tid_ * Bc + j]);
                }
            }
            printf("]\n");
        }
        */

        //line 12: Oi = li / li_new * exp(mi_mi_new) * Oi + exp(mij - mi_new)/li_new * Pij Vj
        for (int col = 0; col < d; col++) {
            Oi[tid * d + col] = (li[tid] / li_new) * expf(mi[tid] - mi_new) * Oi[tid * d + col];
            //         
            for (int j = 0; j < tile_size_j; j++) {
                Oi[tid * d + col] += (expf(mij - mi_new) / li_new) * P[tid * Bc + j] * Vj[j * d + col];
            }
        }
        //line 13: write back mi_new, li_new, Oi to HBM
        l[row] = li_new;  // Write back to global memory
        m[row] = mi_new;  // Write back to global memory

        for (int col = 0; col < d; col++) {
            O[row * d + col] = Oi[tid * d + col]; // Write back to global memory
        }
        __syncthreads();
    }
}        