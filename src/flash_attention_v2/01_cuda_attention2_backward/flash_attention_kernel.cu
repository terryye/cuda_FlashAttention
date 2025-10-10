#include "../../util/cuda_shim.h"
#include <stdio.h>
#include <math.h>
#include <cfloat>

// Kernel for Flash Attention 2 Backward Pass
// dQ =  softmax(QK^T) ∘ (dO V^T - D) K  ; D = rowsum(dO * O)

__global__ void flash_attention_kernel(
    const float* Q,  // Query matrix (N x d)
    const float* K,  // Key matrix (N x d)
    const float* V,  // Value matrix (N x d)
    const float* O,  // Output matrix (N x d)
    const float* dO, // Gradient of output matrix (N x d)
    const float* L, // 
    float* dQ,
    float* dK,
    float* dV,
    int N,           // Sequence length
    int d,           // Embedding dimension
    int Br,          // Block rows
    int Bc,          // Block cols
    float scale      // Softmax scale (1/sqrt(d) typically)
) {
    // Shared memory for tiles
    extern __shared__ float smem[];
    float* Kj = smem;                          // [Bc, d]
    float* Vj = Kj + Bc * d;                // [Bc, d]
    float* Qi = Vj + Bc * d;                // [Br, d]
    float* Oi = Qi + Br * d;                // [Br, d]
    float* dOi = Oi + Br * d;               // [Br, d]
    float* dQi = dOi + Br * d;              // [Br, d]
    float* dKj = dQi + Br * d;              // [Bc, d]
    float* dVj = dKj + Bc * d;              // [Bc, d]
    float* S = dVj + Bc * d;                // [Br, Bc]
    float* P = S + Br * Bc;                 // [Br, Bc]
    float* dP = P + Br * Bc;                // [Br, Bc]
    float* dS = dP + Br * Bc;               // [Br, Bc]

    float Li;
    float Di;
    
    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int row = bid * blockDim.x + tid;

    int Tc = (N + Bc - 1) / Bc; // Total number of tiles in columns
//    int Tr = (N + Br - 1) / Br; // Total number of tiles in rows

    // Check boundary condition early
    if (row >= N) {
        return;
    }

    int num_threads = N - blockDim.x * bid < blockDim.x ? N - blockDim.x * bid : blockDim.x;

    // Initialize dQi to zero once
    for (int col = 0; col < d; col++) {
        dQi[tid * d + col] = 0.0f;
    }

    for (int tile = 0;  tile < Tc; tile++) {
            
        //line 6: load K,V to SRAM
        int j_start = tile * Bc;
        int j_end = j_start + Bc < N ? j_start + Bc : N;
        int tile_size_j = j_end - j_start; // Actual size of the tile (may be smaller at edges)

        // All threads participate in loading K,V
        for (int j = 0; j < tile_size_j; j++) {
            for (int t = 0; t < (d + num_threads - 1) / num_threads; t++)
            {
                int k = t * num_threads + tid;
                if (k >= d) continue; // Guard against out-of-bounds
                Kj[j * d + k] = K[(j_start + j) * d + k]; // Load from global memory
                Vj[j * d + k] = V[(j_start + j) * d + k]; // Load from global memory
                
                //line 7: init dKj, dVj
                dKj[j * d + k] = 0.0f; // Initialize to zero
                dVj[j * d + k] = 0.0f; // Initialize to zero
            }
        }
        __syncthreads();
        
        //line 9: Load Q𝑖,O𝑖,dO𝑖,𝐋𝑖,𝐃𝑖 from HBM to on-chip SRAM
        Di = 0.0f;     // Initialize Di to zero.
        for (int col = 0; col < d; col++) {
            Qi[tid * d + col] = Q[row * d + col];     // Load Qi
            Oi[tid * d + col] = O[row * d + col];     // Load Oi
            dOi[tid * d + col] = dO[row * d + col];   // Load dOi
            Di += dOi[tid * d + col] * Oi[tid * d + col]; // Compute Di
        }
        Li = L[row];                       // Load Li

        __syncthreads();

        //line 10: compute Sij = QiKj^T / sqrt(d).    size S is Br x Bc
        for (int j = 0; j < tile_size_j; j++) {
            //we only compute one row( row tid) of S per thread
            S[tid * Bc + j] = 0.0f;
            for (int k = 0; k < d; k++) {
                S[tid * Bc + j] += Qi[tid * d + k] * Kj[j * d + k]; // Dot product
            }
            S[tid * Bc + j] *= scale; // Scale
        }
        __syncthreads();

        // line 11: compute Pij = exp(Sij - Li)
        for (int j = 0; j < tile_size_j; j++) {
            P[tid * Bc + j] = expf(S[tid * Bc + j] - Li); // Subtract max for numerical stability
        }
  
        __syncthreads();

        //compute dVj
        for (int col = 0; col < d; col++) {
            for (int j = 0; j < tile_size_j; j++) {
                dVj[j * d + col] += P[tid * Bc + j] * dOi[tid * d + col];
            }
        }
        __syncthreads();

        //compute dPij
        for (int j = 0; j < tile_size_j; j++) {
            dP[tid * Bc + j] = 0.0f;  // Initialize to zero
            for (int col = 0; col < d; col++) {
                dP[tid * Bc + j] += dOi[tid * d + col] * Vj[j * d + col];
            }
        }
        __syncthreads();

        //compute dSij
        for (int j = 0; j < tile_size_j; j++) {
            dS[tid * Bc + j] = P[tid * Bc + j] * (dP[tid * Bc + j] - Di); // Element-wise
        }
        __syncthreads();

        //line 15: compute dQi and update to Global memory
        for (int k = 0; k < d; k++) {
            for (int j = 0; j < tile_size_j; j++) {
                dQi[tid * d + k] += dS[tid * Bc + j] * Kj[j * d + k] * scale;  // Apply scale factor
            }
        }
        __syncthreads();
        
        //line 16: compute dKj
        for (int k = 0; k < d; k++) {
            for (int j = 0; j < tile_size_j; j++) {
                dKj[j * d + k] += dS[tid * Bc + j] * Qi[tid * d + k] * scale;  // Apply scale factor
            }
        }
        __syncthreads();

        // Write back dKj and dVj to global memory using atomics
        for (int j = 0; j < tile_size_j; j++) {
            for (int t = 0; t < (d + num_threads - 1) / num_threads; t++) {
                int col = t * num_threads + tid;
                if (col >= d) continue;
                // Use atomics for proper accumulation across blocks
                atomicAdd(&dK[(j_start + j) * d + col], dKj[j * d + col]);
                atomicAdd(&dV[(j_start + j) * d + col], dVj[j * d + col]);
            }
        }
        __syncthreads();

    }
    // Write back dQi to global memory
    if (row < N) {
        for (int col = 0; col < d; col++) {
            dQ[row * d + col] = dQi[tid * d + col];
        }
    }
}
