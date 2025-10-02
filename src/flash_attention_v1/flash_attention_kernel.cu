#include "../../util/cuda_shim.h"
#include <cfloat>

#define MAX_D 256  // Maximum embedding dimension supported

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
    int Bc          // Block cols
) {

    // Shared memory for tiles
    extern __shared__ float shared_mem[];
    float* Qi = shared_mem;                    // Br x d
    float* Kj = &shared_mem[Br * d];          // Bc x d
    float* Vj = &shared_mem[Br * d + Bc * d]; // Bc x d
    float* S = &shared_mem[Br * d + 2 * Bc * d]; // Br x Bc
    
    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Number of tiles
    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;
    
    // Each block handles one tile of Q (outer loop iteration)
    if (bid >= Tr) return;
    
    int i_start = bid * Br;
    int i_end = min(i_start + Br, N); 
    int tile_size_i = i_end - i_start;
    
    // Initialize local accumulators for each thread
    // Using fixed-size array with MAX_D
    float Oi_local[MAX_D];
    float mi_local;
    float li_local;
    
    // Algorithm Line 5: for 1 ≤ i ≤ Tr do
    // Each block handles one value of i in the outer loop
    
    // Algorithm Lines 6-8: Initialize Oi, li, mi
    if (tid < tile_size_i) {
        // Initialize Oi = 0 (zeros matrix)
        for (int k = 0; k < d; k++) {
            Oi_local[k] = 0.0f;
        }
        // Initialize mi = -∞ (negative infinity)
        mi_local = -FLT_MAX;
        // Initialize li = 0 (zeros vector)
        li_local = 0.0f;
    }
    
    // Algorithm Line 6: Load Qi from HBM to on-chip SRAM
    for (int row = tid; row < tile_size_i; row += blockDim.x) {
        for (int col = 0; col < d; col++) {
            Qi[row * d + col] = Q[(i_start + row) * d + col];
        }
    }
    __syncthreads();
    
    float scaler = 1.0f / sqrtf((float)d);

    // Algorithm Line 9: for 1 ≤ j ≤ Tc do
    // Process all K,V tiles
    for (int j = 0; j < Tc; j++) {
        int j_start = j * Bc;
        int j_end = min(j_start + Bc, N);
        int tile_size_j = j_end - j_start;
        
        // Algorithm Line 10: Load Kj, Vj from HBM to on-chip SRAM
        for (int idx = tid; idx < tile_size_j * d; idx += blockDim.x) {
            int row = idx / d;
            int col = idx % d;
            Kj[row * d + col] = K[(j_start + row) * d + col];
            Vj[row * d + col] = V[(j_start + row) * d + col];
        }
        __syncthreads();
        
        // Algorithm Line 11: On chip, compute Sij = QiKj^T ∈ R^(Br×Bc)
        if (tid < tile_size_i) {
            for (int jj = 0; jj < tile_size_j; jj++) {
                float sum = 0.0f;
                for (int k = 0; k < d; k++) {
                    sum += Qi[tid * d + k] * Kj[jj * d + k];
                }
                S[tid * Bc + jj] = sum * scaler;  // scaler is 1/sqrt(d) scaling factor
            }
        }
        __syncthreads();
        
        // Algorithm Lines 11-15: Update statistics and output
        if (tid < tile_size_i) {
            // Algorithm Line 11: On chip, compute mij = rowmax(Sij) ∈ R^Br
            float mij = mi_local;
            for (int jj = 0; jj < tile_size_j; jj++) {
                mij = fmaxf(mij, S[tid * Bc + jj]);
            }
            
            // Algorithm Line 12: On chip, compute mi^new = max(mi, mij) ∈ R^Br
            float mi_new = fmaxf(mi_local, mij);
            
            // Algorithm Line 11 (continued): Pij = exp(Sij - mij) ∈ R^(Br×Bc) (pointwise)
            // Using mi_new instead of mij for numerical stability
            float Pij_sum = 0.0f;
            for (int jj = 0; jj < tile_size_j; jj++) {
                S[tid * Bc + jj] = expf(S[tid * Bc + jj] - mi_new);
                Pij_sum += S[tid * Bc + jj];
            }
            
            // Algorithm Line 12 (continued): li^new = e^(mi-mi^new) * li + rowsum(Pij) ∈ R^Br
            float li_new = expf(mi_local - mi_new) * li_local + Pij_sum;
            
            // Algorithm Line 13: Write Oi ← diag(li^new)^(-1) * (diag(li) * e^(mi-mi^new) * Oi + PijVj) to HBM
            // We compute this incrementally in SRAM
            for (int k = 0; k < d; k++) {
                float sum = 0.0f;
                for (int jj = 0; jj < tile_size_j; jj++) {
                    sum += S[tid * Bc + jj] * Vj[jj * d + k];  // PijVj
                }
                // Update Oi using the online softmax formula
                Oi_local[k] = (li_local * expf(mi_local - mi_new) * Oi_local[k] + sum) / li_new;
            }
            
            // Algorithm Line 14: Write li ← li^new, mi ← mi^new to HBM
            // We keep these in local registers and write at the end
            mi_local = mi_new;
            li_local = li_new;
        }
        __syncthreads();
    }
    
    // Write final results to HBM
    if (tid < tile_size_i) {
        int global_idx = i_start + tid;
        
        // Write Oi to HBM (Line 13 - final write)
        for (int k = 0; k < d; k++) {
            O[global_idx * d + k] = Oi_local[k];
        }
        
        // Write li, mi to HBM (Line 14 - final write)
        l[global_idx] = li_local;
        m[global_idx] = mi_local;
    }
}