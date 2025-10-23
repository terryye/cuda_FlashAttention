#pragma once

#include "../../util/cuda_shim.h"
#include "../../util/assertc.h"
#include <stdio.h>
#include <math.h>
#include <cfloat>
#include <math.h>
#include <float.h>


#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Helper function to get number of blocks
__host__ __device__ inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// Warp-level reduction for sum
__device__ float warp_reduce_sum(float val) {
    //#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, mask);
    }
    return val;
}

// Add this warp reduction function for max at the top of the file with other helpers
__device__ float warp_reduce_max(float val) {
    for (int mask = WARP_SIZE/2; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, mask));
    }
    return val;
}

template <int Br, int Bc, int d_max, int num_warps>
__global__ void flash_attention_2_forward_kernel(
    const float*  Q,
    const float*  K,
    const float*  V,
    float*  O,
    float*  L,
    const int N,
    const int d,
    const float softmax_scale
) {
    // Thread block handles one block row of attention matrix
    const int row_block_idx = blockIdx.x;
    
    // Shared memory for K and V tiles
    extern __shared__ float smem[];
    float* K_smem = smem; // [Bc][d]
    float* V_smem = smem + Bc * d; // [Bc][d]
    
    // Thread and warp identifiers
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_threads = blockDim.x;
    
    // Each warp handles different rows
    const int rows_per_warp = Br / num_warps;
    const int cols_per_thread = (d_max + WARP_SIZE - 1) / WARP_SIZE;
    const int warp_row_start = warp_id * rows_per_warp;
    
    // Global row indices for this thread block
    const int global_row_start = row_block_idx * Br;
    const int global_row_end = min(global_row_start + Br, N);
    
    // Registers for Q tile (each warp loads its portion)
    float Q_reg[rows_per_warp][cols_per_thread];  // +1 to handle when d is not divisible by WARP_SIZE
    
    // Load Q tile for this warp
    // each warp loads its assigned rows, each thread loads its assigned columns
    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        int global_row = global_row_start + warp_row_start + local_row;
        if (global_row < global_row_end) {
            // Each thread only loads its assigned columns
            for (int col_offset = 0; col_offset < cols_per_thread; col_offset++) {
                int col = lane_id + col_offset * WARP_SIZE;
                if (col < d) {
                    Q_reg[local_row][col_offset] = Q[global_row * d + col];
                } else {
                    Q_reg[local_row][col_offset] = 0.0f;
                }
            }
        } else {
            for (int col = lane_id; col < d; col += WARP_SIZE) {
                Q_reg[local_row][col] = 0.0f;
            }
        }
    }
    
    // Ensure all threads in warp have loaded their Q values
    __syncwarp();
    
    // Initialize row-wise statistics
    float m[rows_per_warp];
    float l[rows_per_warp];
    // Each thread only needs to store its portion of O_acc
    float O_acc[rows_per_warp][cols_per_thread];
    
    //#pragma unroll
    for (int i = 0; i < rows_per_warp; i++) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        // Each thread only initializes its columns
        for (int col_offset = 0; col_offset < cols_per_thread; col_offset++) {
            O_acc[i][col_offset] = 0.0f;
        }
    }
    
    // Main loop over K,V blocks
    const int num_blocks = div_up(N, Bc);
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int col_block_start = block_idx * Bc;
        const int col_block_end = min(col_block_start + Bc, N);
        const int actual_Bc = col_block_end - col_block_start;
        
        __syncthreads();
        
        // Collaboratively load K and V tiles to shared memory
        int col_block_start_idx = col_block_start * d;
        for (int idx = tid; idx < actual_Bc * d; idx += num_threads) {
            K_smem[idx] = K[col_block_start_idx + idx];
            V_smem[idx] = V[col_block_start_idx + idx];
        }
        
        __syncthreads();
        
        // Compute S = Q @ K^T for this warp's rows
        float S_row[Bc];
        
        for (int local_row = 0; local_row < rows_per_warp; local_row++) { // each row of Q in this warp
            int global_row = global_row_start + warp_row_start + local_row;
            if (global_row < global_row_end) {
                // Compute dot products
                for (int col = 0; col < actual_Bc; col++) { // each column in K_smem
                    float dot = 0.0f;
                    // Access Q_reg and K_smem to compute dot product
                    for (int k = lane_id; k < d; k += WARP_SIZE) { // each dimension
                        int q_col = k / WARP_SIZE;
                        float q_val = Q_reg[local_row][q_col];
                        dot += q_val * K_smem[col * d + k];  
                    }
                    // Perform warp reduction to sum all threads' contributions
                    dot = warp_reduce_sum(dot);     //got result for one element of S_row before scaling
                    
                    //We store S_row in all threads instead of only lane 0 to avoid warp communication, since we need it for softmax
                    //if we have limited registers we can optimize it further by only storing in lane 0 and broadcasting later, but this is simpler and faster 
                    S_row[col] = dot * softmax_scale;  
                }
                

                // Online softmax - compute row max
                float row_max = -INFINITY;
                // Each thread processes its chunk of columns
                for (int col = lane_id; col < actual_Bc; col += WARP_SIZE) {
                    row_max = fmaxf(row_max, S_row[col]);
                }
                // Reduce across warp
                row_max = warp_reduce_max(row_max);

                
                // Update statistics
                float m_new = fmaxf(m[local_row], row_max);
                float exp_diff = expf(m[local_row] - m_new);
                
                // Compute exp and sum
                float row_sum = 0.0f;
                for (int col = 0; col < actual_Bc; col++) {
                    S_row[col] = expf(S_row[col] - m_new);
                    row_sum += S_row[col];
                }
                
                // Update output accumulator with rescaling
                for (int col_offset = 0; col_offset < cols_per_thread; col_offset++) {
                    int col = lane_id + col_offset * WARP_SIZE;
                    if (col < d) {
                        O_acc[local_row][col_offset] *= exp_diff;
                    }
                }

                // Accumulate P @ V
                for (int k = 0; k < actual_Bc; k++) {
                    float s_val = S_row[k];
                    for (int col_offset = 0; col_offset < cols_per_thread; col_offset++) {
                        int col = lane_id + col_offset * WARP_SIZE;
                        if (col < d) {
                            O_acc[local_row][col_offset] += s_val * V_smem[k * d + col];
                        }
                    }
                }
                
                // Update running statistics
                l[local_row] = exp_diff * l[local_row] + row_sum;
                m[local_row] = m_new;
            }
        }
    }
    
    // Write output - normalize and store
    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        int global_row = global_row_start + warp_row_start + local_row;
        if (global_row < global_row_end) {
            float inv_l = 1.0f / l[local_row];
            // Each thread writes its portion of the output
            for (int col_offset = 0; col_offset < cols_per_thread; col_offset++) {
                int col = lane_id + col_offset * WARP_SIZE;
                if (col < d) {
                    O[global_row * d + col] = O_acc[local_row][col_offset] * inv_l;
                }
            }
            // Store logsumexp
            if (lane_id == 0) {
                L[global_row] = m[local_row] + logf(l[local_row]);
            }
        }
    }
}

// Host wrapper function
void flash_attention_2_forward(
    const float* Q,
    const float* K, 
    const float* V,
    float* O,
    float* L,
    int seq_len,
    int head_dim,
    float softmax_scale
) {
    // Configure kernel parameters
    const int Br = 64;  // Row block size
    const int Bc = 64;  // Column block size
    const int num_warps = 4;
    const int threads_per_block = num_warps * WARP_SIZE;
    const int d_max = 128;  // Maximum supported head dimension
    
    assertc(head_dim <= d_max); // Constraint for this implementation
    assertc(Br % num_warps == 0);
    
    int num_row_blocks = div_up(seq_len, Br);
    


    dim3 grid = new_dim3(num_row_blocks,1,1);
    dim3 block = new_dim3(threads_per_block,1,1);
    
    int smem_size = 2 * Bc * head_dim * sizeof(float); // K_smem + V_smem
    
    if (head_dim <= 64) {
        #ifndef __INTELLISENSE__
        flash_attention_2_forward_kernel<Br, Bc, 64, num_warps><<<grid, block, smem_size>>>(
            Q, K, V, O, L, seq_len, head_dim, softmax_scale
        );
        #endif
    } else {
        #ifndef __INTELLISENSE__
        flash_attention_2_forward_kernel<Br, Bc, 128, num_warps><<<grid, block, smem_size>>>(
            Q, K, V, O, L, seq_len, head_dim, softmax_scale
        );
        #endif
    }
}
