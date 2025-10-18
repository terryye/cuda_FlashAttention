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
    #pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, mask);
    }
    return val;
}

template <int Br, int Bc, int d_max, int num_warps>
__global__ void flash_attention_2_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
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
    const int warp_row_start = warp_id * rows_per_warp;
    
    // Global row indices for this thread block
    const int global_row_start = row_block_idx * Br;
    const int global_row_end = min(global_row_start + Br, N);
    
    // Registers for Q tile (each warp loads its portion)
    float Q_reg[rows_per_warp][d_max];
    
    // Load Q tile for this warp
    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        int global_row = global_row_start + warp_row_start + local_row;
        if (global_row < global_row_end) {
            for (int col = lane_id; col < d; col += WARP_SIZE) {
                Q_reg[local_row][col] = Q[global_row * d + col];
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
    float O_acc[rows_per_warp][d_max];
    
    #pragma unroll
    for (int i = 0; i < rows_per_warp; i++) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < d_max; j++) {
            O_acc[i][j] = 0.0f;
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
        for (int idx = tid; idx < actual_Bc * d; idx += num_threads) {
            int col_in_block = idx / d;
            int d_idx = idx % d;
            int global_col = col_block_start + col_in_block;
            K_smem[col_in_block * d + d_idx] = K[global_col * d + d_idx];
            V_smem[col_in_block * d + d_idx] = V[global_col * d + d_idx];
        }
        
        __syncthreads();
        
        // Compute S = Q @ K^T for this warp's rows
        float S_row[Bc];
        
        for (int local_row = 0; local_row < rows_per_warp; local_row++) {
            int global_row = global_row_start + warp_row_start + local_row;
            if (global_row < global_row_end) {
                // Compute dot products
                for (int col = 0; col < actual_Bc; col++) {
                    float dot = 0.0f;
                    // Only access elements this thread loaded
                    for (int k = lane_id; k < d; k += WARP_SIZE) {
                        dot += Q_reg[local_row][k] * K_smem[col * d + k];
                    }
                    // Perform warp reduction to sum all threads' contributions
                    dot = warp_reduce_sum(dot);
                    S_row[col] = dot * softmax_scale;
                }
                
                // Online softmax - compute row max
                float row_max = -INFINITY;
                for (int col = 0; col < actual_Bc; col++) {
                    row_max = fmaxf(row_max, S_row[col]);
                }
                
                // Update statistics
                float m_new = fmaxf(m[local_row], row_max);
                float exp_diff = expf(m[local_row] - m_new);
                
                // Compute exp and sum
                float row_sum = 0.0f;
                #pragma unroll
                for (int col = 0; col < actual_Bc; col++) {
                    S_row[col] = expf(S_row[col] - m_new);
                    row_sum += S_row[col];
                }
                
                // Update output accumulator with rescaling
                #pragma unroll
                for (int j = 0; j < d; j++) {
                    O_acc[local_row][j] *= exp_diff;
                }
                
                // Accumulate P @ V
                #pragma unroll
                for (int col = 0; col < actual_Bc; col++) {
                    #pragma unroll
                    for (int j = 0; j < d; j++) {
                        O_acc[local_row][j] += S_row[col] * V_smem[col * d + j];
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
            for (int col = lane_id; col < d; col += WARP_SIZE) {
                O[global_row * d + col] = O_acc[local_row][col] * inv_l;
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
    float softmax_scale,
    cudaStream_t stream = 0
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
        flash_attention_2_forward_kernel<Br, Bc, 64, num_warps><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, L, seq_len, head_dim, softmax_scale
        );
        #endif
    } else {
        
        #ifndef INTELLISENSE
        flash_attention_2_forward_kernel<Br, Bc, 128, num_warps><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, L, seq_len, head_dim, softmax_scale
        );
        #endif
    }
}
