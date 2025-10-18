#pragma once

#include "../../util/cuda_shim.h"
#include "../../util/assertc.h"

#include <stdio.h>
#include <math.h>
#include <cfloat>
#include <vector>
#include <iostream>

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
__global__ void flash_attention_2_backward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    const int N,
    const int d,
    const float softmax_scale
) {
    // Thread block handles one block column of attention matrix
    const int col_block_idx = blockIdx.x;
    
    // Shared memory allocation
    extern __shared__ float smem[];
    float* K_smem = smem; // [Bc][d]
    float* V_smem = smem + Bc * d; // [Bc][d]
    float* dK_smem = smem + 2 * Bc * d; // [Bc][d]
    float* dV_smem = smem + 3 * Bc * d; // [Bc][d]
    
    // Thread and warp identifiers
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_threads = blockDim.x;
    
    // Each warp handles different rows
    const int rows_per_warp = Bc / num_warps;
    const int warp_row_start = warp_id * rows_per_warp;
    
    // Global column indices for this thread block
    const int global_col_start = col_block_idx * Bc;
    const int global_col_end = min(global_col_start + Bc, N);
    const int actual_Bc = global_col_end - global_col_start;
    
    // Initialize dK_smem and dV_smem to 0
    for (int idx = tid; idx < actual_Bc * d; idx += num_threads) {
        dK_smem[idx] = 0.0f;
        dV_smem[idx] = 0.0f;
    }
    
    // Load K and V blocks
    for (int idx = tid; idx < actual_Bc * d; idx += num_threads) {
        int col_in_block = idx / d;
        int d_idx = idx % d;
        int global_col = global_col_start + col_in_block;
        K_smem[col_in_block * d + d_idx] = K[global_col * d + d_idx];
        V_smem[col_in_block * d + d_idx] = V[global_col * d + d_idx];
    }
    
    __syncthreads();
    
    // Main loop over Q,O,dO blocks (row blocks)
    const int num_blocks = div_up(N, Br);
    
    for (int row_block_idx = 0; row_block_idx < num_blocks; row_block_idx++) {
        const int global_row_start = row_block_idx * Br;
        const int global_row_end = min(global_row_start + Br, N);
        
        // Process rows for this warp
        for (int local_row = warp_row_start; local_row < min(warp_row_start + rows_per_warp, Br); local_row++) {
            int global_row = global_row_start + local_row;
            
            if (global_row < global_row_end) {
                // Load row data
                float row_L = L[global_row];
                float row_D = D[global_row];
                
                // Compute S values for this row
                float S_row[Bc];
                for (int col = 0; col < actual_Bc; col++) {
                    float dot = 0.0f;
                    // Each thread computes partial dot product
                    for (int k = lane_id; k < d; k += WARP_SIZE) {
                        dot += Q[global_row * d + k] * K_smem[col * d + k];
                    }
                    // Warp reduction
                    dot = warp_reduce_sum(dot);
                    
                    // Compute P from S using logsumexp
                    S_row[col] = expf(dot * softmax_scale - row_L);
                }
                
                // Compute dP = dO * V^T
                float dP_row[Bc];
                for (int col = 0; col < actual_Bc; col++) {
                    float dot = 0.0f;
                    // Each thread computes partial dot product
                    for (int k = lane_id; k < d; k += WARP_SIZE) {
                        dot += dO[global_row * d + k] * V_smem[col * d + k];
                    }
                    // Warp reduction
                    dot = warp_reduce_sum(dot);
                    dP_row[col] = dot;
                }
                
                // Compute dS = P * (dP - D)
                float dS_row[Bc];
                for (int col = 0; col < actual_Bc; col++) {
                    dS_row[col] = S_row[col] * (dP_row[col] - row_D);
                }
                
                // Accumulate dV += P^T * dO
                // Each thread handles different elements
                for (int col = 0; col < actual_Bc; col++) {
                    for (int j = lane_id; j < d; j += WARP_SIZE) {
                        float contrib = S_row[col] * dO[global_row * d + j];
                        atomicAdd(&dV_smem[col * d + j], contrib);
                    }
                }
                
                // Accumulate dK += dS^T * Q
                // Each thread handles different elements
                for (int col = 0; col < actual_Bc; col++) {
                    for (int j = lane_id; j < d; j += WARP_SIZE) {
                        float contrib = dS_row[col] * Q[global_row * d + j];
                        atomicAdd(&dK_smem[col * d + j], contrib);
                    }
                }
                
                // Compute dQ for this row
                for (int j = lane_id; j < d; j += WARP_SIZE) {
                    float sum = 0.0f;
                    for (int col = 0; col < actual_Bc; col++) {
                        sum += dS_row[col] * K_smem[col * d + j];
                    }
                    atomicAdd(&dQ[global_row * d + j], sum);
                }
            }
        }
        __syncthreads();
    }
    
    // Write dK and dV back to global memory
    for (int idx = tid; idx < actual_Bc * d; idx += num_threads) {
        int col_in_block = idx / d;
        int d_idx = idx % d;
        int global_col = global_col_start + col_in_block;
        dK[global_col * d + d_idx] = dK_smem[col_in_block * d + d_idx];
        dV[global_col * d + d_idx] = dV_smem[col_in_block * d + d_idx];
    }
}

// Kernel to compute D = rowsum(dO * O)
__global__ void compute_D_kernel(
    const float* __restrict__ O,
    const float* __restrict__ dO,
    float* __restrict__ D,
    const int seq_len,
    const int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        float sum = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            sum += dO[idx * head_dim + j] * O[idx * head_dim + j];
        }
        D[idx] = sum;
    }
}

// Host wrapper function
void flash_attention_2_backward(
    const float* Q,
    const float* K,
    const float* V,
    const float* O,
    const float* dO,
    const float* L,
    float* dQ,
    float* dK,
    float* dV,
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
    
    assertc(head_dim <= 128);
    assertc(Bc % num_warps == 0);
    
    // Compute D = rowsum(dO * O)
    float* d_D;
    cudaMalloc((void **)&d_D, seq_len * sizeof(float));
    
    // Launch kernel to compute D
    int threads = 256;
    int blocks = div_up(seq_len, threads);
    #ifndef __INTELLISENSE__
    compute_D_kernel<<<blocks, threads, 0, stream>>>(O, dO, d_D, seq_len, head_dim);
    #endif
    // Initialize dQ to 0
    cudaMemset(dQ, 0, seq_len * head_dim * sizeof(float));
    
    // Launch backward kernel
    int num_col_blocks = div_up(seq_len, Bc);
    dim3 grid = new_dim3(num_col_blocks);
    dim3 block = new_dim3(threads_per_block);
    
    int smem_size = 4 * Bc * head_dim * sizeof(float); // K, V, dK, dV in shared memory
    
    if (head_dim <= 64) {
         #ifndef __INTELLISENSE__

        flash_attention_2_backward_kernel<Br, Bc, 64, num_warps><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, dO, L, d_D, dQ, dK, dV, seq_len, head_dim, softmax_scale
        );
        #endif
    } else {
        #ifndef __INTELLISENSE__
        flash_attention_2_backward_kernel<Br, Bc, 128, num_warps><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, dO, L, d_D, dQ, dK, dV, seq_len, head_dim, softmax_scale
        );
        #endif
    }
    
    cudaFree(d_D);
}
