#pragma once

#include <stdio.h>
#include <math.h>
#include <cfloat>
#include <float.h>
#include <assert.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Helper function to get number of blocks
__host__ __device__ inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// Warp-level reduction for sum
__device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, mask);
    }
    return val;
}

// Warp-level reduction for max
__device__ float warp_reduce_max(float val) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, mask));
    }
    return val;
}

/**
 * FlashAttention-2 Backward Pass Kernel
 * Based on Algorithm 5 in the FlashAttention-2 paper
 *
 * Section 3.2: Parallelism
 * - Each thread block handles Br rows of the backward computation
 * - Loops over Bc columns for K, V tiles
 *
 * Section 3.3: Work Partitioning between Warps
 * - Warps within a block work on different rows
 * - Each warp computes partial gradients for its assigned rows
 */
template <int Br, int Bc, int d_max, int num_warps>
__global__ void flash_attention_2_backward_kernel(
    const float* Q,      // [N, d]
    const float* K,      // [N, d]
    const float* V,      // [N, d]
    const float* O,      // [N, d] - output from forward pass
    const float* L,      // [N] - logsumexp from forward pass
    const float* dO,     // [N, d] - gradient of output
    float* dQ,           // [N, d] - gradient of Q (output)
    float* dK,           // [N, d] - gradient of K (output)
    float* dV,           // [N, d] - gradient of V (output)
    const int N,
    const int d,
    const float softmax_scale
) {
    // Thread block handles one block row of attention matrix
    const int row_block_idx = blockIdx.x;

    // Shared memory layout:
    // K_smem[Bc][d], V_smem[Bc][d], dK_smem[Bc][d], dV_smem[Bc][d]
    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = smem + Bc * d;
    float* dK_smem = smem + 2 * Bc * d;
    float* dV_smem = smem + 3 * Bc * d;

    // Thread and warp identifiers
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_threads = blockDim.x;

    // Each warp handles different rows
    constexpr int rows_per_warp = Br / num_warps;
    constexpr int cols_per_thread = (d_max + WARP_SIZE - 1) / WARP_SIZE;

    // Global row indices for this thread block
    const int global_row_start = row_block_idx * Br;
    const int global_row_end = min(global_row_start + Br, N);
    const int warp_row_start = warp_id * rows_per_warp;

    // Registers for Q, O, dO tiles (each warp loads its portion)
    float Q_reg[rows_per_warp][cols_per_thread];
    float O_reg[rows_per_warp][cols_per_thread];
    float dO_reg[rows_per_warp][cols_per_thread];
    float dQ_acc[rows_per_warp][cols_per_thread];

    // Load Q, O, dO tiles and compute D = rowsum(dO * O) for this warp
    float D[rows_per_warp];

    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        int global_row = global_row_start + warp_row_start + local_row;
        D[local_row] = 0.0f;

        if (global_row < global_row_end) {
            for (int col_offset = 0; col_offset < cols_per_thread; col_offset++) {
                int col = lane_id + col_offset * WARP_SIZE;
                if (col < d) {
                    Q_reg[local_row][col_offset] = Q[global_row * d + col];
                    O_reg[local_row][col_offset] = O[global_row * d + col];
                    dO_reg[local_row][col_offset] = dO[global_row * d + col];
                    D[local_row] += dO_reg[local_row][col_offset] * O_reg[local_row][col_offset];
                }
                else {
                    Q_reg[local_row][col_offset] = 0.0f;
                    O_reg[local_row][col_offset] = 0.0f;
                    dO_reg[local_row][col_offset] = 0.0f;
                }
                dQ_acc[local_row][col_offset] = 0.0f;
            }
            // Reduce D across the warp
            D[local_row] = warp_reduce_sum(D[local_row]);
        }
    }

    __syncwarp();

    // Main loop over K, V blocks (columns)
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

        // Initialize dK_smem and dV_smem to zero
        for (int idx = tid; idx < actual_Bc * d; idx += num_threads) {
            dK_smem[idx] = 0.0f;
            dV_smem[idx] = 0.0f;
        }

        __syncthreads();

        // Each warp computes for its assigned rows
        for (int local_row = 0; local_row < rows_per_warp; local_row++) {
            int global_row = global_row_start + warp_row_start + local_row;

            if (global_row < global_row_end) {
                // Compute S = Q @ K^T for this row
                float S_row[Bc];
                float P_row[Bc];

                for (int col = 0; col < actual_Bc; col++) {
                    float dot = 0.0f;
                    for (int k = 0; k < d; k++) {
                        int q_col = k / WARP_SIZE;
                        if (k % WARP_SIZE == lane_id) {
                            dot += Q_reg[local_row][q_col] * K_smem[col * d + k];
                        }
                    }
                    dot = warp_reduce_sum(dot);
                    S_row[col] = dot * softmax_scale;
                }

                // Compute P = softmax(S) using precomputed L
                float l_val = L[global_row];
                for (int col = 0; col < actual_Bc; col++) {
                    P_row[col] = expf(S_row[col] - l_val);
                }

                // Compute dP = dO @ V^T
                float dP_row[Bc];
                for (int col = 0; col < actual_Bc; col++) {
                    float dot = 0.0f;
                    for (int k = 0; k < d; k++) {
                        int do_col = k / WARP_SIZE;
                        if (k % WARP_SIZE == lane_id) {
                            dot += dO_reg[local_row][do_col] * V_smem[col * d + k];
                        }
                    }
                    dot = warp_reduce_sum(dot);
                    dP_row[col] = dot;
                }

                // Compute dS = P * (dP - D)
                float dS_row[Bc];
                for (int col = 0; col < actual_Bc; col++) {
                    dS_row[col] = P_row[col] * (dP_row[col] - D[local_row]);
                }

                // Accumulate dQ += dS @ K * scale
                for (int k = 0; k < d; k++) {
                    float sum = 0.0f;
                    for (int col = 0; col < actual_Bc; col++) {
                        sum += dS_row[col] * K_smem[col * d + k];
                    }
                    int q_col = k / WARP_SIZE;
                    if (k % WARP_SIZE == lane_id) {
                        dQ_acc[local_row][q_col] += sum * softmax_scale;
                    }
                }

                // Accumulate dV += P^T @ dO (using atomic adds to shared memory)
                for (int col = 0; col < actual_Bc; col++) {
                    for (int k = lane_id; k < d; k += WARP_SIZE) {
                        int do_col = k / WARP_SIZE;
                        atomicAdd(&dV_smem[col * d + k], P_row[col] * dO_reg[local_row][do_col]);
                    }
                }

                // Accumulate dK += dS^T @ Q * scale (using atomic adds to shared memory)
                for (int col = 0; col < actual_Bc; col++) {
                    for (int k = lane_id; k < d; k += WARP_SIZE) {
                        int q_col = k / WARP_SIZE;
                        atomicAdd(&dK_smem[col * d + k], dS_row[col] * Q_reg[local_row][q_col] * softmax_scale);
                    }
                }
            }
        }

        __syncthreads();

        // Write dK and dV from shared memory to global memory
        for (int idx = tid; idx < actual_Bc * d; idx += num_threads) {
            atomicAdd(&dK[col_block_start_idx + idx], dK_smem[idx]);
            atomicAdd(&dV[col_block_start_idx + idx], dV_smem[idx]);
        }
    }

    // Write dQ to global memory
    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        int global_row = global_row_start + warp_row_start + local_row;
        if (global_row < global_row_end) {
            for (int col_offset = 0; col_offset < cols_per_thread; col_offset++) {
                int col = lane_id + col_offset * WARP_SIZE;
                if (col < d) {
                    dQ[global_row * d + col] = dQ_acc[local_row][col_offset];
                }
            }
        }
    }
}

// Host wrapper function
void flash_attention_2_backward(
    const float* Q,
    const float* K,
    const float* V,
    const float* O,
    const float* L,
    const float* dO,
    float* dQ,
    float* dK,
    float* dV,
    int seq_len,
    int head_dim,
    float softmax_scale
) {
    // Configure kernel parameters (matching forward pass)
    const int Br = 32;  // Row block size
    const int Bc = 32;  // Column block size
    const int num_warps = 4;
    const int threads_per_block = num_warps * WARP_SIZE;
    const int d_max = 128;

    assert(head_dim <= d_max);
    assert(Br % num_warps == 0);

    int num_row_blocks = div_up(seq_len, Br);

    dim3 grid = dim3(num_row_blocks, 1, 1);
    dim3 block = dim3(threads_per_block, 1, 1);

    // Shared memory: K_smem + V_smem + dK_smem + dV_smem
    int smem_size = 4 * Bc * head_dim * sizeof(float);

    // Initialize dK and dV to zero
    cudaMemset(dK, 0, seq_len * head_dim * sizeof(float));
    cudaMemset(dV, 0, seq_len * head_dim * sizeof(float));

    if (head_dim <= 64) {
#ifndef __INTELLISENSE__
        flash_attention_2_backward_kernel<32, 32, 64, 4> << <grid, block, smem_size >> > (
            Q, K, V, O, L, dO, dQ, dK, dV, seq_len, head_dim, softmax_scale
            );
#endif
    }
    else {
#ifndef __INTELLISENSE__
        flash_attention_2_backward_kernel<32, 32, 128, 4> << <grid, block, smem_size >> > (
            Q, K, V, O, L, dO, dQ, dK, dV, seq_len, head_dim, softmax_scale
            );
#endif
    }
}
