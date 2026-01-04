#pragma once
#define WARP_SIZE 32

#include "util/cuda_helper.h"
#include "util/attention_helper.h"
#include <stdio.h>
#include <math.h>
#include <cfloat>
#include <math.h>
#include <float.h>


template <int Br, int Bc, int d_max, int num_warps>
__global__ void flash_attention_2_forward_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* L,
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
    //const int rows_per_thread = (rows_per_warp + WARP_SIZE - 1) / WARP_SIZE;
    const int warp_row_start = warp_id * rows_per_warp;

    // Global row indices for this thread block
    const int global_row_start = row_block_idx * Br;
    const int global_row_end = min(global_row_start + Br, N);

    // Registers for Q tile (each warp loads its portion)
    float Q_reg[rows_per_warp][cols_per_thread];  // +1 to handle when d is not divisible by WARP_SIZE

    // Load Q tile for this warp
    // each warp loads its assigned rows, each thread loads its assigned columns    
    load_Q_tile<rows_per_warp, cols_per_thread>(
        Q, Q_reg, global_row_start, global_row_end,
        warp_row_start, lane_id, d
    );
    // Ensure all threads in warp have loaded their Q values
    __syncwarp();

    // there is a little redundancy here since each warp only process one row at a time, but it's easier to manage this way.
    float l[rows_per_warp];
    float m[rows_per_warp];

    // Each thread only needs to store its portion of O_acc, the same position as Q_reg; actually we can combine them later
    float O_acc[rows_per_warp][cols_per_thread];
    for (int i = 0; i < rows_per_warp; i++) {
        for (int j = 0; j < cols_per_thread; j++) {
            O_acc[i][j] = 0.0f;
        }
        l[i] = 0.0f; // initialize l for each row
        m[i] = -INFINITY; // initialize m for each row
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
        // Process this K/V block
        if (global_row_start + warp_row_start < global_row_end) {
            process_kv_block<Bc, cols_per_thread>(K_smem, V_smem, Q_reg, O_acc, m, l,
                rows_per_warp, actual_Bc, d,
                softmax_scale, lane_id);
        }
        __syncwarp();
    }

    //debug show O_acc
    // __syncthreads();
    // for(int local_row=0; local_row<rows_per_warp; local_row++){
    //     for (int col_offset=0; col_offset<cols_per_thread; col_offset++){
    //         int col = lane_id + col_offset * WARP_SIZE;
    //         if(col < d){
    //              printf("Warp %d, Local Row %d, Col %d: Final O_acc=%f\n", warp_id, local_row, col, O_acc[local_row][col_offset]);
    //          }
    //     }
    // }
    // __syncthreads();

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
    const int Br = 32;  // Row block size
    const int Bc = 32;  // Column block size
    const int num_warps = 2;
    const int threads_per_block = num_warps * WARP_SIZE;
    const int d_max = 64;  // Maximum supported head dimension

    assert(head_dim <= d_max); // Constraint for this implementation
    assert(Br % num_warps == 0);

    int num_row_blocks = div_up(seq_len, Br);



    dim3 grid = dim3(num_row_blocks, 1, 1);
    dim3 block = dim3(threads_per_block, 1, 1);

    int smem_size = 2 * Bc * head_dim * sizeof(float); // K_smem + V_smem

    if (head_dim <= 64) {
        flash_attention_2_forward_kernel<Br, Bc, 64, num_warps> << <grid, block, smem_size >> > (
            Q, K, V, O, L, seq_len, head_dim, softmax_scale
            );
    }
    else {
        flash_attention_2_forward_kernel<Br, Bc, 128, num_warps> << <grid, block, smem_size >> > (
            Q, K, V, O, L, seq_len, head_dim, softmax_scale
            );
    }
}
