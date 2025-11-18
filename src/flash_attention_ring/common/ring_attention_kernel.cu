// ring_attention_kernel.cu V1 (updated)
#pragma once

#define WARP_SIZE 32

#include "util/nccl_utils.h"
#include "util/cuda_helper.h"
#include "util/attention_helper.h"
#include <vector>


// Complete Ring Attention kernel
template <int Br, int Bc, int d_max, int num_warps>
__global__ void ring_attention_forward_kernel(
    const float* Q,       
    const float* K,       
    const float* V,       
    float* O,            
    float* L,            
    float* M,            
    const int Q_seq_len, 
    const int KV_seq_len,
    const int d,
    const float softmax_scale,
    const bool last_step
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
    const int global_row_end = min(global_row_start + Br, Q_seq_len);
    
    // Load Q tile
    float Q_reg[rows_per_warp][cols_per_thread];
    
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
    
    // Load l,m from global memory(initialized to 0/-inf already )
    for (int i = 0; i < rows_per_warp; i++) {
        l[i] = L[global_row_start + warp_row_start + i];
        m[i] = M[global_row_start + warp_row_start + i];
    }

    // Each thread only needs to store its portion of O_acc, the same position as Q_reg; actually we can combine them later
    float O_acc[rows_per_warp][cols_per_thread];
    //Load O_acc , following the same pattern as Q_reg
    load_Q_tile<rows_per_warp, cols_per_thread>(
        O, O_acc, global_row_start, global_row_end, 
        warp_row_start, lane_id, d
    );

    // Main loop over K,V blocks
    const int num_blocks = div_up(KV_seq_len, Bc);
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int col_block_start = block_idx * Bc;
        const int col_block_end = min(col_block_start + Bc, KV_seq_len);
        const int actual_Bc = col_block_end - col_block_start;
        
        __syncthreads();
        
        // Load K and V tiles to shared memory
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
    
    // Write final results
    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        int global_row = global_row_start + warp_row_start + local_row;
        if (global_row < global_row_end) {
            if (last_step) {
                // Final step: normalize output
                float inv_l = 1.0f / l[local_row];
                for (int col_offset = 0; col_offset < cols_per_thread; col_offset++) {
                    int col = lane_id + col_offset * WARP_SIZE;
                    if (col < d) {
                        O[global_row * d + col] = O_acc[local_row][col_offset] * inv_l;
                    }
                }
                if (lane_id == 0) {
                    L[global_row] = m[local_row] + logf(l[local_row]);
                }
            } else {
                // Intermediate step: save unnormalized values
                for (int col_offset = 0; col_offset < cols_per_thread; col_offset++) {
                    int col = lane_id + col_offset * WARP_SIZE;
                    if (col < d) {
                        O[global_row * d + col] = O_acc[local_row][col_offset];
                    }
                }
                if (lane_id == 0) {
                    M[global_row] = m[local_row];
                    L[global_row] = l[local_row];
                }
            }
        }
    }
}

// Ring Attention forward pass   
void ring_attention_forward(
    const float* Q_local,
    float* K_local,
    float* V_local,
    float* O_local,
    float* L_local,
    int total_seq_len,
    int local_seq_len,
    int head_dim,
    float softmax_scale,
    ncclComm_t comm,
    int rank,
    int nranks
) {
    // Allocate buffers
    float *K_mem, *V_mem;
    float *M_local; // Max statistics
    size_t kv_size = local_seq_len * head_dim * sizeof(float);
    CHECK_CUDA(cudaMalloc(&K_mem, kv_size));
    CHECK_CUDA(cudaMalloc(&V_mem, kv_size));
    CHECK_CUDA(cudaMalloc(&M_local, local_seq_len * sizeof(float)));
    
    float * K_recv = K_mem;
    float * V_recv = V_mem;
    // Initialize
    CHECK_CUDA(cudaMemset(O_local, 0, kv_size));
    CHECK_CUDA(cudaMemset(L_local, 0, local_seq_len * sizeof(float)));
    //initialize M_local to -infinity
    init_array<<<div_up(local_seq_len, 256), 256>>>(M_local, local_seq_len, -INFINITY);
    
    // Kernel configuration
    const int Br = 64;
    const int Bc = 64;
    const int num_warps = 4;
    const int threads_per_block = num_warps * WARP_SIZE;
    int num_row_blocks = div_up(local_seq_len, Br);
    int smem_size = 2 * Bc * head_dim * sizeof(float);
    
    dim3 grid(num_row_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    
    // Working pointers
    float* K_current = K_local;
    float* V_current = V_local;
    
    int next_rank = (rank + 1) % nranks;
    int prev_rank = (rank - 1 + nranks) % nranks;


    cudaStream_t compute_stream, comm_stream;
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    CHECK_CUDA(cudaStreamCreate(&comm_stream));
    
    // Process all K,V blocks in ring fashion
    for (int step = 0; step < nranks; step++) {
        int kv_block_idx = (rank - step + nranks) % nranks;
        bool last_step = (step == nranks - 1);
        
        printf("Rank %d, Step %d: Processing K,V block from rank %d\n", 
               rank, step, kv_block_idx);
        
        // Call kernel with last_step flag for inline normalization
        ring_attention_forward_kernel<Br, Bc, 64, num_warps><<<grid, block, smem_size, compute_stream>>>(
            Q_local, K_current, V_current, O_local, L_local, M_local,
            local_seq_len, local_seq_len, head_dim, softmax_scale, 
            last_step
        );
        CHECK_CUDA(cudaGetLastError());

        // Ring exchange K,V blocks (except on last step)
        if (step < nranks - 1) {
            ring_exchange_kv(
                (float *)K_current, (float *)K_recv,
                (float *)V_current, (float *)V_recv,
                local_seq_len * head_dim,
                next_rank, prev_rank, comm, comm_stream);

            CHECK_CUDA(cudaDeviceSynchronize());
            // // Wait for both streams
            // CHECK_CUDA(cudaStreamSynchronize(comm_stream));
            // CHECK_CUDA(cudaStreamSynchronize(compute_stream));
            
            swap_float_ptrs(&K_current, &K_recv);
            swap_float_ptrs(&V_current, &V_recv);
        }



    }s
    
    // No separate normalization needed - it's done inline!
    
    // Cleanup
    CHECK_CUDA(cudaFree(K_mem));
    CHECK_CUDA(cudaFree(V_mem));
    CHECK_CUDA(cudaFree(M_local));
}