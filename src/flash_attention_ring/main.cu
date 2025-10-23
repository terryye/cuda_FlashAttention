// ring_attention_multi_gpu.cu - Optimized with Host Staging
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cfloat>
#include "../../util/util.cuh"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Helper functions
__host__ __device__ inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, mask);
    }
    return val;
}

// Ring Attention kernel - processes local Q block against rotating KV blocks
template <int Br, int Bc, int d_max, int num_warps>
__global__ void ring_attention_forward_kernel(
    const float* __restrict__ Q_local,      // Local Q block for this GPU
    const float* __restrict__ K_local,      // Local K block (will rotate)
    const float* __restrict__ V_local,      // Local V block (will rotate)
    float* __restrict__ O_local,            // Local output
    float* __restrict__ L_local,            // Local logsumexp
    float* __restrict__ m_local,            // Local max values for numerically stable softmax
    float* __restrict__ l_local,            // Local sum values for numerically stable softmax
    const int local_seq_len,                // Sequence length per GPU
    const int d,                            // Head dimension
    const float softmax_scale,
    const int num_kv_blocks,                // Total number of KV blocks to process
    const int current_kv_block              // Which KV block is currently in local memory
) {
    const int row_block_idx = blockIdx.x;
    
    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = smem + Bc * d;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_threads = blockDim.x;
    
    const int rows_per_warp = Br / num_warps;
    const int warp_row_start = warp_id * rows_per_warp;
    
    const int global_row_start = row_block_idx * Br;
    const int global_row_end = min(global_row_start + Br, local_seq_len);
    
    // Load Q for this warp (only once, Q doesn't rotate)
    float Q_reg[rows_per_warp][d_max];
    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        int global_row = global_row_start + warp_row_start + local_row;
        if (global_row < global_row_end) {
            for (int col = lane_id; col < d; col += WARP_SIZE) {
                Q_reg[local_row][col] = Q_local[global_row * d + col];
            }
        } else {
            for (int col = lane_id; col < d; col += WARP_SIZE) {
                Q_reg[local_row][col] = 0.0f;
            }
        }
    }
    
    __syncwarp();
    
    // Load current statistics if not first iteration
    float m[rows_per_warp];
    float l[rows_per_warp];
    float O_acc[rows_per_warp][d_max];
    
    if (current_kv_block == 0) {
        // First iteration - initialize
        #pragma unroll
        for (int i = 0; i < rows_per_warp; i++) {
            m[i] = -INFINITY;
            l[i] = 0.0f;
            #pragma unroll
            for (int j = 0; j < d_max; j++) {
                O_acc[i][j] = 0.0f;
            }
        }
    } else {
        // Load existing statistics and output
        for (int local_row = 0; local_row < rows_per_warp; local_row++) {
            int global_row = global_row_start + warp_row_start + local_row;
            if (global_row < global_row_end) {
                if (lane_id == 0) {
                    m[local_row] = m_local[global_row];
                    l[local_row] = l_local[global_row];
                }
                for (int col = lane_id; col < d; col += WARP_SIZE) {
                    O_acc[local_row][col] = O_local[global_row * d + col];
                }
            }
        }
        __syncwarp();
    }
    
    // Process current KV block
    const int num_blocks = div_up(local_seq_len, Bc);
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int col_block_start = block_idx * Bc;
        const int col_block_end = min(col_block_start + Bc, local_seq_len);
        const int actual_Bc = col_block_end - col_block_start;
        
        __syncthreads();
        
        // Load K and V tiles to shared memory
        for (int idx = tid; idx < actual_Bc * d; idx += num_threads) {
            int col_in_block = idx / d;
            int d_idx = idx % d;
            int global_col = col_block_start + col_in_block;
            K_smem[col_in_block * d + d_idx] = K_local[global_col * d + d_idx];
            V_smem[col_in_block * d + d_idx] = V_local[global_col * d + d_idx];
        }
        
        __syncthreads();
        
        // Compute attention scores
        float S_row[Bc];
        
        for (int local_row = 0; local_row < rows_per_warp; local_row++) {
            int global_row = global_row_start + warp_row_start + local_row;
            if (global_row < global_row_end) {
                // Compute Q @ K^T
                for (int col = 0; col < actual_Bc; col++) {
                    float dot = 0.0f;
                    for (int k = lane_id; k < d; k += WARP_SIZE) {
                        dot += Q_reg[local_row][k] * K_smem[col * d + k];
                    }
                    dot = warp_reduce_sum(dot);
                    S_row[col] = dot * softmax_scale;
                }
                
                // Online softmax
                float row_max = -INFINITY;
                for (int col = 0; col < actual_Bc; col++) {
                    row_max = fmaxf(row_max, S_row[col]);
                }
                
                float m_new = fmaxf(m[local_row], row_max);
                float exp_diff = expf(m[local_row] - m_new);
                
                float row_sum = 0.0f;
                #pragma unroll
                for (int col = 0; col < actual_Bc; col++) {
                    S_row[col] = expf(S_row[col] - m_new);
                    row_sum += S_row[col];
                }
                
                // Update output with rescaling
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
                
                // Update statistics
                l[local_row] = exp_diff * l[local_row] + row_sum;
                m[local_row] = m_new;
            }
        }
    }
    
    // Write back statistics and output
    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        int global_row = global_row_start + warp_row_start + local_row;
        if (global_row < global_row_end) {
            // Store intermediate statistics
            if (lane_id == 0) {
                m_local[global_row] = m[local_row];
                l_local[global_row] = l[local_row];
            }
            
            // Store output (will be normalized after all KV blocks processed)
            for (int col = lane_id; col < d; col += WARP_SIZE) {
                O_local[global_row * d + col] = O_acc[local_row][col];
            }
            
            // If this is the last KV block, normalize and compute logsumexp
            if (current_kv_block == num_kv_blocks - 1) {
                float inv_l = 1.0f / l[local_row];
                for (int col = lane_id; col < d; col += WARP_SIZE) {
                    O_local[global_row * d + col] *= inv_l;
                }
                if (lane_id == 0) {
                    L_local[global_row] = m[local_row] + logf(l[local_row]);
                }
            }
        }
    }
}

// Main function
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Get device count before MPI_Init
    int deviceCount;
    CHK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set CUDA device
    int device = rank % deviceCount;
    CHK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, device));
    
    printf("Rank %d/%d using GPU %d (%s)\n", rank, size, device, prop.name);

    // Create streams for overlapping
    cudaStream_t compute_stream, h2d_stream, d2h_stream;
    CHK(cudaStreamCreate(&compute_stream));
    CHK(cudaStreamCreate(&h2d_stream));
    CHK(cudaStreamCreate(&d2h_stream));

    // Problem parameters
    const int total_seq_len = 2048;  // Total sequence length
    const int head_dim = 64;
    const float softmax_scale = 1.0f / sqrtf(head_dim);
    
    // Each GPU handles a portion of the sequence
    const int local_seq_len = total_seq_len / size;
    //const int local_offset = rank * local_seq_len;
    
    // Allocate host memory
    size_t local_size = local_seq_len * head_dim;
    float *h_Q = (float*)malloc(local_size * sizeof(float));
    float *h_K = (float*)malloc(local_size * sizeof(float));
    float *h_V = (float*)malloc(local_size * sizeof(float));
    float *h_O = (float*)malloc(local_size * sizeof(float));
    float *h_L = (float*)malloc(local_seq_len * sizeof(float));
    
    // Allocate PINNED host memory for MPI communication
    float *h_K_send, *h_V_send, *h_K_recv, *h_V_recv;
    CHK(cudaMallocHost(&h_K_send, local_size * sizeof(float)));
    CHK(cudaMallocHost(&h_V_send, local_size * sizeof(float)));
    CHK(cudaMallocHost(&h_K_recv, local_size * sizeof(float)));
    CHK(cudaMallocHost(&h_V_recv, local_size * sizeof(float)));
    
    // Initialize data
    srand(42 + rank);
    for (int i = 0; i < local_size; i++) {
        h_Q[i] = (rand() % 1000) / 1000.0f - 0.5f;
        h_K[i] = (rand() % 1000) / 1000.0f - 0.5f;
        h_V[i] = (rand() % 1000) / 1000.0f - 0.5f;
    }
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    float *d_K_current, *d_V_current;  // Current KV blocks being processed
    float *d_K_next, *d_V_next;        // Next KV blocks (double buffering)
    float *d_m, *d_l;  // Statistics
    
    CHK(cudaMalloc(&d_Q, local_size * sizeof(float)));
    CHK(cudaMalloc(&d_K, local_size * sizeof(float)));
    CHK(cudaMalloc(&d_V, local_size * sizeof(float)));
    CHK(cudaMalloc(&d_K_current, local_size * sizeof(float)));
    CHK(cudaMalloc(&d_V_current, local_size * sizeof(float)));
    CHK(cudaMalloc(&d_K_next, local_size * sizeof(float)));
    CHK(cudaMalloc(&d_V_next, local_size * sizeof(float)));
    CHK(cudaMalloc(&d_O, local_size * sizeof(float)));
    CHK(cudaMalloc(&d_L, local_seq_len * sizeof(float)));
    CHK(cudaMalloc(&d_m, local_seq_len * sizeof(float)));
    CHK(cudaMalloc(&d_l, local_seq_len * sizeof(float)));
    
    // Copy initial data to device
    CHK(cudaMemcpy(d_Q, h_Q, local_size * sizeof(float), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_K, h_K, local_size * sizeof(float), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_V, h_V, local_size * sizeof(float), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_K_current, d_K, local_size * sizeof(float), cudaMemcpyDeviceToDevice));
    CHK(cudaMemcpy(d_V_current, d_V, local_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Initialize arrays
    CHK(cudaMemset(d_O, 0, local_size * sizeof(float)));
    CHK(cudaMemset(d_m, 0, local_seq_len * sizeof(float)));
    CHK(cudaMemset(d_l, 0, local_seq_len * sizeof(float)));
    
    // Configure kernel
    const int Br = 64;
    const int Bc = 64;
    const int num_warps = 4;
    const int threads_per_block = num_warps * WARP_SIZE;
    //const int d_max = 128;
    
    int num_row_blocks = div_up(local_seq_len, Br);
    dim3 grid(num_row_blocks);
    dim3 block(threads_per_block);
    int smem_size = 2 * Bc * head_dim * sizeof(float);
    
    printf("Rank %d: Starting Ring Attention with host staging\n", rank);
    
    // Timing
    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));
    CHK(cudaEventRecord(start));
    
    // Ring Attention main loop with optimized host staging
    for (int step = 0; step < size; step++) {
        // Launch kernel to process current KV block
        if (head_dim <= 64) {
            ring_attention_forward_kernel<Br, Bc, 64, num_warps><<<grid, block, smem_size, compute_stream>>>(
                d_Q, d_K_current, d_V_current, d_O, d_L, d_m, d_l,
                local_seq_len, head_dim, softmax_scale, size, step
            );
        } else {
            ring_attention_forward_kernel<Br, Bc, 128, num_warps><<<grid, block, smem_size, compute_stream>>>(
                d_Q, d_K_current, d_V_current, d_O, d_L, d_m, d_l,
                local_seq_len, head_dim, softmax_scale, size, step
            );
        }
        
        // Check for kernel errors
        CHK(cudaGetLastError());
        
        if (step < size - 1) {
            // Start async copy to host for sending
            CHK(cudaMemcpyAsync(h_K_send, d_K_current, local_size * sizeof(float), 
                                cudaMemcpyDeviceToHost, d2h_stream));
            CHK(cudaMemcpyAsync(h_V_send, d_V_current, local_size * sizeof(float), 
                                cudaMemcpyDeviceToHost, d2h_stream));
            
            // Wait for D2H to complete before MPI
            CHK(cudaStreamSynchronize(d2h_stream));
            
            // MPI communication with host buffers
            MPI_Request send_k_req, send_v_req, recv_k_req, recv_v_req;
            int next_rank = (rank + 1) % size;
            int prev_rank = (rank - 1 + size) % size;
            
            MPI_Isend(h_K_send, local_size, MPI_FLOAT, next_rank, 0, MPI_COMM_WORLD, &send_k_req);
            MPI_Isend(h_V_send, local_size, MPI_FLOAT, next_rank, 1, MPI_COMM_WORLD, &send_v_req);
            MPI_Irecv(h_K_recv, local_size, MPI_FLOAT, prev_rank, 0, MPI_COMM_WORLD, &recv_k_req);
            MPI_Irecv(h_V_recv, local_size, MPI_FLOAT, prev_rank, 1, MPI_COMM_WORLD, &recv_v_req);
            
            // Wait for MPI to complete
            MPI_Wait(&recv_k_req, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_v_req, MPI_STATUS_IGNORE);
            
            // Start async copy to device for next iteration
            CHK(cudaMemcpyAsync(d_K_next, h_K_recv, local_size * sizeof(float), 
                                cudaMemcpyHostToDevice, h2d_stream));
            CHK(cudaMemcpyAsync(d_V_next, h_V_recv, local_size * sizeof(float), 
                                cudaMemcpyHostToDevice, h2d_stream));
            
            // Wait for sends to complete
            MPI_Wait(&send_k_req, MPI_STATUS_IGNORE);
            MPI_Wait(&send_v_req, MPI_STATUS_IGNORE);
            
            // Wait for H2D to complete
            CHK(cudaStreamSynchronize(h2d_stream));
            
            // Swap buffers for next iteration
            float *temp = d_K_current;
            d_K_current = d_K_next;
            d_K_next = temp;
            
            temp = d_V_current;
            d_V_current = d_V_next;
            d_V_next = temp;
        }
        
        // Ensure kernel completes before next iteration
        CHK(cudaStreamSynchronize(compute_stream));
    }
    
    CHK(cudaEventRecord(stop));
    CHK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copy result back
    CHK(cudaMemcpy(h_O, d_O, local_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(h_L, d_L, local_seq_len * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Rank %d: Processed %d tokens in %.2f ms on GPU %d\n",
           rank, local_seq_len, milliseconds, device);
    
    // Gather results at rank 0
    if (rank == 0) {
        float *h_O_global = (float*)malloc(total_seq_len * head_dim * sizeof(float));
        MPI_Gather(h_O, local_size, MPI_FLOAT, h_O_global, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        printf("Ring Attention completed successfully!\n");
        free(h_O_global);
    } else {
        MPI_Gather(h_O, local_size, MPI_FLOAT, NULL, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    
    // Cleanup
    CHK(cudaStreamDestroy(compute_stream));
    CHK(cudaStreamDestroy(h2d_stream));
    CHK(cudaStreamDestroy(d2h_stream));
    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
    CHK(cudaFree(d_Q));
    CHK(cudaFree(d_K));
    CHK(cudaFree(d_V));
    CHK(cudaFree(d_K_current));
    CHK(cudaFree(d_V_current));
    CHK(cudaFree(d_K_next));
    CHK(cudaFree(d_V_next));
    CHK(cudaFree(d_O));
    CHK(cudaFree(d_L));
    CHK(cudaFree(d_m));
    CHK(cudaFree(d_l));
    CHK(cudaFreeHost(h_K_send));
    CHK(cudaFreeHost(h_V_send));
    CHK(cudaFreeHost(h_K_recv));
    CHK(cudaFreeHost(h_V_recv));
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    free(h_L);
    
    MPI_Finalize();
    return 0;
}