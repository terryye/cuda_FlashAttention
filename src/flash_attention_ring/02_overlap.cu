/**
 create a test that simulates the Ring Attention pattern - overlapping computation with K,V block transfers with streams
 */
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "util/nccl_utils.h"

 // Simple kernel to simulate attention computation
__global__ void simulate_attention_kernel(float* data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        // Simulate computation
        for (int i = 0; i < iterations; i++) {
            val = val * 1.0001f + 0.001f;
        }
        data[idx] = val;
    }
}

int main(int argc, char* argv[]) {

    int rank, nranks;
    ncclComm_t comm;

    // Initialize everything in one call
    if (init_mpi_nccl(argc, argv, &rank, &nranks, &comm) != 0) {
        fprintf(stderr, "Initialization failed\n");
        return 1;
    }

    // Create streams
    cudaStream_t compute_stream, comm_stream;
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    CHECK_CUDA(cudaStreamCreate(&comm_stream));

    // Allocate buffers (simulating K,V blocks) - smaller size for testing
    const int block_size = 1024; // Smaller for initial test
    float* d_kv_current;  // Current K,V block for computation
    float* d_kv_next;     // Next K,V block being received

    CHECK_CUDA(cudaMalloc(&d_kv_current, block_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kv_next, block_size * sizeof(float)));

    // Initialize data with rank-specific values, each block filled with rank + 1
    std::vector<float> h_data(block_size, rank + 1.0f);
    CHECK_CUDA(cudaMemcpy(d_kv_current, h_data.data(), block_size * sizeof(float),
        cudaMemcpyHostToDevice));

    int next_rank = (rank + 1) % nranks;
    int prev_rank = (rank - 1 + nranks) % nranks;

    if (rank == 0) {
        std::cout << "Starting ring communication pattern..." << std::endl;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Ring attention simulation
    for (int step = 0; step < nranks; step++) {
        //ring exchange on comm_stream

        //send d_kv_current, recv to d_kv_next
        ring_exchange(d_kv_current, d_kv_next, block_size, next_rank, prev_rank, comm);

        // Simulate computation on current block (can overlap with communication)
        simulate_attention_kernel << <(block_size + 255) / 256, 256, 0, compute_stream >> > (
            d_kv_current, block_size, 100
            );

        // Wait for both streams
        CHECK_CUDA(cudaStreamSynchronize(comm_stream));
        CHECK_CUDA(cudaStreamSynchronize(compute_stream));

        // Swap buffers for next iteration d_kv_current <-> d_kv_next
        float* temp = d_kv_current;
        d_kv_current = d_kv_next;
        d_kv_next = temp;

        // Verify received data
        std::vector<float> h_check(1);
        CHECK_CUDA(cudaMemcpy(h_check.data(), d_kv_current, sizeof(float),
            cudaMemcpyDeviceToHost));

        std::cout << "Rank " << rank << ", Step " << step
            << ": Received block starting with " << h_check[0]
            << " (expected from pre rank " << prev_rank << ")" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    //global synchronization point that blocks all MPI processes until every process in the communicator (MPI_COMM_WORLD) reaches this call. It ensures that all ranks have completed their work before proceeding.
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_kv_current));
    CHECK_CUDA(cudaFree(d_kv_next));
    CHECK_CUDA(cudaStreamDestroy(compute_stream));
    CHECK_CUDA(cudaStreamDestroy(comm_stream));
    CHECK_NCCL(ncclCommDestroy(comm));

    MPI_Finalize();

    std::cout << "Rank " << rank << ": Overlap test completed!" << std::endl;
    return 0;
}