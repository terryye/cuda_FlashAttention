// test_ring_attention_simple.cu
#include <iostream>
#include <vector>
#include "util/nccl_utils.h"
#include "util/naive_attention.h"
#include "./common/flash_attention_kernel.cu"

int main(int argc, char* argv[]) {
    int rank, nranks;
    ncclComm_t comm;

    // Initialize MPI and NCCL
    if (init_mpi_nccl(argc, argv, &rank, &nranks, &comm) != 0) {
        return 1;
    }

    // Test parameters
    const int seq_len = 5096;  // Total sequence length
    const int head_dim = 64;
    const float softmax_scale = 1.0f;

    // Reference outputs
    std::vector<float> h_O_naive(seq_len * head_dim);
    std::vector<float> h_O_flash(seq_len * head_dim);

    // Only run on rank 0 for reference computation
    if (rank == 0) {
        std::cout << "=== Test Case ===" << std::endl;

        // Create test data
        std::vector<float> h_Q, h_K, h_V;
        create_simple_test_data(h_Q, h_K, h_V, seq_len, head_dim);

        print_matrix("Q", h_Q.data(), seq_len, head_dim);
        print_matrix("K", h_K.data(), seq_len, head_dim);
        print_matrix("V", h_V.data(), seq_len, head_dim);

        // 1. Compute naive attention
        std::cout << "\n=== Naive Attention ===" << std::endl;
        naive_attention_forward(h_Q.data(), h_K.data(), h_V.data(),
            h_O_naive.data(), seq_len, head_dim, softmax_scale);
        print_matrix("O_naive", h_O_naive.data(), seq_len, head_dim);

        // 2. Compute FlashAttention on single GPU
        std::cout << "\n=== FlashAttention (Single GPU) ===" << std::endl;

        // Allocate device memory
        float* d_Q, * d_K, * d_V, * d_O, * d_L;
        size_t qkv_size = seq_len * head_dim * sizeof(float);
        CHECK_CUDA(cudaMalloc(&d_Q, qkv_size));
        CHECK_CUDA(cudaMalloc(&d_K, qkv_size));
        CHECK_CUDA(cudaMalloc(&d_V, qkv_size));
        CHECK_CUDA(cudaMalloc(&d_O, qkv_size));
        CHECK_CUDA(cudaMalloc(&d_L, seq_len * sizeof(float)));

        // Copy to device
        CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), qkv_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), qkv_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), qkv_size, cudaMemcpyHostToDevice));

        // Run single-GPU FlashAttention
        flash_attention_2_forward(d_Q, d_K, d_V, d_O, d_L, seq_len, head_dim, softmax_scale);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy result back
        CHECK_CUDA(cudaMemcpy(h_O_flash.data(), d_O, qkv_size, cudaMemcpyDeviceToHost));
        print_matrix("O_flash", h_O_flash.data(), seq_len, head_dim);

        // Compare naive vs flash
        std::cout << "\n=== Comparison: Naive vs FlashAttention ===" << std::endl;
        bool match = compare_outputs(h_O_naive.data(), h_O_flash.data(), seq_len * head_dim);

        if (!match) {
            std::cout << "WARNING: FlashAttention output doesn't match naive!" << std::endl;
        }

        // Cleanup
        CHECK_CUDA(cudaFree(d_Q));
        CHECK_CUDA(cudaFree(d_K));
        CHECK_CUDA(cudaFree(d_V));
        CHECK_CUDA(cudaFree(d_O));
        CHECK_CUDA(cudaFree(d_L));
    }

    // Broadcast reference output to all ranks
    MPI_Bcast(h_O_naive.data(), seq_len * head_dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n=== Ring Attention (Distributed) ===" << std::endl;
        std::cout << "Running on " << nranks << " GPUs" << std::endl;
        std::cout << "Each GPU processes " << seq_len / nranks << " rows of Q" << std::endl;
    }

    // TODO: Next we'll implement the distributed Ring Attention here
    // and compare with h_O_naive

    cleanup_mpi_nccl(comm);
    return 0;
}