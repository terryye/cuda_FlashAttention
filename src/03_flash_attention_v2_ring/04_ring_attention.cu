// test_ring_attention_simple.cu (updated)
#include <iostream>
#include <vector>
#include <mpi.h>
#include "util/nccl_utils.h"
#include "util/naive_attention.h"
#include "./common/ring_attention_kernel.cu"

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

    // Reference output
    std::vector<float> h_O_naive(seq_len * head_dim);

    // Only compute reference on rank 0
    if (rank == 0) {
        std::cout << "=== Test Case ===" << std::endl;

        // Create test data
        std::vector<float> h_Q_full, h_K_full, h_V_full;
        create_simple_test_data(h_Q_full, h_K_full, h_V_full, seq_len, head_dim);

        print_matrix("Q", h_Q_full.data(), seq_len, head_dim);
        print_matrix("K", h_K_full.data(), seq_len, head_dim);
        print_matrix("V", h_V_full.data(), seq_len, head_dim);

        // Compute naive attention
        std::cout << "\n=== Naive Attention Reference ===" << std::endl;
        naive_attention_forward(h_Q_full.data(), h_K_full.data(), h_V_full.data(),
            h_O_naive.data(), seq_len, head_dim, softmax_scale);
        print_matrix("O_naive", h_O_naive.data(), seq_len, head_dim);
    }

    // Broadcast reference output to all ranks
    MPI_Bcast(h_O_naive.data(), seq_len * head_dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n=== Ring Attention (Distributed) ===" << std::endl;
        std::cout << "Running on " << nranks << " GPUs" << std::endl;
        std::cout << "Each GPU processes " << seq_len / nranks << " rows of Q" << std::endl;
    }

    // Prepare distributed data
    int local_seq_len = seq_len / nranks;
    if (seq_len % nranks != 0) {
        if (rank == 0) {
            //make things easier by requiring divisibility
            std::cerr << "seq_len must be divisible by nranks!" << std::endl;
        }
        cleanup_mpi_nccl(comm);
        return 1;
    }

    // Create full test data on all ranks (for easy setup)
    std::vector<float> h_Q_full, h_K_full, h_V_full;
    create_simple_test_data(h_Q_full, h_K_full, h_V_full, seq_len, head_dim);

    // Extract local portions
    std::vector<float> h_Q_local(local_seq_len * head_dim);
    std::vector<float> h_K_local(local_seq_len * head_dim);
    std::vector<float> h_V_local(local_seq_len * head_dim);

    // Each rank gets its portion of rows
    int start_row = rank * local_seq_len;
    for (int i = 0; i < local_seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            int local_idx = i * head_dim + j;
            int global_idx = (start_row + i) * head_dim + j;
            h_Q_local[local_idx] = h_Q_full[global_idx];
            h_K_local[local_idx] = h_K_full[global_idx];
            h_V_local[local_idx] = h_V_full[global_idx];
        }
    }

    printf("Rank %d: Q_local rows %d-%d\n", rank, start_row, start_row + local_seq_len - 1);

    // Allocate device memory for distributed computation
    float* d_Q_local, * d_K_local, * d_V_local, * d_O_local, * d_L_local;
    size_t local_qkv_size = local_seq_len * head_dim * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_Q_local, local_qkv_size));
    CHECK_CUDA(cudaMalloc(&d_K_local, local_qkv_size));
    CHECK_CUDA(cudaMalloc(&d_V_local, local_qkv_size));
    CHECK_CUDA(cudaMalloc(&d_O_local, local_qkv_size));
    CHECK_CUDA(cudaMalloc(&d_L_local, local_seq_len * sizeof(float)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_Q_local, h_Q_local.data(), local_qkv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K_local, h_K_local.data(), local_qkv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V_local, h_V_local.data(), local_qkv_size, cudaMemcpyHostToDevice));

    // Run ring attention
    ring_attention_forward(
        d_Q_local, d_K_local, d_V_local, d_O_local, d_L_local,
        seq_len, local_seq_len, head_dim, softmax_scale,
        comm, rank, nranks
    );

    // Copy result back
    std::vector<float> h_O_local(local_seq_len * head_dim);
    CHECK_CUDA(cudaMemcpy(h_O_local.data(), d_O_local, local_qkv_size,
        cudaMemcpyDeviceToHost));

    // Gather all outputs to rank 0
    std::vector<float> h_O_gathered;
    if (rank == 0) {
        h_O_gathered.resize(seq_len * head_dim);
    }

    //The messages are concatenated on the root process in the order of the process ranks, so the data from rank 0 comes first, followed by rank 1, and so on.
    // int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
    //                MPI_Comm comm)
    MPI_Gather(h_O_local.data(), local_seq_len * head_dim, MPI_FLOAT,
        h_O_gathered.data(), local_seq_len * head_dim, MPI_FLOAT,
        0, MPI_COMM_WORLD);

    // Compare on rank 0
    if (rank == 0) {
        std::cout << "\n=== Ring Attention Output ===" << std::endl;
        print_matrix("O_ring", h_O_gathered.data(), seq_len, head_dim);

        std::cout << "\n=== Final Comparison: Naive vs Ring Attention ===" << std::endl;
        bool match = compare_outputs(h_O_naive.data(), h_O_gathered.data(),
            seq_len * head_dim, 5e-3f);

        if (!match) {
            std::cout << "Test FAILED!" << std::endl;
        }
        else {
            std::cout << "Test PASSED!" << std::endl;
        }
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_Q_local));
    CHECK_CUDA(cudaFree(d_K_local));
    CHECK_CUDA(cudaFree(d_V_local));
    CHECK_CUDA(cudaFree(d_O_local));
    CHECK_CUDA(cudaFree(d_L_local));

    cleanup_mpi_nccl(comm);
    return 0;
}