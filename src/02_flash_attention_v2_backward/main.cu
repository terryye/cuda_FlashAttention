// FlashAttention-2 Backward Pass Test
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "flash_attention_backward_kernel.cu"
#include "util/naive_attention.h"

void print_matrix(const char* name, const float* mat, int rows, int cols, int max_rows = 4, int max_cols = 8) {
    std::cout << "\n" << name << " (showing " << std::min(rows, max_rows)
        << "x" << std::min(cols, max_cols) << "):" << std::endl;
    for (int i = 0; i < std::min(rows, max_rows); i++) {
        for (int j = 0; j < std::min(cols, max_cols); j++) {
            printf("%8.4f ", mat[i * cols + j]);
        }
        std::cout << std::endl;
    }
}

void compare_gradients(const char* name, const float* grad_flash, const float* grad_naive,
    int size, float& max_diff, float& avg_diff) {
    max_diff = 0.0f;
    avg_diff = 0.0f;
    int num_large_diff = 0;

    for (int i = 0; i < size; i++) {
        float diff = std::abs(grad_flash[i] - grad_naive[i]);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;

        if (diff > 1e-3 || std::isnan(diff)) {
            num_large_diff++;
            if (num_large_diff <= 5) {
                std::cout << "Large diff in " << name << " at index " << i
                    << ": flash=" << grad_flash[i]
                    << ", naive=" << grad_naive[i]
                    << ", diff=" << diff << std::endl;
            }
        }
    }
    avg_diff /= size;

    std::cout << name << " - Max diff: " << max_diff << ", Avg diff: " << avg_diff
        << ", Large diffs: " << num_large_diff << "/" << size << std::endl;
}

/**
 * Test Case 1: Simple test with easy-to-calculate data
 * Uses identity-like patterns to make manual verification possible
 */
void test_simple_backward() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Case 1: Simple Backward Pass" << std::endl;
    std::cout << "========================================" << std::endl;

    const int seq_len = 4;
    const int head_dim = 4;
    const float softmax_scale = 1.0f;

    size_t qkv_size = seq_len * head_dim;

    // Allocate host memory
    float* h_Q = new float[qkv_size];
    float* h_K = new float[qkv_size];
    float* h_V = new float[qkv_size];
    float* h_O = new float[qkv_size];
    float* h_L = new float[seq_len];
    float* h_dO = new float[qkv_size];
    float* h_dQ_naive = new float[qkv_size];
    float* h_dK_naive = new float[qkv_size];
    float* h_dV_naive = new float[qkv_size];
    float* h_dQ_flash = new float[qkv_size];
    float* h_dK_flash = new float[qkv_size];
    float* h_dV_flash = new float[qkv_size];

    // Initialize with simple patterns
    // Q: identity-like pattern
    float Q_vals[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    // K: same as Q
    float K_vals[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    // V: simple incremental values
    float V_vals[] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f, 3.0f,
        4.0f, 4.0f, 4.0f, 4.0f
    };

    // dO: gradient from upstream (simple ones)
    float dO_vals[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    memcpy(h_Q, Q_vals, qkv_size * sizeof(float));
    memcpy(h_K, K_vals, qkv_size * sizeof(float));
    memcpy(h_V, V_vals, qkv_size * sizeof(float));
    memcpy(h_dO, dO_vals, qkv_size * sizeof(float));

    // Print inputs
    print_matrix("Q", h_Q, seq_len, head_dim);
    print_matrix("K", h_K, seq_len, head_dim);
    print_matrix("V", h_V, seq_len, head_dim);
    print_matrix("dO", h_dO, seq_len, head_dim);

    // Compute forward pass (to get O and L)
    naive_forward_pass(h_Q, h_K, h_V, h_O, h_L, seq_len, head_dim, softmax_scale);
    print_matrix("O (forward)", h_O, seq_len, head_dim);

    // Compute naive backward
    naive_attention_backward(h_Q, h_K, h_V, h_O, h_L, h_dO,
        h_dQ_naive, h_dK_naive, h_dV_naive,
        seq_len, head_dim, softmax_scale);

    // Allocate device memory
    float* d_Q, * d_K, * d_V, * d_O, * d_L, * d_dO, * d_dQ, * d_dK, * d_dV;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, seq_len * sizeof(float));
    cudaMalloc(&d_dO, qkv_size * sizeof(float));
    cudaMalloc(&d_dQ, qkv_size * sizeof(float));
    cudaMalloc(&d_dK, qkv_size * sizeof(float));
    cudaMalloc(&d_dV, qkv_size * sizeof(float));

    // Copy to device
    cudaMemcpy(d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, h_O, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_L, seq_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, h_dO, qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    // Run FlashAttention-2 backward
    flash_attention_2_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK, d_dV,
        seq_len, head_dim, softmax_scale);

    cudaDeviceSynchronize();

    // Copy back to host
    cudaMemcpy(h_dQ_flash, d_dQ, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK_flash, d_dK, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV_flash, d_dV, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    print_matrix("dQ (naive)", h_dQ_naive, seq_len, head_dim);
    print_matrix("dQ (flash)", h_dQ_flash, seq_len, head_dim);
    print_matrix("dK (naive)", h_dK_naive, seq_len, head_dim);
    print_matrix("dK (flash)", h_dK_flash, seq_len, head_dim);
    print_matrix("dV (naive)", h_dV_naive, seq_len, head_dim);
    print_matrix("dV (flash)", h_dV_flash, seq_len, head_dim);

    // Compare results
    std::cout << "\n--- Comparison Results ---" << std::endl;
    float max_diff, avg_diff;
    compare_gradients("dQ", h_dQ_flash, h_dQ_naive, qkv_size, max_diff, avg_diff);
    bool dQ_pass = max_diff < 1e-3;

    compare_gradients("dK", h_dK_flash, h_dK_naive, qkv_size, max_diff, avg_diff);
    bool dK_pass = max_diff < 1e-3;

    compare_gradients("dV", h_dV_flash, h_dV_naive, qkv_size, max_diff, avg_diff);
    bool dV_pass = max_diff < 1e-3;

    std::cout << "\nTest Case 1: " << (dQ_pass && dK_pass && dV_pass ? "PASSED ✓" : "FAILED ✗") << std::endl;

    // Cleanup
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O; delete[] h_L; delete[] h_dO;
    delete[] h_dQ_naive; delete[] h_dK_naive; delete[] h_dV_naive;
    delete[] h_dQ_flash; delete[] h_dK_flash; delete[] h_dV_flash;

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L); cudaFree(d_dO);
    cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);
}

/**
 * Test Case 2: Complex test with larger dimensions and random data
 * Realistic scenario with more complex attention patterns
 */
void test_complex_backward() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Case 2: Complex Backward Pass" << std::endl;
    std::cout << "========================================" << std::endl;

    const int seq_len = 128;
    const int head_dim = 64;
    const float softmax_scale = 1.0f / sqrtf(head_dim);

    size_t qkv_size = seq_len * head_dim;

    // Allocate host memory
    float* h_Q = new float[qkv_size];
    float* h_K = new float[qkv_size];
    float* h_V = new float[qkv_size];
    float* h_O = new float[qkv_size];
    float* h_L = new float[seq_len];
    float* h_dO = new float[qkv_size];
    float* h_dQ_naive = new float[qkv_size];
    float* h_dK_naive = new float[qkv_size];
    float* h_dV_naive = new float[qkv_size];
    float* h_dQ_flash = new float[qkv_size];
    float* h_dK_flash = new float[qkv_size];
    float* h_dV_flash = new float[qkv_size];

    // Initialize with random values (fixed seed for reproducibility)
    srand(42);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.5f;  // [-0.5, 0.5]
        h_K[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.5f;
        h_V[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.5f;
        h_dO[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.2f; // [-0.2, 0.2]
    }

    std::cout << "Sequence length: " << seq_len << std::endl;
    std::cout << "Head dimension: " << head_dim << std::endl;
    std::cout << "Softmax scale: " << softmax_scale << std::endl;

    print_matrix("Q", h_Q, seq_len, head_dim, 3, 6);
    print_matrix("K", h_K, seq_len, head_dim, 3, 6);
    print_matrix("V", h_V, seq_len, head_dim, 3, 6);
    print_matrix("dO", h_dO, seq_len, head_dim, 3, 6);

    // Compute forward pass (to get O and L)
    naive_forward_pass(h_Q, h_K, h_V, h_O, h_L, seq_len, head_dim, softmax_scale);
    print_matrix("O (forward)", h_O, seq_len, head_dim, 3, 6);

    // Compute naive backward
    std::cout << "\nComputing naive backward pass..." << std::endl;
    naive_attention_backward(h_Q, h_K, h_V, h_O, h_L, h_dO,
        h_dQ_naive, h_dK_naive, h_dV_naive,
        seq_len, head_dim, softmax_scale);

    // Allocate device memory
    float* d_Q, * d_K, * d_V, * d_O, * d_L, * d_dO, * d_dQ, * d_dK, * d_dV;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, seq_len * sizeof(float));
    cudaMalloc(&d_dO, qkv_size * sizeof(float));
    cudaMalloc(&d_dQ, qkv_size * sizeof(float));
    cudaMalloc(&d_dK, qkv_size * sizeof(float));
    cudaMalloc(&d_dV, qkv_size * sizeof(float));

    // Copy to device
    cudaMemcpy(d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, h_O, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_L, seq_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, h_dO, qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    // Run FlashAttention-2 backward
    std::cout << "Running FlashAttention-2 backward pass..." << std::endl;
    flash_attention_2_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK, d_dV,
        seq_len, head_dim, softmax_scale);

    cudaDeviceSynchronize();

    // Copy back to host
    cudaMemcpy(h_dQ_flash, d_dQ, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK_flash, d_dK, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV_flash, d_dV, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print sample results
    print_matrix("dQ (naive)", h_dQ_naive, seq_len, head_dim, 3, 6);
    print_matrix("dQ (flash)", h_dQ_flash, seq_len, head_dim, 3, 6);
    print_matrix("dK (naive)", h_dK_naive, seq_len, head_dim, 3, 6);
    print_matrix("dK (flash)", h_dK_flash, seq_len, head_dim, 3, 6);
    print_matrix("dV (naive)", h_dV_naive, seq_len, head_dim, 3, 6);
    print_matrix("dV (flash)", h_dV_flash, seq_len, head_dim, 3, 6);

    // Compare results
    std::cout << "\n--- Comparison Results ---" << std::endl;
    float max_diff, avg_diff;
    compare_gradients("dQ", h_dQ_flash, h_dQ_naive, qkv_size, max_diff, avg_diff);
    bool dQ_pass = max_diff < 5e-3;  // More lenient for larger matrices

    compare_gradients("dK", h_dK_flash, h_dK_naive, qkv_size, max_diff, avg_diff);
    bool dK_pass = max_diff < 5e-3;

    compare_gradients("dV", h_dV_flash, h_dV_naive, qkv_size, max_diff, avg_diff);
    bool dV_pass = max_diff < 5e-3;

    std::cout << "\nTest Case 2: " << (dQ_pass && dK_pass && dV_pass ? "PASSED ✓" : "FAILED ✗") << std::endl;

    // Cleanup
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O; delete[] h_L; delete[] h_dO;
    delete[] h_dQ_naive; delete[] h_dK_naive; delete[] h_dV_naive;
    delete[] h_dQ_flash; delete[] h_dK_flash; delete[] h_dV_flash;

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L); cudaFree(d_dO);
    cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);
}

int main() {
    std::cout << "FlashAttention-2 Backward Pass Implementation" << std::endl;
    std::cout << "Based on Section 3 of the FlashAttention-2 paper" << std::endl;
    std::cout << "Section 3.2: Parallelism" << std::endl;
    std::cout << "Section 3.3: Work Partitioning between Warps" << std::endl;

    test_simple_backward();
    test_complex_backward();

    return 0;
}
