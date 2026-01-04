// flash_attention_2_forward.cu
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "flash_attention_kernel.cu"  // Include the kernel implementation
#include "util/naive_attention.h"




// Test function
void test_flash_attention_2() {
    // Test parameters
    const int seq_len = 512;
    const int head_dim = 64;
    const float softmax_scale = 1.0f / sqrtf(head_dim);

    // Allocate host memory
    size_t qkv_size = seq_len * head_dim;
    float* h_Q = new float[qkv_size];
    float* h_K = new float[qkv_size];
    float* h_V = new float[qkv_size];
    float* h_O_naive = new float[qkv_size];
    float* h_O_flash = new float[qkv_size];
    float* h_L = new float[seq_len];

    // Initialize with random values
    srand(42); // Fixed seed for reproducibility
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = (rand() % 1000) / 1000.0f - 0.5f;
        h_K[i] = (rand() % 1000) / 1000.0f - 0.5f;
        h_V[i] = (rand() % 1000) / 1000.0f - 0.5f;
    }

    // Compute naive attention
    naive_attention_forward(
        h_Q, h_K, h_V, h_O_naive,
        seq_len, head_dim, softmax_scale
    );

    // Allocate device memory
    float* d_Q, * d_K, * d_V, * d_O, * d_L;
    cudaMalloc((void**)&d_Q, qkv_size * sizeof(float));
    cudaMalloc((void**)&d_K, qkv_size * sizeof(float));
    cudaMalloc((void**)&d_V, qkv_size * sizeof(float));
    cudaMalloc((void**)&d_O, qkv_size * sizeof(float));
    cudaMalloc((void**)&d_L, seq_len * sizeof(float));

    // Copy to device
    cudaMemcpy(d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    // Run FlashAttention-2
    flash_attention_2_forward(
        d_Q, d_K, d_V, d_O, d_L,
        seq_len, head_dim, softmax_scale
    );

    cudaDeviceSynchronize();

    // Copy back to host
    cudaMemcpy(h_O_flash, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_L, d_L, seq_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int num_large_diff = 0;

    for (size_t i = 0; i < qkv_size; i++) {
        float diff = std::abs(h_O_naive[i] - h_O_flash[i]);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
        if (diff > 1e-3 || std::isnan(diff)) {
            num_large_diff++;
            if (num_large_diff < 10) { // Print first few large differences
                std::cout << "Large diff at index " << i << ": naive=" << h_O_naive[i]
                    << ", flash=" << h_O_flash[i] << ", diff=" << diff << std::endl;
            }
        }
    }
    avg_diff /= qkv_size;

    std::cout << "\nTest Results:" << std::endl;
    std::cout << "Max difference: " << max_diff << std::endl;
    std::cout << "Avg difference: " << avg_diff << std::endl;
    std::cout << "Number of large differences (>1e-3): " << num_large_diff << " out of " << qkv_size << std::endl;
    std::cout << "Test " << (max_diff < 5e-3 ? "PASSED" : "FAILED") << std::endl;
    //print H_O_flash first 4 rows
    std::cout << "\nFirst 4 rows,8cols of FlashAttention-2 output O:" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < min(head_dim, 8); j++) {
            std::cout << h_O_flash[i * head_dim + j] << " ";
        }
        std::cout << std::endl;
    }
    // Cleanup host memory
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O_naive;
    delete[] h_O_flash;
    delete[] h_L;

    // Cleanup device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

// Test function with simple integer values for manual verification
void test_simple_attention() {
    std::cout << "\n=== Simple Test Case ===" << std::endl;

    // Simple test parameters
    const int seq_len = 4;
    const int head_dim = 4;
    const float softmax_scale = 1.0f;

    // Allocate host memory
    size_t qkv_size = seq_len * head_dim;
    float* h_Q = new float[qkv_size];
    float* h_K = new float[qkv_size];
    float* h_V = new float[qkv_size];
    float* h_O_naive = new float[qkv_size];
    float* h_O_flash = new float[qkv_size];
    float* h_L = new float[seq_len];

    // Initialize with simple integer values
    // Q matrix (4x4)
    float Q_vals[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 0.0f
    };

    // K matrix (4x4) - same as Q for simplicity
    float K_vals[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 0.0f
    };

    // V matrix (4x4) - distinct values for each row
    float V_vals[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    memcpy(h_Q, Q_vals, qkv_size * sizeof(float));
    memcpy(h_K, K_vals, qkv_size * sizeof(float));
    memcpy(h_V, V_vals, qkv_size * sizeof(float));

    // Print input matrices
    std::cout << "\nInput Q matrix:" << std::endl;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            std::cout << h_Q[i * head_dim + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nInput K matrix:" << std::endl;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            std::cout << h_K[i * head_dim + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nInput V matrix:" << std::endl;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            std::cout << h_V[i * head_dim + j] << " ";
        }
        std::cout << std::endl;
    }

    // Compute naive attention
    naive_attention_forward(
        h_Q, h_K, h_V, h_O_naive,
        seq_len, head_dim, softmax_scale
    );

    // Allocate device memory
    float* d_Q, * d_K, * d_V, * d_O, * d_L;
    cudaMalloc((void**)&d_Q, qkv_size * sizeof(float));
    cudaMalloc((void**)&d_K, qkv_size * sizeof(float));
    cudaMalloc((void**)&d_V, qkv_size * sizeof(float));
    cudaMalloc((void**)&d_O, qkv_size * sizeof(float));
    cudaMalloc((void**)&d_L, seq_len * sizeof(float));

    // Copy to device
    cudaMemcpy(d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    // Run FlashAttention-2
    flash_attention_2_forward(
        d_Q, d_K, d_V, d_O, d_L,
        seq_len, head_dim, softmax_scale
    );

    cudaDeviceSynchronize();

    // Copy back to host
    cudaMemcpy(h_O_flash, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_L, d_L, seq_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "\nNaive Attention Output O:" << std::endl;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            std::cout << h_O_naive[i * head_dim + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nFlashAttention-2 Output O:" << std::endl;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            std::cout << h_O_flash[i * head_dim + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nLog-sum-exp values L:" << std::endl;
    for (int i = 0; i < seq_len; i++) {
        std::cout << "L[" << i << "] = " << h_L[i] << std::endl;
    }

    // Compare results
    float max_diff = 0.0f;
    for (size_t i = 0; i < qkv_size; i++) {
        float diff = std::abs(h_O_naive[i] - h_O_flash[i]);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "\nMax difference: " << max_diff << std::endl;
    std::cout << "Simple test " << (max_diff < 1e-4 ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O_naive;
    delete[] h_O_flash;
    delete[] h_L;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

int main() {
    test_simple_attention();  // Run simple test first
    test_flash_attention_2(); // Run original test
    return 0;
}