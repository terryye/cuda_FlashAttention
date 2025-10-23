// flash_attention_2_forward.cu
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "flash_attention_kernel.cu"  // Include the kernel implementation

// Naive attention implementation for comparison
void naive_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim,
    float softmax_scale
) {
    float* S = new float[seq_len * seq_len];
    
    // S = Q @ K^T
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float dot = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                dot += Q[i * head_dim + k] * K[j * head_dim + k];
            }
            S[i * seq_len + j] = dot * softmax_scale;
        }
    }
    
    // Softmax
    for (int i = 0; i < seq_len; i++) {
        float row_max = -INFINITY;
        for (int j = 0; j < seq_len; j++) {
            row_max = std::max(row_max, S[i * seq_len + j]);
        }
        
        float row_sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            S[i * seq_len + j] = expf(S[i * seq_len + j] - row_max);
            row_sum += S[i * seq_len + j];
        }
        
        for (int j = 0; j < seq_len; j++) {
            S[i * seq_len + j] /= row_sum;
        }
    }
    
    // O = P @ V
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            float sum = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                sum += S[i * seq_len + k] * V[k * head_dim + j];
            }
            O[i * head_dim + j] = sum;
        }
    }
    
    delete[] S;
}

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
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc((void **) &d_Q, qkv_size * sizeof(float));
    cudaMalloc((void **) &d_K, qkv_size * sizeof(float));
    cudaMalloc((void **) &d_V, qkv_size * sizeof(float));
    cudaMalloc((void **) &d_O, qkv_size * sizeof(float));
    cudaMalloc((void **) &d_L, seq_len * sizeof(float));
    
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
        for (int j = 0; j < min(head_dim,8); j++) {
            std::cout << h_O_flash[ head_dim + j] << " ";
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

int main() {
    test_flash_attention_2();
    return 0;
}