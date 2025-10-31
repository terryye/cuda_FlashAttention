
#include <stdio.h>
#include <iostream>
#include "util/cuda_shim.h"
#include "./flash_attention_kernel.cu"

// Host function to launch Flash Attention
void flash_attention(
    const float* d_Q,
    const float* d_K, 
    const float* d_V,
    float* d_O,
    float* d_l,
    float* d_m,
    int N,
    int d,
    int Bc = 32,  // Block size for columns . for test use.
    int M = 1024 // Shared memory size limit (in floats) 
) {
    // Line 1: Set block sizes Br, Bc
    // Bc = M / ( 4 * d ); to make test easier, we set Bc directly.
    int Br = min(d, Bc);  // Set Br = min(d, Bc)
    
    //Line 2: Initialize O = 0, l = 0, m = -∞ on HBM
    cudaMemset(d_O, 0, N * d * sizeof(float));  // O = 0
    cudaMemset(d_l, 0, N * sizeof(float));      // l = 0
    // Initialize m to -infinity
    float* h_m_init = new float[N];
    for (int i = 0; i < N; i++) {
        h_m_init[i] = -FLT_MAX;
    }
    cudaMemcpy(d_m, h_m_init, N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_m_init;

    //line 3: Calculate Tr = ceil(N/Br), Tc = ceil(N/Bc)
    // Calculate grid and block dimensions
    int Tr = (N + Br - 1) / Br; //Q,    O, l, m
    int Tc = (N + Bc - 1) / Bc; //K, V
    
    printf("Flash Attention launch with Br=%d, Bc=%d, Tr=%d, Tc=%d\n", Br, Bc, Tr, Tc);
    
    // line 7: for 1 <= i <= Tr do  block and thread implement this loop
    dim3 threadsPerBlock = new_dim3(Br, 1, 1);  // Each block has Br threads, each thread handles one row of Qi
    dim3 blocksPerGrid = new_dim3(Tr, 1, 1);    // Each block handles one tile of Qi

    // Calculate shared memory size
    size_t shared_mem_size = sizeof(float) * (
        Br * d +      // Qi
        Bc * d +      // Kj
        Bc * d +      // Vj
        Br * Bc +     // S
        Br * Bc +     // P
        Br * d +      // Oi (in SRAM)
        Br +          // li (in SRAM)
        Br            // mi (in SRAM)
    );

    // Launch kernel
    // line 5: for 1 ≤ i ≤ Tc do
    #ifndef __INTELLISENSE__
    flash_attention_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(
        d_Q, d_K, d_V, d_O, d_l, d_m, N, d, Br, Bc,1.0f / sqrtf((float)d)
    );
    #endif
    cudaDeviceSynchronize(); // Ensure sequential execution of blocks
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}


// Test function
#include <cmath>
#include <cassert>

// Naive CPU attention for verification
void naive_attention(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    for (int i = 0; i < N; ++i) {
        float* out = &O[i * d];
        std::fill(out, out + d, 0.0f);
        float max_score = -FLT_MAX;
        float* scores = new float[N];
        // Compute scores
        for (int j = 0; j < N; ++j) {
            float score = 0.0f;
            for (int k = 0; k < d; ++k)
                score += Q[i * d + k] * K[j * d + k];
            score /= sqrtf((float)d);
            scores[j] = score;
            if (score > max_score) max_score = score;
        }
        // Softmax normalization
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            scores[j] = expf(scores[j] - max_score);
            sum_exp += scores[j];
        }
        // Weighted sum
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < d; ++k)
                out[k] += scores[j] * V[j * d + k];
        }
        for (int k = 0; k < d; ++k)
            out[k] /= sum_exp;
        delete[] scores;
    }
}

void run_test(const char* test_name, float* Q, float* K, float* V, int N, int d, int Bc = 32) {
    std::cout << "\n=== Test: " << test_name << " ===" << std::endl;
    std::cout << "N=" << N << ", d=" << d << ", Bc=" << Bc << std::endl;
    
    // Allocate output memory
    float *h_O = new float[N * d];
    float *h_l = new float[N];
    float *h_m = new float[N];
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    cudaMalloc((void**)&d_Q, N * d * sizeof(float));
    cudaMalloc((void**)&d_K, N * d * sizeof(float));
    cudaMalloc((void**)&d_V, N * d * sizeof(float));
    cudaMalloc((void**)&d_O, N * d * sizeof(float));
    cudaMalloc((void**)&d_l, N * sizeof(float));
    cudaMalloc((void**)&d_m, N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_Q, Q, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * d * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run Flash Attention
    flash_attention(d_Q, d_K, d_V, d_O, d_l, d_m, N, d, Bc);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_l, d_l, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m, d_m, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "Output O:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < d && j < 8; j++) {  // Print first 8 values
            std::cout << h_O[i * d + j] << " ";
        }
        if (d > 8) std::cout << "...";
        std::cout << " | l=" << h_l[i] << ", m=" << h_m[i] << std::endl;
    }

    // Naive CPU reference
    float* ref_O = new float[N * d];
    naive_attention(Q, K, V, ref_O, N, d);

    // Compare results
    bool correct = true;
    float max_diff = 0.0f;
    for (int i = 0; i < N * d; ++i) {
        float diff = fabs(h_O[i] - ref_O[i]);
        if (diff > 1e-3f) correct = false;
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Verification: " << (correct ? "PASS" : "FAIL") << " (max diff = " << max_diff << ")" << std::endl;

    // Optionally print reference output for debugging
    if (!correct) {
        std::cout << "Reference O:" << std::endl;
        for (int i = 0; i < N; i++) {
            std::cout << "Row " << i << ": ";
            for (int j = 0; j < d && j < 8; j++) {
                std::cout << ref_O[i * d + j] << " ";
            }
            if (d > 8) std::cout << "...";
            std::cout << std::endl;
        }
    }
    
    // Cleanup
    delete[] h_O;
    delete[] h_l;
    delete[] h_m;
    delete[] ref_O;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_l);
    cudaFree(d_m);
}



int main() {
    // Test 1: simple 2x4 test
    {
        float Q[] = {
            1, 0, 1, 0,
            0, 1, 0, 1
        };
        float K[] = {
            1, 0, 1, 0,
            0, 1, 0, 1
        };
        float V[] = {
            10, 20, 30, 40,
            50, 60, 70, 80
        };
        run_test("Simple 2x4 test", Q, K, V, 2, 4, 2);
    }
    // Test 2: Identity attention (perfect match)
    {
        float Q[] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
        float K[] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
        float V[] = {
            1, 0, 0, 0,
            0, 2, 0, 0,
            0, 0, 3, 0,
            0, 0, 0, 4
        };
        run_test("Identity 4x4", Q, K, V, 4, 4, 2);
    }
    
    // Test 3: All ones (uniform attention)
    {
        float Q[] = {
            1, 1,
            1, 1,
            1, 1
        };
        float K[] = {
            1, 1,
            1, 1,
            1, 1
        };
        float V[] = {
            1, 2,
            3, 4,
            5, 6
        };
        run_test("Uniform attention 3x2", Q, K, V, 3, 2, 2);
    }
    
    // Test 4: Orthogonal Q and K (no attention)
    {
        float Q[] = {
            1, 0,
            0, 1
        };
        float K[] = {
            0, 1,
            -1, 0
        };
        float V[] = {
            10, 20,
            30, 40
        };
        run_test("Orthogonal Q,K 2x2", Q, K, V, 2, 2, 2);
    }
    
    // Test 5: Single element
    {
        float Q[] = {1};
        float K[] = {1};
        float V[] = {42};
        run_test("Single element", Q, K, V, 1, 1, 1);
    }
    
    // Test 6: Larger test with patterns
    {
        const int N = 8, d = 4;
        float *Q = new float[N * d];
        float *K = new float[N * d];
        float *V = new float[N * d];
        
        // Create diagonal attention pattern
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < d; j++) {
                Q[i * d + j] = (i == j) ? 1.0f : 0.0f;
                K[i * d + j] = (i == j) ? 1.0f : 0.0f;
                V[i * d + j] = i * 10 + j;
            }
        }
        run_test("Diagonal pattern 8x4", Q, K, V, N, d, 4);
        
        delete[] Q;
        delete[] K;
        delete[] V;
    }
    
    // Test 7: Random larger test
    {
        const int N = 64, d = 32;
        float *Q = new float[N * d];
        float *K = new float[N * d];
        float *V = new float[N * d];
        
        srand(42);  // Fixed seed for reproducibility
        for (int i = 0; i < N * d; i++) {
            Q[i] = (float)rand() / RAND_MAX;
            K[i] = (float)rand() / RAND_MAX;
            V[i] = (float)rand() / RAND_MAX * 100;  // Scale V for visibility
        }
        run_test("Random 64x32", Q, K, V, N, d, 16);
        
        delete[] Q;
        delete[] K;
        delete[] V;
    }
    
    // Test 8: Test with different tile sizes
    {
        float Q[] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
        float K[] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
        float V[] = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        };
        run_test("4x4 with Br=1, Bc=1", Q, K, V, 4, 4, 1);
        run_test("4x4 with Br=2, Bc=2", Q, K, V, 4, 4, 2);
        run_test("4x4 with Br=4, Bc=4", Q, K, V, 4, 4, 4);
    }
    
    return 0;
}