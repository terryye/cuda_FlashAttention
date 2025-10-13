
#include <stdio.h>
#include <iostream>
#include "../../util/cuda_shim.h"
#include "./flash_attention_kernel.cu"

// Helper function to compute forward pass for testing
void compute_forward_pass(const float* Q, const float* K, const float* V,
                         float* O, float* L, int N, int d, float scale) {
    // Simple (non-optimized) attention forward pass for testing
    float* S = new float[N * N];
    float* P = new float[N * N];
    
    // Compute S = Q * K^T * scale
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum * scale;
        }
    }
    
    // Compute row-wise softmax
    for (int i = 0; i < N; i++) {
        // Find max for numerical stability
        float max_val = -1e9f;
        for (int j = 0; j < N; j++) {
            max_val = fmax(max_val, S[i * N + j]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            P[i * N + j] = expf(S[i * N + j] - max_val);
            sum += P[i * N + j];
        }
        
        // Normalize and store L
        L[i] = max_val + logf(sum);
        for (int j = 0; j < N; j++) {
            P[i * N + j] /= sum;
        }
    }
    
    // Compute O = P * V
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += P[i * N + k] * V[k * d + j];
            }
            O[i * d + j] = sum;
        }
    }
    
    delete[] S;
    delete[] P;
}

// Generic test runner for different configurations
// Naive backward pass for attention (for verification)
void naive_attention_backward(const float* Q, const float* K, const float* V,
                             const float* O, const float* L, const float* dO,
                             float* dQ, float* dK, float* dV,
                             int N, int d, float scale) {
    // Compute S = Q * K^T * scale
    float* S = new float[N * N];
    float* P = new float[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum * scale;
        }
    }
    // Softmax
    for (int i = 0; i < N; i++) {
        float max_val = -1e9f;
        for (int j = 0; j < N; j++) max_val = fmax(max_val, S[i * N + j]);
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            P[i * N + j] = expf(S[i * N + j] - max_val);
            sum += P[i * N + j];
        }
        for (int j = 0; j < N; j++) P[i * N + j] /= sum;
    }
    // Backward pass
    // dV = P^T * dO
    for (int i = 0; i < N; i++)
        for (int j = 0; j < d; j++)
            dV[i * d + j] = 0.0f;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < d; k++)
                dV[j * d + k] += P[i * N + j] * dO[i * d + k];
    // dP = dO * V^T
    float* dP = new float[N * N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++)
                sum += dO[i * d + k] * V[j * d + k];
            dP[i * N + j] = sum;
        }
    // dS = dP * softmax jacobian
    float* dS = new float[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++) {
                float J = (j == k ? P[i * N + j] * (1 - P[i * N + k]) : -P[i * N + j] * P[i * N + k]);
                s += dP[i * N + k] * J;
            }
            dS[i * N + j] = s;
        }
    }
    // dQ = dS * K
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++)
                sum += dS[i * N + j] * K[j * d + k];
            dQ[i * d + k] = sum * scale;
        }
    // dK = dS^T * Q
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++)
                sum += dS[j * N + i] * Q[j * d + k];
            dK[i * d + k] = sum * scale;
        }
    delete[] S;
    delete[] P;
    delete[] dP;
    delete[] dS;
}

void run_test(const char* test_name, float* Q, float* K, float* V, 
              int N, int d, int block_size, float* dO = nullptr, float* expected_dV = nullptr) {
    const int Br = block_size;
    const int Bc = block_size;
    const float scale = 1.0f / sqrtf(d);
        
    int Tc = (N + Bc - 1) / Bc;
    int Tr = (N + Br - 1) / Br;

    printf("\n\n%s\n", test_name);
    printf("================================\n");
    printf("N=%d, d=%d, Br=%d, Bc=%d, Tr=%d, Tc=%d, scale=%.4f\n", N, d, Br, Bc, Tr, Tc, scale);
    
    // Allocate memory
    float* O = new float[N * d];
    float* L = new float[N];
    float* dQ = new float[N * d];
    float* dK = new float[N * d];
    float* dV = new float[N * d];
    
    // Compute forward pass
    compute_forward_pass(Q, K, V, O, L, N, d, scale);
    
    // Initialize dO (gradient from upstream)
    bool default_dO = (dO == nullptr);

    if (dO == nullptr) {
        dO = new float[N * d];
        for (int i = 0; i < N * d; i++) {
            dO[i] = 1.0f;  // Unit gradient
        }
    }
    // Initialize gradients to zero
    memset(dQ, 0, N * d * sizeof(float));
    memset(dK, 0, N * d * sizeof(float));
    memset(dV, 0, N * d * sizeof(float));
    
    // Print inputs
    printf("\nInput Q:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            printf("%6.2f ", Q[i * d + j]);
        }
        printf("\n");
    }
    
    printf("\nInput K:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            printf("%6.2f ", K[i * d + j]);
        }
        printf("\n");
    }
    
    printf("\nInput V:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            printf("%6.2f ", V[i * d + j]);
        }
        printf("\n");
    }
    
    printf("\nOutput O (from forward pass):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            printf("%8.4f ", O[i * d + j]);
        }
        printf("\n");
    }
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_dO, *d_L;
    float *d_dQ, *d_dK, *d_dV;
    
    cudaMalloc((void **) &d_Q, N * d * sizeof(float));
    cudaMalloc((void **) &d_K, N * d * sizeof(float));
    cudaMalloc((void **) &d_V, N * d * sizeof(float));
    cudaMalloc((void **) &d_O, N * d * sizeof(float));
    cudaMalloc((void **) &d_dO, N * d * sizeof(float));
    cudaMalloc((void **) &d_L, N * sizeof(float));
    cudaMalloc((void **) &d_dQ, N * d * sizeof(float));
    cudaMalloc((void **) &d_dK, N * d * sizeof(float));
    cudaMalloc((void **) &d_dV, N * d * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_Q, Q, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, O, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, dO, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dQ, dQ, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dK, dK, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dV, dV, N * d * sizeof(float), cudaMemcpyHostToDevice);
    
    // Calculate shared memory size
    size_t smem_size = (2 * Bc * d +    // s_Kj, s_Vj
                    2 * Br * d +     // s_Qi, s_Oi
                    2 * Br * d +     // s_dOi, s_dQi
                    2 * Bc * d +     // s_dKj, s_dVj
                    4 * Br * Bc)     // s_Sij, s_Pij, s_dPij, s_dSij
                    * sizeof(float);

    dim3 threadsPerBlock = new_dim3(Br, 1, 1);
    dim3 blocksPerGrid = new_dim3(Tr, 1, 1);

    #ifndef __INTELLISENSE__
        flash_attention_kernel<<<blocksPerGrid, threadsPerBlock, smem_size>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_L, d_dQ, d_dK, d_dV, N, d, Br, Bc, scale
        );
    #endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(dQ, d_dQ, N * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dK, d_dK, N * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dV, d_dV, N * d * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print gradients
    printf("\nGradient dQ:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            printf("%8.4f ", dQ[i * d + j]);
        }
        printf("\n");
    }
    
    printf("\nGradient dK:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            printf("%8.4f ", dK[i * d + j]);
        }
        printf("\n");
    }
    
    printf("\nGradient dV:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            printf("%8.4f ", dV[i * d + j]);
        }
        printf("\n");
    }

    // Compare with naive backward
    float* naive_dQ = new float[N * d];
    float* naive_dK = new float[N * d];
    float* naive_dV = new float[N * d];
    naive_attention_backward(Q, K, V, O, L, dO, naive_dQ, naive_dK, naive_dV, N, d, scale);
    if (false){
        printf("\n[Naive] Gradient dQ:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < d; j++) {
                printf("%8.4f ", naive_dQ[i * d + j]);
            }
            printf("\n");
        }
        printf("\n[Naive] Gradient dK:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < d; j++) {
                printf("%8.4f ", naive_dK[i * d + j]);
            }
            printf("\n");
        }
        printf("\n[Naive] Gradient dV:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < d; j++) {
                printf("%8.4f ", naive_dV[i * d + j]);
            }
            printf("\n");
        }
    }
    // Validation
    auto check_match = [](const char* name, float* ref, float* test, int size) {
        bool passed = true;
        for (int i = 0; i < size; i++) {
            if (fabs(ref[i] - test[i]) > 1e-3) {
                printf("%s MISMATCH at %d: got %.4f, expected %.4f\n", name, i, test[i], ref[i]);
                passed = false;
            }
        }
        if (passed) printf("✓ PASSED: %s matches naive\n", name);
    };
    check_match("dQ", naive_dQ, dQ, N * d);
    check_match("dK", naive_dK, dK, N * d);
    check_match("dV", naive_dV, dV, N * d);

    // Verify against expected values if provided
    if (expected_dV != nullptr) {
        printf("\nValidation against expected dV:\n");
        bool passed = true;
        for (int i = 0; i < N * d; i++) {
            if (fabs(dV[i] - expected_dV[i]) > 1e-3) {
                printf("MISMATCH at index %d: got %.4f, expected %.4f\n", 
                       i, dV[i], expected_dV[i]);
                passed = false;
            }
        }
        if (passed) {
            printf("✓ PASSED: dV matches expected values\n");
        }
    }
    
    // Cleanup
    delete[] O;
    delete[] L;
    
    if (default_dO)
        delete[] dO;

    delete[] dQ;
    delete[] dK;
    delete[] dV;
    delete[] naive_dQ;
    delete[] naive_dK;
    delete[] naive_dV;
    
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_dO);
    cudaFree(d_L);
    cudaFree(d_dQ);
    cudaFree(d_dK);
    cudaFree(d_dV);
}

// Concrete test cases
void run_concrete_tests() {
    printf("\n\nRunning Concrete Test Cases\n");
    printf("===========================\n");
    
    // Test 0: Simple 2x4 test
    {
        float Q[] = {
            1, 0, 1, 0,
            0, 1, 0, 1
        };
        float K[] = {
            1, 1, 0, 0,
            0, 0, 1, 1
        };
        float V[] = {
            1, 2, 3, 4,
            5, 6, 7, 8
        };
        float dO[] = {
            1, 1, 0, 0,
            0, 0, 1, 1
        };
        run_test("Test 1: Simple 2x4", Q, K, V, 2, 4, 1, dO);
    }
    // Test 1: Simple 2x4 test
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
        run_test("Test 1: Simple 2x4", Q, K, V, 2, 4, 1);
    }
    // Test 2: Orthogonal vectors
    {
        float Q[] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
        float K[] = {
            0, 0, 0, 1,
            0, 0, 1, 0,
            0, 1, 0, 0,
            1, 0, 0, 0
        };
        float V[] = {
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4, 4, 4, 4
        };
        run_test("Test 2: Orthogonal Q/K", Q, K, V, 4, 4, 2);
    }
    
    // Test 3: All ones test
    {
        float Q[] = {
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1
        };
        float K[] = {
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1
        };
        float V[] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
        run_test("Test 3: All ones Q/K", Q, K, V, 4, 4, 2);
    }
    
    // Test 4: Single attention focus
    {
        float Q[] = {
            10, 0, 0, 0,
            0, 10, 0, 0,
            0, 0, 10, 0,
            0, 0, 0, 10
        };
        float K[] = {
            10, 0, 0, 0,
            0, 10, 0, 0,
            0, 0, 10, 0,
            0, 0, 0, 10
        };
        float V[] = {
            100, 0, 0, 0,
            0, 100, 0, 0,
            0, 0, 100, 0,
            0, 0, 0, 100
        };
        run_test("Test 4: Strong diagonal attention", Q, K, V, 4, 4, 2);
    }
    
    // Test 5: Alternating pattern
    {
        float Q[] = {
            1, 0, -1, 0,
            0, 1, 0, -1,
            -1, 0, 1, 0,
            0, -1, 0, 1
        };
        float K[] = {
            1, 0, -1, 0,
            0, 1, 0, -1,
            -1, 0, 1, 0,
            0, -1, 0, 1
        };
        float V[] = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        };
        run_test("Test 5: Alternating +/- pattern", Q, K, V, 4, 4, 2);
    }
    
    // Test 6: Larger test with pattern
    {
        float* Q = new float[8 * 8];
        float* K = new float[8 * 8];
        float* V = new float[8 * 8];
        
        // Create a pattern for Q and K
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                // Diagonal pattern for Q
                Q[i * 8 + j] = (i == j) ? 2.0f : 0.0f;
                // Shifted diagonal for K
                K[i * 8 + j] = ((i + 1) % 8 == j) ? 2.0f : 0.0f;
                // Incrementing values for V
                V[i * 8 + j] = i * 8 + j + 1;
            }
        }
        
        run_test("Test 6: 8x8 diagonal pattern", Q, K, V, 8, 8, 4);
        
        delete[] Q;
        delete[] K;
        delete[] V;
    }
    
    // Test 7: Zero test (edge case)
    {
        float Q[] = {
            0, 0, 0, 0,
            0, 0, 0, 0
        };
        float K[] = {
            0, 0, 0, 0,
            0, 0, 0, 0
        };
        float V[] = {
            1, 2, 3, 4,
            5, 6, 7, 8
        };
        run_test("Test 7: Zero Q/K (edge case)", Q, K, V, 2, 4, 2);
    }
    
    // Test 8: Single element dominance
    {
        float Q[] = {
            1000, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        };
        float K[] = {
            1000, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        };
        float V[] = {
            1, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        };
        run_test("Test 8: Single dominant element", Q, K, V, 4, 4, 2);
    }
}

int main() {
    printf("Testing FlashAttention-2 Backward Pass\n");
    printf("======================================\n");
    
    // Run all concrete test cases
    run_concrete_tests();
    
    return 0;
}