
#include <stdio.h>
#include <iostream>
#include "../../util/cuda_shim.h"
#include "./flash_attention_kernel.cu"
/**
 * Flash Attention implementation in CUDA
 * Reference: https://arxiv.org/abs/2205.14135
 * in flash attention one, we try to maxmize the Bc, because each iteration will put Oi back to HBM
 * 
 */
// Host function to launch Flash Attention
void flash_attention(
    const float* d_Q,
    const float* d_K, 
    const float* d_V,
    float* d_O,
    float* d_L,
    int N,
    int d,
    int Br = 32, // Block size for rows
    int Bc = 32  // Block size for columns
) {
    
    //line 3: Calculate Tr = ceil(N/Br), Tc = ceil(N/Bc)
    // Calculate grid and block dimensions
    int Tr = (N + Br - 1) / Br; //Q, O, l
    int Tc = (N + Bc - 1) / Bc; //K, V  no need to use Tc here in this implementation, just for understanding
    
    printf("Flash Attention launch with Br=%d, Bc=%d, Tr=%d, Tc=%d\n", Br, Bc, Tr, Tc);
    
    // line 7: for 1 <= i <= Tr do  block and thread implement this loop
    dim3 threadsPerBlock = new_dim3(Br, 1, 1);  // Each block has Br threads, each thread handles one row of Qi
    dim3 blocksPerGrid = new_dim3(Tr, 1, 1);    // Each block handles one tile of Qi

    // Calculate shared memory size. for simplicity, we do not reuse shared memory for different variables
    size_t shared_mem_size = sizeof(float) * (
        Br * d +      // Qi             
        Bc * d +      // Kj
        Bc * d +      // Vj
        Br * Bc +     // S
        Br * Bc +     // P
        Br * d      // Oi (in SRAM).  
    );

    // Launch kernel
    #ifndef __INTELLISENSE__
    flash_attention_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(
        d_Q, d_K, d_V, d_O, d_L, N, d, Br, Bc, 1.0f / sqrtf((float)d)
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
void run_test(const char* test_name, float* Q, float* K, float* V, int N, int d, int Br, int Bc = 8) {
    std::cout << "\n=== Test: " << test_name << " ===" << std::endl;
    std::cout << "N=" << N << ", d=" << d << ", Bc=" << Bc << std::endl;
    
    // Allocate output memory
    float *h_O = new float[N * d];
    float *h_L = new float[N];
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc((void**)&d_Q, N * d * sizeof(float));
    cudaMalloc((void**)&d_K, N * d * sizeof(float));
    cudaMalloc((void**)&d_V, N * d * sizeof(float));
    cudaMalloc((void**)&d_O, N * d * sizeof(float));
    cudaMalloc((void**)&d_L, N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_Q, Q, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * d * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run Flash Attention
    flash_attention(d_Q, d_K, d_V, d_O, d_L, N, d, Br, Bc);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_L, d_L, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "Output O:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < d && j < 8; j++) {  // Print first 8 values
            std::cout << h_O[i * d + j] << " ";
        }
        if (d > 8) std::cout << "...";
        std::cout << " | L=" << h_L[i] << std::endl;
    }
    
    // Cleanup
    delete[] h_O;
    delete[] h_L;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
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
        run_test("Simple 2x4 test", Q, K, V, 2, 4, 2, 2);
    } 
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
        run_test("Simple 2x4 test", Q, K, V, 2, 4, 1, 1);
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