// flash_attention_2_backward.cu
#include <cmath>
#include <iostream>
#include <vector>
#include "./flash_attention_kernel_backward.cu"
// Naive forward implementation (for testing)
void naive_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim,
    float softmax_scale
) {
    std::vector<float> S(seq_len * seq_len);
    
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
}

// Naive backward implementation for testing
void naive_attention_backward(
    const float* Q,
    const float* K,
    const float* V,
    const float* O,
    const float* dO,
    float* dQ,
    float* dK,
    float* dV,
    int seq_len,
    int head_dim,
    float softmax_scale
) {
    std::vector<float> S(seq_len * seq_len);
    std::vector<float> P(seq_len * seq_len);
    
    // Recompute S and P
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
            P[i * seq_len + j] = expf(S[i * seq_len + j] - row_max);
            row_sum += P[i * seq_len + j];
        }
        
        for (int j = 0; j < seq_len; j++) {
            P[i * seq_len + j] /= row_sum;
        }
    }
    
    // Compute D = rowsum(dO * O)
    std::vector<float> D(seq_len);
    for (int i = 0; i < seq_len; i++) {
        D[i] = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            D[i] += dO[i * head_dim + j] * O[i * head_dim + j];
        }
    }
    
    // dV = P^T @ dO
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            dV[i * head_dim + j] = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                dV[i * head_dim + j] += P[k * seq_len + i] * dO[k * head_dim + j];
            }
        }
    }
    
    // dP = dO @ V^T
    std::vector<float> dP(seq_len * seq_len);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            dP[i * seq_len + j] = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                dP[i * seq_len + j] += dO[i * head_dim + k] * V[j * head_dim + k];
            }
        }
    }
    
    // dS = P * (dP - D)
    std::vector<float> dS(seq_len * seq_len);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            dS[i * seq_len + j] = P[i * seq_len + j] * (dP[i * seq_len + j] - D[i]);
        }
    }
    
    // dQ = dS @ K
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            dQ[i * head_dim + j] = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                dQ[i * head_dim + j] += dS[i * seq_len + k] * K[k * head_dim + j];
            }
            dQ[i * head_dim + j] *= softmax_scale;
        }
    }
    
    // dK = dS^T @ Q
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            dK[i * head_dim + j] = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                dK[i * head_dim + j] += dS[k * seq_len + i] * Q[k * head_dim + j];
            }
            dK[i * head_dim + j] *= softmax_scale;
        }
    }
}

// Test function
void test_flash_attention_2_backward() {
    // Test parameters
    const int seq_len = 256;  // Smaller for backward pass testing
    const int head_dim = 64;
    const float softmax_scale = 1.0f / sqrtf(head_dim);
    
    // Allocate host memory
    size_t qkv_size = seq_len * head_dim;
    std::vector<float> h_Q(qkv_size);
    std::vector<float> h_K(qkv_size);
    std::vector<float> h_V(qkv_size);
    std::vector<float> h_O(qkv_size);
    std::vector<float> h_dO(qkv_size);
    std::vector<float> h_L(seq_len);
    
    std::vector<float> h_dQ_naive(qkv_size);
    std::vector<float> h_dK_naive(qkv_size);
    std::vector<float> h_dV_naive(qkv_size);
    
    std::vector<float> h_dQ_flash(qkv_size);
    std::vector<float> h_dK_flash(qkv_size);
    std::vector<float> h_dV_flash(qkv_size);
    
    // Initialize with random values
    srand(42);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = (rand() % 1000) / 1000.0f - 0.5f;
        h_K[i] = (rand() % 1000) / 1000.0f - 0.5f;
        h_V[i] = (rand() % 1000) / 1000.0f - 0.5f;
        h_dO[i] = (rand() % 1000) / 1000.0f - 0.5f;
    }
    
    // First compute forward pass to get O and L
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, seq_len * sizeof(float));
    
    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Forward declaration - you would include the forward pass header here
    // For now, using naive forward to get O
    naive_attention_forward(h_Q.data(), h_K.data(), h_V.data(), h_O.data(), 
                          seq_len, head_dim, softmax_scale);
    
    // Compute L for the test
    for (int i = 0; i < seq_len; i++) {
        float row_max = -INFINITY;
        for (int j = 0; j < seq_len; j++) {
            float dot = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                dot += h_Q[i * head_dim + k] * h_K[j * head_dim + k];
            }
            row_max = std::max(row_max, dot * softmax_scale);
        }
        float row_sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            float dot = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                dot += h_Q[i * head_dim + k] * h_K[j * head_dim + k];
            }
            row_sum += expf(dot * softmax_scale - row_max);
        }
        h_L[i] = row_max + logf(row_sum);
    }
    
    cudaMemcpy(d_O, h_O.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_L.data(), seq_len * sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute naive backward
    naive_attention_backward(
        h_Q.data(), h_K.data(), h_V.data(), h_O.data(), h_dO.data(),
        h_dQ_naive.data(), h_dK_naive.data(), h_dV_naive.data(),
        seq_len, head_dim, softmax_scale
    );
    
    // Allocate device memory for gradients
    float *d_dO, *d_dQ, *d_dK, *d_dV;
    cudaMalloc(&d_dO, qkv_size * sizeof(float));
    cudaMalloc(&d_dQ, qkv_size * sizeof(float));
    cudaMalloc(&d_dK, qkv_size * sizeof(float));
    cudaMalloc(&d_dV, qkv_size * sizeof(float));
    
    cudaMemcpy(d_dO, h_dO.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run FlashAttention-2 backward
    flash_attention_2_backward(
        d_Q, d_K, d_V, d_O, d_dO, d_L,
        d_dQ, d_dK, d_dV,
        seq_len, head_dim, softmax_scale
    );
    
    cudaDeviceSynchronize();
    
    // Copy back to host
    cudaMemcpy(h_dQ_flash.data(), d_dQ, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK_flash.data(), d_dK, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV_flash.data(), d_dV, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare results
    auto check_gradient = [&](const std::vector<float>& naive, const std::vector<float>& flash, const std::string& name) {
        float max_diff = 0.0f;
        float avg_diff = 0.0f;
        int num_large_diff = 0;
        
        for (size_t i = 0; i < qkv_size; i++) {
            float diff = std::abs(naive[i] - flash[i]);
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
            if (diff > 1e-2) {
                num_large_diff++;
            }
        }
        avg_diff /= qkv_size;
        
        std::cout << name << " Results:" << std::endl;
        std::cout << "  Max difference: " << max_diff << std::endl;
        std::cout << "  Avg difference: " << avg_diff << std::endl;
        std::cout << "  Large differences (>1e-2): " << num_large_diff << std::endl;
        std::cout << "  " << (max_diff < 1e-1 ? "PASSED" : "FAILED") << std::endl << std::endl;
    };
    
    check_gradient(h_dQ_naive, h_dQ_flash, "dQ");
    check_gradient(h_dK_naive, h_dK_flash, "dK");
    check_gradient(h_dV_naive, h_dV_flash, "dV");
    
    // Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
    cudaFree(d_dO); cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);
}

int main() {
    test_flash_attention_2_backward();
    return 0;
}