#pragma once

#include <stdio.h>
#include <iostream>

// Helper function to compute forward pass for testing
void naive_forward_pass(const float* Q, const float* K, const float* V,
    float* O, float* L, int N, int d, float scale = 0) {
    if (scale == 0) scale = 1.0f / sqrtf(d);
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
        if (L != nullptr)
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

// Naive attention implementation for comparison, compatible with the L output version
void naive_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim,
    float softmax_scale
) {
    naive_forward_pass(
        Q, K, V, O,
        nullptr,// No log-sum-exp output needed here
        seq_len, head_dim,
        softmax_scale
    );
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
