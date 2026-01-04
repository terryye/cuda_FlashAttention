#pragma once
#include <iomanip>
#include "cuda_helper.h"
#include <iostream>

template <int rows_per_warp, int cols_per_thread>
__device__ __forceinline__ void load_Q_tile(
    const float* Q,
    float Q_reg[rows_per_warp][cols_per_thread],
    int global_row_start,
    int global_row_end,
    int warp_row_start,
    int lane_id,
    int d
) {
    // Load Q tile for this warp
    // each warp loads its assigned rows, each thread loads its assigned columns
    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        int global_row = global_row_start + warp_row_start + local_row; // enables each warp to load its assigned rows
        if (global_row < global_row_end) {
            // Each thread only loads its assigned columns
            for (int col_offset = 0; col_offset < cols_per_thread; col_offset++) {
                int col = lane_id + col_offset * WARP_SIZE;
                if (col < d) {
                    Q_reg[local_row][col_offset] = Q[global_row * d + col];
                }
                else {
                    Q_reg[local_row][col_offset] = 0.0f;
                }
            }
        }
        else {
            for (int col_offset = 0; col_offset < cols_per_thread; col_offset++) {
                Q_reg[local_row][col_offset] = 0.0f;
            }
        }
    }
}

template <int Bc, int cols_per_thread>
__device__ void process_kv_block(
    const float* K_smem,
    const float* V_smem,
    float Q_reg[][cols_per_thread],
    float O_acc[][cols_per_thread],
    float* m,
    float* l,
    int rows_per_warp,
    int actual_Bc,
    int d,
    float softmax_scale,
    int lane_id
) {

    const int S_row_size_per_thread = (Bc - 1 + WARP_SIZE) / WARP_SIZE;
    float S_row[S_row_size_per_thread];

    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        // Compute S = Q @ K^T
        float row_max = -INFINITY;

        for (int col = 0; col < actual_Bc; col++) {
            float dot = 0.0f;
            for (int k = lane_id; k < d; k += WARP_SIZE) {
                int q_col = k / WARP_SIZE;
                dot += Q_reg[local_row][q_col] * K_smem[col * d + k];
            }
            dot = warp_reduce_sum(dot);
            __syncwarp();

            if (col % WARP_SIZE == lane_id) {
                float v = dot * softmax_scale;
                S_row[col / WARP_SIZE] = v;
                row_max = fmaxf(row_max, v);
            }
        }
        // Compute row max across warp
        row_max = warp_reduce_max(row_max);
        __syncwarp();


        // Update statistics and rescale
        float m_new = fmaxf(m[local_row], row_max);
        float row_sum = 0.0f;

        // FIX: Only process valid entries in S_row
        int valid_S_entries = (actual_Bc + WARP_SIZE - 1) / WARP_SIZE;
        for (int col = 0; col < valid_S_entries; col++) {
            int actual_col = col * WARP_SIZE + lane_id;
            if (actual_col < actual_Bc) {
                S_row[col] = expf(S_row[col] - m_new);
                row_sum += S_row[col];
            }
            else {
                S_row[col] = 0.0f;  // Zero out invalid entries
            }
        }
        row_sum = warp_reduce_sum(row_sum);
        __syncwarp();

        float exp_diff = expf(m[local_row] - m_new);
        l[local_row] = exp_diff * l[local_row] + row_sum;
        m[local_row] = m_new;


        // Rescale existing output
        for (int col_offset = 0; col_offset < cols_per_thread; col_offset++) {
            int col = lane_id + col_offset * WARP_SIZE;
            if (col < d) {
                O_acc[local_row][col_offset] *= exp_diff;
            }
        }


        // Accumulate P @ V - FIX: Add bounds check
        for (int V_col = 0; V_col < d; V_col++) {
            float O_val = 0.0f;
            for (int k = 0; k < valid_S_entries; k++) {
                int v_row = k * WARP_SIZE + lane_id;
                if (v_row < actual_Bc) {  // ADD THIS CHECK!
                    O_val += S_row[k] * V_smem[v_row * d + V_col];
                }
            }
            O_val = warp_reduce_sum(O_val);
            __syncwarp();

            if (V_col % WARP_SIZE == lane_id) {
                O_acc[local_row][V_col / WARP_SIZE] += O_val;
            }
        }
    }
}



// Print a small matrix for debugging
void print_matrix(const char* name, const float* data, int rows, int cols, int max_rows = 4, int max_cols = 8) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < std::min(rows, max_rows); i++) {
        for (int j = 0; j < std::min(cols, max_cols); j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                << data[i * cols + j] << " ";
        }
        if (cols > max_cols) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > max_rows) std::cout << "..." << std::endl;
}

// Create simple test case similar to your main.cu
void create_simple_test_data(std::vector<float>& Q, std::vector<float>& K,
    std::vector<float>& V, int seq_len, int head_dim) {
    Q.resize(seq_len * head_dim);
    K.resize(seq_len * head_dim);
    V.resize(seq_len * head_dim);

    // Q matrix - identity-like pattern
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            Q[i * head_dim + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // K matrix - same as Q for predictable attention scores
    std::copy(Q.begin(), Q.end(), K.begin());

    // V matrix - distinct values for each row
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            V[i * head_dim + j] = i * 4.0f + j + 1.0f;
        }
    }
}
bool compare_outputs(const float* ref, const float* test, int size,
    float rtol = 1e-3, float atol = 1.0) {
    int diff_count = 0;
    float max_diff = 0.0f;
    int max_idx = 0;

    for (int i = 0; i < size; i++) {
        float diff = std::abs(ref[i] - test[i]);
        float rel_diff = diff / (std::abs(ref[i]) + 1e-8);

        if (rel_diff > rtol && diff > atol) {
            if (diff_count < 10) {  // Print first 10 differences
                printf("Diff at index %d: ref=%.4f, test=%.4f, diff=%.4f (rel=%.6f)\n",
                    i, ref[i], test[i], diff, rel_diff);
            }
            diff_count++;

            if (diff > max_diff) {
                max_diff = diff;
                max_idx = i;
            }
        }
    }

    if (diff_count > 0) {
        printf("Total significant differences: %d out of %d\n", diff_count, size);
        printf("Max difference: %.4f (rel=%.6f) at index %d\n",
            max_diff, max_diff / (std::abs(ref[max_idx]) + 1e-8), max_idx);
    }
    else {
        printf("All outputs match within tolerance (rtol=%.1e, atol=%.1f)\n", rtol, atol);
    }

    return diff_count == 0;
}