#include <cmath>
#include <cfloat>
#include <iostream>
#include <vector>


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

int main() {
    constexpr int N = 2;
    constexpr int d = 2;

    // Simple deterministic inputs.
    std::vector<float> Q = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    std::vector<float> K = Q;  // Identical for this test case.
    std::vector<float> V = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };
    std::vector<float> O(N * d, 0.0f);

    naive_attention(Q.data(), K.data(), V.data(), O.data(), N, d);

    const std::vector<float> expected = {
        1.6604769f, 2.6604770f,
        2.3395231f, 3.3395231f
    };

    bool passed = true;
    for (size_t i = 0; i < O.size(); ++i) {
        if (std::abs(O[i] - expected[i]) > 1e-4f) {
            std::cerr << "Mismatch at index " << i << ": got " << O[i]
                << ", expected " << expected[i] << '\n';
            passed = false;
        }
    }

    if (passed) {
        std::cout << "naive_attention test passed. Output:" << std::endl;
        for (int i = 0; i < N; ++i) {
            std::cout << "Row " << i << ": ";
            for (int j = 0; j < d; ++j) {
                std::cout << O[i * d + j] << (j + 1 == d ? '\n' : ' ');
            }
        }
        return 0;
    }

    std::cerr << "naive_attention test failed." << std::endl;
    return 1;
}