# FlashAttention-2 Backward Pass - Implementation Summary

## Overview

Successfully implemented the **FlashAttention-2 backward pass** based on **Section 3** of the FlashAttention-2 paper, with emphasis on:

-   **Section 3.2**: Parallelism strategy
-   **Section 3.3**: Work partitioning between warps

## Test Results ✓

### Test Case 1: Simple Backward Pass

**Configuration:**

-   Sequence length: 4
-   Head dimension: 4
-   Data: Identity-like patterns (easy to verify manually)

**Results:**

-   ✓ dQ gradient: Max diff = 8.94e-08 (< 1e-3 threshold)
-   ✓ dK gradient: Max diff = 8.94e-08 (< 1e-3 threshold)
-   ✓ dV gradient: Max diff = 2.98e-08 (< 1e-3 threshold)
-   **Status: PASSED ✓**

### Test Case 2: Complex Backward Pass

**Configuration:**

-   Sequence length: 128
-   Head dimension: 64
-   Softmax scale: 1/√64 = 0.125
-   Data: Random values in range [-0.5, 0.5]

**Results:**

-   ✓ dQ gradient: Max diff = 1.86e-09 (< 5e-3 threshold)
-   ✓ dK gradient: Max diff = 1.63e-09 (< 5e-3 threshold)
-   ✓ dV gradient: Max diff = 1.68e-08 (< 5e-3 threshold)
-   **Status: PASSED ✓**

## Implementation Details

### Algorithm (Based on FlashAttention-2 Algorithm 5)

The backward pass computes gradients for Q, K, V given:

-   **Inputs**: Q, K, V, O (output from forward), L (logsumexp), dO
-   **Outputs**: dQ, dK, dV

**Main Steps:**

1. **Initialize** (Lines 95-115 in kernel):

    - Load Q, O, dO tiles into registers for each warp
    - Compute D = rowsum(dO ⊙ O) for softmax gradient
    - Initialize dQ accumulator to zero

2. **Loop over K, V blocks** (Lines 126-224):

    ```
    For each column block j:
      a) Load K[j], V[j] to shared memory
      b) Compute S[i,j] = Q[i] @ K[j]^T * scale
      c) Compute P[i,j] = exp(S[i,j] - L[i])
      d) Compute dP[i,j] = dO[i] @ V[j]^T
      e) Compute dS[i,j] = P[i,j] * (dP[i,j] - D[i])
      f) Accumulate:
         - dQ[i] += dS[i,j] @ K[j] * scale
         - dK[j] += dS[i,j]^T @ Q[i] * scale (atomic)
         - dV[j] += P[i,j]^T @ dO[i] (atomic)
    ```

3. **Write output** (Lines 226-234):
    - Write final dQ values to global memory
    - dK and dV already written via atomic operations

### Section 3.2: Parallelism

**Block-level parallelism:**

-   Each thread block processes `Br = 32` rows
-   Grid size: `⌈N / Br⌉` blocks
-   Each block iterates over `⌈N / Bc⌉` column blocks (Bc = 32)

**Memory hierarchy:**

-   **Shared memory**: K_smem, V_smem, dK_smem, dV_smem (4 × Bc × d floats)
-   **Registers**: Q_reg, O_reg, dO_reg, dQ_acc per warp
-   **Global memory**: Final dQ, dK, dV outputs

### Section 3.3: Work Partitioning between Warps

**Warp assignment:**

-   `num_warps = 4` warps per block
-   Each warp handles `rows_per_warp = Br / num_warps = 8` rows
-   Warp ID determines row assignment: `warp_row_start = warp_id * rows_per_warp`

**Intra-warp cooperation:**

-   Threads collaborate on matrix operations using warp primitives
-   `__shfl_xor_sync()` for warp reductions (sum, max)
-   Each thread stores `cols_per_thread = ⌈d / WARP_SIZE⌉` columns

**Example (d=64, WARP_SIZE=32):**

```
cols_per_thread = 2
Thread 0: columns [0, 32]
Thread 1: columns [1, 33]
...
Thread 31: columns [31, 63]
```

## Key Optimizations

### 1. Memory Access Patterns

-   **Coalesced reads/writes**: Threads access consecutive memory locations
-   **Shared memory banking**: K, V tiles loaded once, reused by all warps
-   **Register blocking**: Q, O, dO kept in registers to minimize memory traffic

### 2. Warp-level Primitives

```cuda
// Warp reduction for sum
float warp_reduce_sum(float val) {
    for (int mask = WARP_SIZE/2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, mask);
    }
    return val;
}
```

-   Eliminates need for shared memory in reductions
-   Enables fine-grained parallelism within warps

### 3. Atomic Operations

-   Used for dK and dV updates (multiple rows contribute to same column)
-   Operations on shared memory first, then one write to global memory
-   Reduces global memory contention

### 4. Numerical Stability

-   Uses precomputed logsumexp (L) from forward pass
-   Avoids recomputing softmax: `P = exp(S - L)` instead of full softmax
-   Maintains precision through careful scaling

## Memory Layout

**Shared Memory** (per thread block):

```
┌────────────────┐
│ K_smem         │  Bc × d floats
├────────────────┤
│ V_smem         │  Bc × d floats
├────────────────┤
│ dK_smem        │  Bc × d floats (accumulator)
├────────────────┤
│ dV_smem        │  Bc × d floats (accumulator)
└────────────────┘
Total: 4 × Bc × d = 4 × 32 × 64 = 8192 floats = 32KB
```

**Register File** (per warp):

```
Q_reg[8][2]     - 8 rows × 2 cols per thread
O_reg[8][2]     - 8 rows × 2 cols per thread
dO_reg[8][2]    - 8 rows × 2 cols per thread
dQ_acc[8][2]    - 8 rows × 2 cols per thread
D[8]            - 8 scalars
Total per thread: 65 floats = 260 bytes
```

## Performance Characteristics

**Computational Complexity:**

-   Time: O(N² × d) same as naive attention
-   Memory: O(N × d) instead of O(N²) for attention matrix

**Memory Bandwidth:**

-   Each Q, K, V, O, dO element read once per outer loop iteration
-   K, V tiles reused across all rows in block
-   Shared memory reduces global memory traffic by ~Br factor

**Occupancy:**

-   4 warps × 32 threads = 128 threads per block
-   Shared memory: 32KB per block
-   Typical GPU: 32-48KB shared memory per SM → good occupancy

## Verification Strategy

### Naive Reference Implementation

Located in `util/naive_attention.h`:

```cpp
void naive_attention_backward(Q, K, V, O, L, dO, dQ, dK, dV, N, d, scale) {
    // Standard O(N²) backward pass
    1. Compute S = Q @ K^T * scale
    2. Compute P = softmax(S)
    3. Compute dV = P^T @ dO
    4. Compute dP = dO @ V^T
    5. Compute dS = dP ⊙ softmax_jacobian(P)
    6. Compute dQ = dS @ K * scale
    7. Compute dK = dS^T @ Q * scale
}
```

### Comparison Metrics

-   **Max difference**: Maximum absolute error across all elements
-   **Average difference**: Mean absolute error
-   **Large differences**: Count of errors > 1e-3

## Files Structure

```
week_05/03_cuda_attention2_backward_warps/
├── flash_attention_backward_kernel.cu   # CUDA kernel implementation
├── main.cu                              # Test harness with 2 test cases
├── run.sh                               # Build and run script
├── README.md                            # Documentation
└── IMPLEMENTATION_SUMMARY.md            # This file
```

## Building and Running

```bash
cd week_05/03_cuda_attention2_backward_warps
./run.sh
```

Or manually with Modal:

```bash
cd ../..
modal run scripts/modal_nvcc.py --code-path week_05/03_cuda_attention2_backward_warps/main.cu
```

## Future Enhancements

-   [ ] **Causal masking**: Support for autoregressive attention
-   [ ] **Mixed precision**: FP16/BF16 for A100/H100 GPUs
-   [ ] **Multi-query attention**: Share K, V across heads
-   [ ] **Fused kernels**: Combine backward with other operations
-   [ ] **Flash Decoding**: Specialized for inference
-   [ ] **Performance tuning**: Auto-tune Br, Bc for different GPUs

## References

1. **FlashAttention-2 Paper**: Dao (2023)

    - Section 3: Backward Pass Algorithm
    - Algorithm 5: Backward pass with tiling
    - Section 3.2: Parallelism
    - Section 3.3: Work partitioning between warps

2. **Original FlashAttention**: Dao et al. (2022)

3. **CUDA Programming Guide**: NVIDIA
    - Warp-level primitives
    - Shared memory optimization
    - Atomic operations

## Acknowledgments

Implementation based on the FlashAttention-2 paper by Tri Dao (2023) and the course materials from NEU HPC for AI.
