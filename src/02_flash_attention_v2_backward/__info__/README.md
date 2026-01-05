# FlashAttention-2 Backward Pass Implementation

This directory contains the implementation of the **FlashAttention-2 backward pass** based on **Section 3** of the FlashAttention-2 paper.

## Implementation Details

### Architecture Overview

The implementation follows the algorithm described in the paper with these key features:

#### Section 3.2: Parallelism

-   **Thread Block Level**: Each thread block handles `Br` rows of the attention matrix
-   **Column Tiling**: Loops over `Bc` columns for K and V tiles
-   **Shared Memory**: Uses shared memory for K, V tiles and accumulates gradients for dK, dV

#### Section 3.3: Work Partitioning between Warps

-   **Warp Assignment**: Warps within a block work on different rows
-   **Row Distribution**: Each warp computes partial gradients for its assigned rows (`rows_per_warp = Br / num_warps`)
-   **Intra-Warp Cooperation**: Threads within a warp collaborate on:
    -   Computing attention scores (S = Q @ K^T)
    -   Softmax probabilities (P)
    -   Gradient propagation (dP, dS)

### Kernel Parameters

-   `Br = 32`: Row block size
-   `Bc = 32`: Column block size
-   `num_warps = 4`: Number of warps per thread block
-   `WARP_SIZE = 32`: CUDA warp size
-   `d_max = 128`: Maximum head dimension supported

### Algorithm Flow

The backward pass computes gradients for Q, K, V given:

-   **Inputs**: Q, K, V, O (output), L (logsumexp), dO (gradient of output)
-   **Outputs**: dQ, dK, dV

**Key Steps**:

1. **Load tiles**: Each warp loads its portion of Q, O, dO
2. **Compute D**: `D = rowsum(dO ⊙ O)` for each row (needed for softmax gradient)
3. **Loop over K,V blocks**:
    - Compute attention scores: `S = Q @ K^T * scale`
    - Compute probabilities: `P = softmax(S)` using precomputed L
    - Compute gradient: `dP = dO @ V^T`
    - Compute softmax gradient: `dS = P ⊙ (dP - D)`
    - Accumulate gradients:
        - `dQ += dS @ K * scale`
        - `dK += dS^T @ Q * scale` (atomic adds to shared memory)
        - `dV += P^T @ dO` (atomic adds to shared memory)
4. **Write results**: Write dQ, dK, dV to global memory

### Memory Optimization

**Shared Memory Layout** (per block):

```
K_smem:  [Bc × d]  - K tile
V_smem:  [Bc × d]  - V tile
dK_smem: [Bc × d]  - dK accumulator
dV_smem: [Bc × d]  - dV accumulator
Total: 4 × Bc × d floats
```

**Register Usage** (per warp):

-   `Q_reg[rows_per_warp][cols_per_thread]`: Q values
-   `O_reg[rows_per_warp][cols_per_thread]`: O values
-   `dO_reg[rows_per_warp][cols_per_thread]`: dO values
-   `dQ_acc[rows_per_warp][cols_per_thread]`: dQ accumulator

### Synchronization Strategy

-   **`__syncthreads()`**: Before/after shared memory operations
-   **`__syncwarp()`**: For warp-level reductions
-   **Atomic operations**: For concurrent updates to dK_smem and dV_smem

## Files

-   **`flash_attention_backward_kernel.cu`**: CUDA kernel implementation

    -   `flash_attention_2_backward_kernel()`: Main backward pass kernel
    -   `flash_attention_2_backward()`: Host wrapper function
    -   Warp reduction utilities (`warp_reduce_sum`, `warp_reduce_max`)

-   **`main.cu`**: Test harness with two test cases

    -   `test_simple_backward()`: Simple test with identity-like patterns
    -   `test_complex_backward()`: Complex test with random data (128 × 64)
    -   Naive backward implementation for verification

-   **`run.sh`**: Script to compile and run on Modal

## Test Cases

### Test Case 1: Simple Backward Pass

-   **Size**: 4 × 4 sequences, 4-dimensional heads
-   **Data**: Identity-like patterns for Q, K; simple incremental values for V
-   **Purpose**: Manual verification with easy-to-calculate values
-   **Pass Criteria**: Max difference < 1e-3

### Test Case 2: Complex Backward Pass

-   **Size**: 128 × 128 sequences, 64-dimensional heads
-   **Data**: Random values in range [-0.5, 0.5]
-   **Purpose**: Realistic scenario with complex attention patterns
-   **Pass Criteria**: Max difference < 5e-3

## Running the Tests

```bash
./run.sh
```

## Expected Output

The test outputs will show:

-   Input matrices (Q, K, V, dO)
-   Forward pass output (O)
-   Computed gradients from both naive and FlashAttention-2 implementations
-   Comparison metrics:
    -   Maximum difference
    -   Average difference
    -   Number of large differences
-   Pass/Fail status for each test case

## Implementation Highlights

### 1. Efficient Memory Access

-   Coalesced global memory reads/writes
-   Shared memory for frequently accessed K, V tiles
-   Register blocking for Q, O, dO to minimize memory traffic

### 2. Warp-Level Primitives

-   `__shfl_xor_sync()` for warp reductions (sum and max)
-   Eliminates need for shared memory in reductions
-   Enables fine-grained parallelism within warps

### 3. Numerical Stability

-   Uses precomputed logsumexp (L) from forward pass
-   Avoids recomputing softmax normalization constants
-   Maintains precision through careful scaling

### 4. Gradient Accumulation

-   Atomic adds for dK and dV updates (multiple rows contribute)
-   Direct writes for dQ (each row computed independently)
-   Minimizes global memory contention

## Performance Considerations

-   **Block-level tiling**: Reduces global memory bandwidth
-   **Warp-level parallelism**: Maximizes instruction throughput
-   **Shared memory reuse**: K, V loaded once per block iteration
-   **Register pressure**: Carefully tuned to avoid spilling

## References

-   **FlashAttention-2 Paper**: [Dao, 2023] - Section 3 (Backward Pass)
-   Algorithm 5: Backward pass with tiling
-   Figure descriptions for parallelism and work partitioning

## Future Improvements

-   [ ] Support for causal masking
-   [ ] Mixed precision (FP16/BF16)
-   [ ] Multi-head attention batching
-   [ ] Fused backward pass with activation functions
-   [ ] Performance profiling and optimization
