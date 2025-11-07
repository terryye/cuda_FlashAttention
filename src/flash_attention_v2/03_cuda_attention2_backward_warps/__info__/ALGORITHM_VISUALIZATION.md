# FlashAttention-2 Backward Pass - Algorithm Visualization

## Thread Block Organization

```
┌─────────────────────────────────────────────────────────────────┐
│                     Thread Block (Br = 32 rows)                  │
├─────────────────────────────────────────────────────────────────┤
│  Warp 0 (32 threads) → processes rows 0-7                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ T0  T1  T2  ... T31                                 │       │
│  │ Each thread handles cols_per_thread=2 columns        │       │
│  └──────────────────────────────────────────────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│  Warp 1 (32 threads) → processes rows 8-15                      │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ T32 T33 T34 ... T63                                 │       │
│  └──────────────────────────────────────────────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│  Warp 2 (32 threads) → processes rows 16-23                     │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ T64 T65 T66 ... T95                                 │       │
│  └──────────────────────────────────────────────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│  Warp 3 (32 threads) → processes rows 24-31                     │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ T96 T97 T98 ... T127                                │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Computation Flow (Per Block)

```
Input:  Q[Br×d], K[N×d], V[N×d], O[Br×d], L[Br], dO[Br×d]
Output: dQ[Br×d], dK[N×d], dV[N×d]

┌─────────────────────────────────────────────────────────────┐
│ Step 1: Load to Registers (Parallel across warps)           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Warp 0:  Q_reg[0:7][:]  ← Q[0:7][:]                       │
│           O_reg[0:7][:]  ← O[0:7][:]                       │
│           dO_reg[0:7][:] ← dO[0:7][:]                      │
│           D[0:7] = rowsum(dO_reg ⊙ O_reg)                  │
│                                                             │
│  Warp 1:  Q_reg[8:15][:] ← Q[8:15][:]                      │
│           ... (similar)                                     │
│                                                             │
│  Warp 2, 3: (similar for their row ranges)                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 2: Loop over K, V column blocks                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For j = 0 to ⌈N/Bc⌉:  (Bc = 32)                           │
│                                                             │
│    2a. Load to Shared Memory (All threads collaborate)      │
│        ┌────────────────────────────┐                      │
│        │ K_smem  ← K[j*Bc:(j+1)*Bc] │                      │
│        │ V_smem  ← V[j*Bc:(j+1)*Bc] │                      │
│        │ dK_smem ← zeros            │                      │
│        │ dV_smem ← zeros            │                      │
│        └────────────────────────────┘                      │
│        __syncthreads()                                      │
│                                                             │
│    2b. Compute per Warp (Each warp processes its rows)      │
│                                                             │
│        For each local_row in warp's rows:                   │
│                                                             │
│          ┌─────────────────────────────────────┐           │
│          │ S = Q_reg @ K_smem^T * scale        │           │
│          │   (via warp reduction)              │           │
│          └─────────────────────────────────────┘           │
│                                                             │
│          ┌─────────────────────────────────────┐           │
│          │ P = exp(S - L)                      │           │
│          │   (use precomputed L)               │           │
│          └─────────────────────────────────────┘           │
│                                                             │
│          ┌─────────────────────────────────────┐           │
│          │ dP = dO_reg @ V_smem^T              │           │
│          │    (via warp reduction)             │           │
│          └─────────────────────────────────────┘           │
│                                                             │
│          ┌─────────────────────────────────────┐           │
│          │ dS = P ⊙ (dP - D)                   │           │
│          │    (softmax gradient)               │           │
│          └─────────────────────────────────────┘           │
│                                                             │
│          ┌─────────────────────────────────────┐           │
│          │ dQ_acc += dS @ K_smem * scale       │           │
│          │          (accumulate in registers)  │           │
│          └─────────────────────────────────────┘           │
│                                                             │
│          ┌─────────────────────────────────────┐           │
│          │ dK_smem += dS^T @ Q_reg * scale     │           │
│          │           (atomic add to smem)      │           │
│          └─────────────────────────────────────┘           │
│                                                             │
│          ┌─────────────────────────────────────┐           │
│          │ dV_smem += P^T @ dO_reg             │           │
│          │           (atomic add to smem)      │           │
│          └─────────────────────────────────────┘           │
│                                                             │
│    2c. Write dK, dV to Global Memory                        │
│        __syncthreads()                                      │
│        dK[j*Bc:(j+1)*Bc] += dK_smem  (atomic)              │
│        dV[j*Bc:(j+1)*Bc] += dV_smem  (atomic)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 3: Write dQ to Global Memory                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Each warp writes its rows:                                 │
│    Warp 0: dQ[0:7][:]   ← dQ_acc[0:7][:]                   │
│    Warp 1: dQ[8:15][:]  ← dQ_acc[8:15][:]                  │
│    Warp 2: dQ[16:23][:] ← dQ_acc[16:23][:]                 │
│    Warp 3: dQ[24:31][:] ← dQ_acc[24:31][:]                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Warp-Level Computation Detail (Example: Computing S = Q @ K^T)

```
Goal: Compute dot product between Q_reg[local_row] and K_smem[col]

Thread Layout (for d=64, WARP_SIZE=32):
┌────────────────────────────────────────────────────────┐
│ Thread 0:  handles dimensions [0, 32]                  │
│ Thread 1:  handles dimensions [1, 33]                  │
│ Thread 2:  handles dimensions [2, 34]                  │
│ ...                                                    │
│ Thread 31: handles dimensions [31, 63]                 │
└────────────────────────────────────────────────────────┘

Step 1: Partial Sums
─────────────────────
T0:  dot = Q_reg[local_row][0] * K_smem[col][0]
           + Q_reg[local_row][1] * K_smem[col][32]

T1:  dot = Q_reg[local_row][0] * K_smem[col][1]
           + Q_reg[local_row][1] * K_smem[col][33]
...

Step 2: Warp Reduction (using __shfl_xor_sync)
────────────────────────────────────────────────
Iteration 1 (mask=16):
  Each thread XORs with thread 16 away
  T0  ←→ T16,  T1  ←→ T17,  ...
  dot += __shfl_xor_sync(FULL_MASK, dot, 16)

Iteration 2 (mask=8):
  Each thread XORs with thread 8 away
  T0  ←→ T8,   T1  ←→ T9,   ...
  dot += __shfl_xor_sync(FULL_MASK, dot, 8)

Iteration 3 (mask=4):
  dot += __shfl_xor_sync(FULL_MASK, dot, 4)

Iteration 4 (mask=2):
  dot += __shfl_xor_sync(FULL_MASK, dot, 2)

Iteration 5 (mask=1):
  dot += __shfl_xor_sync(FULL_MASK, dot, 1)

Result: All threads now have the same final sum value
        S[local_row][col] = dot * softmax_scale
```

## Memory Access Pattern

```
Global Memory Layout:
┌─────────────────────────────────────────────┐
│ Q:  [N × d]  →  Read once per block         │
│ K:  [N × d]  →  Read in Bc-sized tiles      │
│ V:  [N × d]  →  Read in Bc-sized tiles      │
│ O:  [N × d]  →  Read once per block         │
│ L:  [N]      →  Read once per block         │
│ dO: [N × d]  →  Read once per block         │
├─────────────────────────────────────────────┤
│ dQ: [N × d]  →  Written once per block      │
│ dK: [N × d]  →  Atomic adds from all blocks │
│ dV: [N × d]  →  Atomic adds from all blocks │
└─────────────────────────────────────────────┘

Shared Memory per Block:
┌─────────────────────────────────────────────┐
│ K_smem:  [Bc × d] = [32 × 64] = 2048 floats │
│ V_smem:  [Bc × d] = [32 × 64] = 2048 floats │
│ dK_smem: [Bc × d] = [32 × 64] = 2048 floats │
│ dV_smem: [Bc × d] = [32 × 64] = 2048 floats │
│ Total: 8192 floats = 32KB                   │
└─────────────────────────────────────────────┘

Register File per Thread:
┌─────────────────────────────────────────────┐
│ Q_reg:   [rows_per_warp][cols_per_thread]   │
│          [8][2] = 16 floats                 │
│ O_reg:   [8][2] = 16 floats                 │
│ dO_reg:  [8][2] = 16 floats                 │
│ dQ_acc:  [8][2] = 16 floats                 │
│ D:       [8] = 8 floats                     │
│ S_row:   [Bc] = 32 floats (temp)            │
│ P_row:   [Bc] = 32 floats (temp)            │
│ dP_row:  [Bc] = 32 floats (temp)            │
│ dS_row:  [Bc] = 32 floats (temp)            │
│ Total: ~160 floats = 640 bytes              │
└─────────────────────────────────────────────┘
```

## Synchronization Points

```
Timeline of a single block:
┌────────────────────────────────────────────────────────┐
│ Load Q, O, dO to registers                             │
│ Compute D (warp-level, uses __syncwarp)                │
├────────────────────────────────────────────────────────┤
│ FOR each column block j:                               │
│   │                                                     │
│   ├─ __syncthreads() ──────────────────────────────────┤
│   │  Load K, V to shared memory                        │
│   │  Initialize dK_smem, dV_smem                       │
│   │                                                     │
│   ├─ __syncthreads() ──────────────────────────────────┤
│   │  Compute S, P, dP, dS (per warp)                   │
│   │    (uses __syncwarp for reductions)                │
│   │  Accumulate dQ (registers)                         │
│   │  Accumulate dK_smem, dV_smem (atomic)              │
│   │                                                     │
│   ├─ __syncthreads() ──────────────────────────────────┤
│   │  Write dK_smem, dV_smem to global (atomic)         │
│   │                                                     │
├────────────────────────────────────────────────────────┤
│ Write dQ to global memory                              │
└────────────────────────────────────────────────────────┘
```

## Data Dependencies

```
Forward Pass Outputs → Backward Pass Inputs
┌──────────┐      ┌──────────┐
│ O [N×d]  │ ───→ │          │
└──────────┘      │          │
┌──────────┐      │ Backward │
│ L [N]    │ ───→ │  Kernel  │
└──────────┘      │          │
                  │          │
User Inputs       │          │
┌──────────┐      │          │
│ Q [N×d]  │ ───→ │          │
│ K [N×d]  │ ───→ │          │
│ V [N×d]  │ ───→ │          │
│ dO[N×d]  │ ───→ │          │
└──────────┘      └────┬─────┘
                       │
                       ↓
Backward Pass Outputs
┌──────────┐
│ dQ [N×d] │ ← Gradient w.r.t Q
│ dK [N×d] │ ← Gradient w.r.t K
│ dV [N×d] │ ← Gradient w.r.t V
└──────────┘
```

## Numerical Example (Simplified: N=4, d=4, Br=Bc=2)

```
Input:
Q = [[1, 0, 0, 0],     K = [[1, 0, 0, 0],     V = [[1, 1, 1, 1],
     [0, 1, 0, 0],          [0, 1, 0, 0],          [2, 2, 2, 2],
     [0, 0, 1, 0],          [0, 0, 1, 0],          [3, 3, 3, 3],
     [0, 0, 0, 1]]          [0, 0, 0, 1]]          [4, 4, 4, 4]]

dO = [[1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]]

Block 0 processes rows 0-1:
  Column block 0 (cols 0-1):
    S[0,0] = 1, S[0,1] = 0  →  P[0,0] ≈ 0.73, P[0,1] ≈ 0.27
    S[1,0] = 0, S[1,1] = 1  →  P[1,0] ≈ 0.27, P[1,1] ≈ 0.73

    dP from dO @ V^T, then dS from P ⊙ (dP - D)

    Accumulate:
      dQ[0] += dS[0] @ K[0:1] * scale
      dK[0:1] += dS[0:1]^T @ Q[0:1] * scale
      dV[0:1] += P[0:1]^T @ dO[0:1]

  Column block 1 (cols 2-3):
    (similar computation)

Block 1 processes rows 2-3:
  (similar to Block 0)

Final gradients computed through accumulation across all blocks.
```
