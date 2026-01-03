# FlashAttention-2 Backward Pass - Mathematical Formulas

## Notation

-   $N$: Sequence length
-   $d$: Head dimension
-   $\tau$: Softmax temperature/scale = $1/\sqrt{d}$
-   $\odot$: Element-wise multiplication
-   $\mathbf{1}$: Vector of ones

## Forward Pass (for reference)

### Attention Scores

$$\mathbf{S} = \tau \mathbf{Q} \mathbf{K}^T \in \mathbb{R}^{N \times N}$$

### Softmax (with numerical stability)

For row $i$:
$$m_i = \max_j S_{ij}$$

$$\ell_i = \sum_j e^{S_{ij} - m_i}$$

$$P_{ij} = \frac{e^{S_{ij} - m_i}}{\ell_i}$$

Alternative form using logsumexp:
$$L_i = m_i + \log(\ell_i) = \log\sum_j e^{S_{ij}}$$

$$P_{ij} = e^{S_{ij} - L_i}$$

### Output

$$\mathbf{O} = \mathbf{P} \mathbf{V} \in \mathbb{R}^{N \times d}$$

## Backward Pass

### Given

-   Forward outputs: $\mathbf{O}$, $\mathbf{L}$ (logsumexp values)
-   Upstream gradient: $\frac{\partial \mathcal{L}}{\partial \mathbf{O}} = d\mathbf{O}$
-   Original inputs: $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$

### Goal

Compute: $\frac{\partial \mathcal{L}}{\partial \mathbf{Q}}$, $\frac{\partial \mathcal{L}}{\partial \mathbf{K}}$, $\frac{\partial \mathcal{L}}{\partial \mathbf{V}}$

### Step 1: Compute D (rowsum statistic)

For each row $i$:
$$D_i = \sum_k (d\mathbf{O})_{ik} \cdot \mathbf{O}_{ik}$$

In matrix form:
$$\mathbf{D} = \text{rowsum}(d\mathbf{O} \odot \mathbf{O}) \in \mathbb{R}^N$$

**Why?** This is needed for the softmax gradient computation.

### Step 2: Gradient w.r.t. V

$$\frac{\partial \mathcal{L}}{\partial \mathbf{V}} = \mathbf{P}^T d\mathbf{O}$$

**Derivation:**
$$\mathbf{O} = \mathbf{P} \mathbf{V}$$
$$d\mathbf{O} = \mathbf{P} \, d\mathbf{V}$$
$$d\mathbf{V} = \mathbf{P}^T d\mathbf{O}$$

### Step 3: Gradient w.r.t. P

$$\frac{\partial \mathcal{L}}{\partial \mathbf{P}} = d\mathbf{O} \mathbf{V}^T$$

**Derivation:**
$$\mathbf{O} = \mathbf{P} \mathbf{V}$$
$$\frac{\partial \mathcal{L}}{\partial P_{ij}} = \sum_k \frac{\partial \mathcal{L}}{\partial O_{ik}} \frac{\partial O_{ik}}{\partial P_{ij}} = \sum_k (d\mathbf{O})_{ik} V_{jk}$$

In matrix form:
$$d\mathbf{P} = d\mathbf{O} \mathbf{V}^T$$

### Step 4: Gradient w.r.t. S (through softmax)

The softmax Jacobian for row $i$:

$$
\frac{\partial P_{ij}}{\partial S_{ik}} = \begin{cases}
P_{ij}(1 - P_{ik}) & \text{if } j = k \\
-P_{ij} P_{ik} & \text{if } j \neq k
\end{cases}
$$

This gives:
$$\frac{\partial \mathcal{L}}{\partial S_{ij}} = \sum_k (d\mathbf{P})_{ik} \frac{\partial P_{ik}}{\partial S_{ij}}$$

$$= (d\mathbf{P})_{ij} P_{ij}(1 - P_{ij}) + \sum_{k \neq j} (d\mathbf{P})_{ik} (-P_{ij} P_{ik})$$

$$= P_{ij} \left[ (d\mathbf{P})_{ij} - (d\mathbf{P})_{ij} P_{ij} - \sum_{k \neq j} (d\mathbf{P})_{ik} P_{ik} \right]$$

$$= P_{ij} \left[ (d\mathbf{P})_{ij} - \sum_k (d\mathbf{P})_{ik} P_{ik} \right]$$

Let $D_i = \sum_k (d\mathbf{P})_{ik} P_{ik}$, then:

$$\frac{\partial \mathcal{L}}{\partial S_{ij}} = P_{ij} \left[ (d\mathbf{P})_{ij} - D_i \right]$$

**Key insight:** Using the property that $\sum_k P_{ik} = 1$ and $d\mathbf{O} \mathbf{V}^T = d\mathbf{P}$:

$$D_i = \sum_k (d\mathbf{P})_{ik} P_{ik} = \sum_k \left(\sum_\ell (d\mathbf{O})_{i\ell} V_{k\ell}\right) P_{ik}$$
$$= \sum_\ell (d\mathbf{O})_{i\ell} \left(\sum_k P_{ik} V_{k\ell}\right) = \sum_\ell (d\mathbf{O})_{i\ell} O_{i\ell}$$

Therefore:
$$\boxed{d\mathbf{S} = \mathbf{P} \odot (d\mathbf{P} - \mathbf{D} \mathbf{1}^T)}$$

where $\mathbf{D} \mathbf{1}^T$ broadcasts $D_i$ to all columns in row $i$.

### Step 5: Gradient w.r.t. Q

$$\mathbf{S} = \tau \mathbf{Q} \mathbf{K}^T$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{Q}} = \tau \cdot d\mathbf{S} \mathbf{K}$$

### Step 6: Gradient w.r.t. K

$$\mathbf{S} = \tau \mathbf{Q} \mathbf{K}^T$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{K}} = \tau \cdot (d\mathbf{S})^T \mathbf{Q}$$

## Summary of Backward Pass Formulas

```
Input:  Q, K, V, O, L, dO
Output: dQ, dK, dV

1. Compute D = rowsum(dO ⊙ O)                    [N]
2. Recompute P = exp(τ Q K^T - L·1^T)            [N×N, but done blockwise]
3. Compute dV = P^T dO                            [N×d]
4. Compute dP = dO V^T                            [N×N]
5. Compute dS = P ⊙ (dP - D·1^T)                 [N×N]
6. Compute dQ = τ · dS K                          [N×d]
7. Compute dK = τ · dS^T Q                        [N×d]
```

## Implementation-Specific Formulas

### Tiled Computation

For block row $i$ (size $B_r$) and block column $j$ (size $B_c$):

$$\mathbf{S}_{ij} = \tau \mathbf{Q}_i (\mathbf{K}_j)^T \in \mathbb{R}^{B_r \times B_c}$$

$$\mathbf{P}_{ij} = \exp(\mathbf{S}_{ij} - \mathbf{L}_i \mathbf{1}^T) \in \mathbb{R}^{B_r \times B_c}$$

$$d\mathbf{Q}_i \mathrel{+}= \tau \cdot d\mathbf{S}_{ij} \mathbf{K}_j$$

$$d\mathbf{K}_j \mathrel{+}= \tau \cdot (d\mathbf{S}_{ij})^T \mathbf{Q}_i$$

$$d\mathbf{V}_j \mathrel{+}= (\mathbf{P}_{ij})^T d\mathbf{O}_i$$

### Numerical Stability

Always use the precomputed logsumexp $L_i$ from forward pass:

$$P_{ij} = \exp(S_{ij} - L_i)$$

Never recompute softmax from scratch, as this:

1. Wastes computation
2. May introduce numerical errors
3. Requires storing/recomputing max values

## Computational Complexity

### Forward Pass

-   Time: $O(N^2 d)$
-   Space: $O(Nd)$ with tiling (vs. $O(N^2)$ naive)

### Backward Pass

-   Time: $O(N^2 d)$
-   Space: $O(Nd)$ with tiling

### Memory Bandwidth (per tile)

-   Read: $Q_i$, $K_j$, $V_j$, $O_i$, $L_i$, $dO_i$
-   Write: $dQ_i$, $dK_j$, $dV_j$
-   Shared memory reuse reduces global memory traffic

## Key Differences from Naive Implementation

| Aspect             | Naive                             | FlashAttention-2           |
| ------------------ | --------------------------------- | -------------------------- |
| Store $\mathbf{P}$ | Yes ($N^2$)                       | No, recompute blockwise    |
| Store $\mathbf{S}$ | Yes ($N^2$)                       | No, compute on-the-fly     |
| D computation      | After computing all $d\mathbf{P}$ | Before main loop using $O$ |
| Memory             | $O(N^2)$                          | $O(Nd)$                    |
| Parallelism        | Row-parallel                      | Block + Warp-parallel      |

## Verification Formula

For numerical testing, the gradients should satisfy:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{Q}} \approx \frac{\mathcal{L}(\mathbf{Q} + \epsilon \mathbf{e}_i) - \mathcal{L}(\mathbf{Q})}{\epsilon}$$

for small $\epsilon$ and direction $\mathbf{e}_i$.

In practice, we verify against a trusted naive implementation:

-   Max absolute error < $10^{-3}$ for simple tests
-   Max absolute error < $5 \times 10^{-3}$ for complex tests
