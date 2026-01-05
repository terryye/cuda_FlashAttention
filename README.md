# CUDA Flash Attention Implementations

A comprehensive, educational implementation of attention mechanisms from scratch, including naive attention, Flash Attention V1, Flash Attention V2, and distributed ring-based variants with MPI and NCCL.

## Overview

This project demonstrates the evolution of attention mechanisms used in transformer models, progressing from a naive CPU baseline to highly optimized GPU implementations with distributed multi-GPU communication. Each implementation is self-contained and can be studied independently.

### Development Environment

-   Visual Studio Code with Remote - SSH extension for remote development
-   Nsight Visual Studio Code Edition for CUDA debugging and profiling , ** both local and remote **
-   The .vscode and headers files have configurations for local development in MacOS systems and remote compilation/debug in Linux systems with NVIDIA GPUs.

## Prerequisites

### For Modal (Cloud Execution)

-   Python 3.10+ with pip
-   Modal account and CLI: `pip install modal && python3 -m modal setup`

### For Local Debug Execution (Self Managed linux system with NVIDIA GPUs, e.g. Vast.ai **kvm** instances, or your own server)

-   NVIDIA CUDA Toolkit 12.x or 13.x
-   NVIDIA NCCL library
-   OpenMPI 4.x
-   C++17 compatible compiler (clang++ or g++)
-   NVIDIA NVSHMEM (optional, for advanced implementations)
-   NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
-   run `scripts/install.sh` to set up the linux environment

## Getting Started

### Quick Start

```bash
# Navigate to any CUDA implementation
cd src/01_flash_attention_v1

# Run on Modal (Cloud)
./run.sh
# Run on self-managed local GPU system
./run_local.sh
```

for ring attention,

```bash
./run.sh 00 # will run 00_mpi_vecadd.cu only
./run.sh 01 # will run 01_nccl_verify.cu only
# ... etc.

./run.sh # will run 00 - 04 implementations one by one

./run_local.sh # will only run 03_attention_1GPU.cu on single GPU local system. you can modify the script to run other implementations.
```

## Implementations

### 00_naive_attention

CPU baseline implementation of scaled dot-product attention for correctness verification.

-   Single-threaded CPU implementation
-   Useful as a reference for validating other implementations
-   Run: `cd src/00_naive_attention && ./run.sh`
-   Reference: [Attention is all you need](https://arxiv.org/abs/1706.03762)

### 01_flash_attention_v1

First version of Flash Attention with I/O efficiency optimizations.

-   Block-wise computation to improve cache locality
-   Reduced memory I/O compared to naive attention
-   Reference: [Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

### 02_flash_attention_v2

-   Complete implementation of Flash Attention V2, including both forward and backward passes.
-   Warp-level optimizations for improved performance.
-   02_flash_attention_v2_forward: Forward pass implementation of Flash Attention V2.
-   02_flash_attention_v2_backward: Backward pass (gradient computation) for Flash Attention V2.
-   Reference: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
    ](https://arxiv.org/abs/2307.08691) Becareful with the typos of the algorithm description in the paper. I marked those typos in the attached [paper file](./papers/flash%20attention2.pdf)

### 03_flash_attention_v2_ring

Distributed Flash Attention V2 using ring communication pattern.

-   MPI and NCCL support for multi-GPU/multi-node computation
-   Ring-based communication for efficient distributed attention
-   Reference: [Ring Attention with Blockwise Transformers for Near-Infinite Context
    ](https://arxiv.org/abs/2310.01889)

## Project Structure

```
src/
├── 00_naive_attention/      # CPU baseline
├── 01_flash_attention_v1/   # Flash Attention V1
├── 02_flash_attention_v2_backward/  # FA V2 backward pass
├── 02_flash_attention_v2_forward/   # FA V2 forward pass
├── 03_flash_attention_v2_ring/      # Distributed FA V2
└── util/                    # Utility functions and helpers

headers/                      # CUDA headers for vscode intellisense. refers to https://github.com/terryye/cuda_vscode for more details
scripts/                      # Installation and testing scripts. refers to https://github.com/terryye/cuda_vscode for more details
```
