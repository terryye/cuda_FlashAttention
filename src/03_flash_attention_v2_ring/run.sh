#!/bin/bash
N_GPU=2

srun () {
    printf '+ %s\n' "$*"
    "$@"
}

# if intval $1 == 0  or no argument passed, run vecadd test
if [ -z "$1" ] || [ "$1" -eq 0 ]; then
    srun modal run ../../scripts/modal_mpi.py::compile_and_run_cuda_$N_GPU  --code-path ./00_mpi_vecadd.cu
fi
# else if intval $1 == 1 then run nccl verify test
if [ -z "$1" ] || [ "$1" -eq 1 ]; then
    srun modal run ../../scripts/modal_mpi.py::compile_and_run_cuda_$N_GPU  --code-path ./01_nccl_verify.cu
fi

if [ -z "$1" ] || [ "$1" -eq 2 ]; then
    srun modal run ../../scripts/modal_mpi.py::compile_and_run_cuda_$N_GPU  --code-path ./02_overlap.cu
fi

if [ -z "$1" ] || [ "$1" -eq 3 ]; then
    srun modal run ../../scripts/modal_mpi.py::compile_and_run_cuda_$N_GPU  --code-path ./03_attention_1GPU.cu
fi

if [ -z "$1" ] || [ "$1" -eq 4 ]; then
    srun modal run ../../scripts/modal_mpi.py::compile_and_run_cuda_$N_GPU  --code-path ./04_ring_attention.cu
fi