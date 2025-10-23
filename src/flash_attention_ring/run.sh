#!/bin/bash

cd ../..
modal run scripts/modal_mpi_gpu.py::compile_and_run_cuda_4 --code-path week_07/00_cuda_attention_multiGPU/main.cu
