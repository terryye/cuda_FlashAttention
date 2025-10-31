#!/bin/bash

set -Eeuoa pipefail

# Check that there are 1 or 2 arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <path_to_cuda_file> [gpu_count]"
  exit 1
fi

# Check that $1 is a path to an existing file
if [ ! -f "$1" ]; then
  echo "Error: '$1' is not a valid file."
  exit 2
fi

#Check that if $2 is bound , else default to 4
if [ "$#" -lt 2 ] ; then
    gpu_count=4
else
    gpu_count=$2
fi

# Check that $2 is a number (integer or decimal)
if ! [[ "$gpu_count" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: '$gpu_count' is not a valid number."
  exit 3
fi

# Get the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

cd ../..

# Calculate relative path from repo root to script directory
RELATIVE_PATH=${SCRIPT_DIR#$(pwd)/}

modal run scripts/modal_mpi_gpu.py::compile_and_run_cuda_$gpu_count --code-path $RELATIVE_PATH/$1
