#!/bin/bash

set -Eeuao pipefail


# Check that there are exactly 2 arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <file_path> <mpi_world_size>"
  exit 1
fi

mpicc $1 -o /tmp/output.bin
mpirun -np $2 /tmp/output.bin
