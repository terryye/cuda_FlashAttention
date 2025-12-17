#!/bin/bash

# Get the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# Calculate relative path from repo root to script directory
RELATIVE_PATH=${SCRIPT_DIR#$(pwd)/}

../../scripts/run_local_gpu.sh $RELATIVE_PATH/$1
