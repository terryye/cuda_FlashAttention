#!/bin/bash
set -x
mkdir -p ./bin
clang++ -g -o ./bin/main.bin main.cpp
./bin/main.bin