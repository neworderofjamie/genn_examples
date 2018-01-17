#!/bin/bash
# **NOTE** CUDA module exports CUDA_HOME, but GeNN expects CUDA_PATH
export CUDA_PATH=$CUDA_HOME
export MPI_PATH=$MPI_HOME

# Generate code
./build_mpi.sh
