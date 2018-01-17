#!/bin/bash
# **NOTE** CUDA module exports CUDA_HOME, but GeNN expects CUDA_PATH
export CUDA_PATH=$CUDA_HOME
export MPI_PATH=$MPI_HOME

# Build simulator
make clean all MPI_ENABLE=1

# Run simulator
./simulator_$OMPI_COMM_WORLD_RANK
