#!/bin/bash
# **NOTE** CUDA module exports CUDA_HOME, but GeNN expects CUDA_PATH
export CUDA_PATH=$CUDA_HOME
export MPI_PATH=$MPI_HOME

# Generate code
genn-buildmodel.sh -m -i $GENN_ROBOTICS_PATH/common model.cc
