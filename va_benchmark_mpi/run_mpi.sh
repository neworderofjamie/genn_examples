#!/bin/bash
# Build simulator
make clean all MPI_ENABLE=1

# Run simulator
./simulator_$OMPI_COMM_WORLD_RANK
