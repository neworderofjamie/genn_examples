pushd potjans_microcircuit_CODE
make clean all
popd
g++ simulator_shared_library.cc -std=c++11 -I $CUDA_PATH/include -I $GENN_PATH/lib/include -I $GENN_ROBOTICS_PATH -ldl -o simulator_shared_library
