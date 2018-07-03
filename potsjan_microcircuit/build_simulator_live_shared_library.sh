pushd potjans_microcircuit_CODE
make clean all
popd
g++ simulator_live_shared_library.cc -std=c++11 -pthread `pkg-config --libs --cflags opencv` -I $CUDA_PATH/include -I $GENN_PATH/lib/include -I $BOB_ROBOTICS_PATH -ldl -o simulator_live_shared_library
