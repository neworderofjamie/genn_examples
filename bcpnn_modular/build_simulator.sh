pushd bcpnn_modular_CODE
make
popd
g++ simulator.cc -std=c++11 -I $GENN_PATH/lib/include -I $BOB_ROBOTICS_PATH -I $CUDA_PATH/include -L $CUDA_PATH/lib64 -ldl -lcudart -o simulator