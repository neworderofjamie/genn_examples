pushd bcpnn_modular_CODE
make clean all
popd
g++ simulator.cc -std=c++11 -I $GENN_PATH/lib/include -I $GENN_ROBOTICS_PATH/common -I $CUDA_PATH/include -L $CUDA_PATH/lib64 -ldl -lcudart -o simulator