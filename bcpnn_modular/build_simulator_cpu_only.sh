pushd bcpnn_modular_CODE
make clean all
popd
g++ simulator.cc -std=c++11 -D CPU_ONLY=1 -I $GENN_PATH/lib/include -I $GENN_ROBOTICS_PATH/common -ldl -o simulator