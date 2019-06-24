pushd potjans_microcircuit_CODE
make
popd
g++ simulator_live_shared_library.cc -std=c++14 -pthread `pkg-config --libs --cflags opencv` -I $BOB_ROBOTICS_PATH -I $BOB_ROBOTICS_PATH/third_party/plog/include -ldl -o simulator_live_shared_library
