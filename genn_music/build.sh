genn-buildmodel.sh -i $BOB_ROBOTICS_PATH modelExc.cc || exit 1
genn-buildmodel.sh -i $BOB_ROBOTICS_PATH modelInh.cc || exit 1

make -f MakefileExc SIM_CODE=ModelExc_CODE || exit 1
make -f MakefileInh SIM_CODE=ModelInh_CODE || exit 1