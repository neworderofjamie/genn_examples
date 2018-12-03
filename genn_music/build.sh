genn-buildmodel.sh modelExc.cc || exit 1
genn-buildmodel.sh modelInh.cc || exit 1

make -f MakefileExc SIM_CODE=ModelExc_CODE || exit 1
make -f MakefileInh SIM_CODE=ModelInh_CODE || exit 1