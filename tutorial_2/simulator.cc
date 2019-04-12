// Standard C++ includes
#include <fstream>

// tenHHRing simulation code
#include "tenHHRing_CODE/definitions.h"

int main()
{
    allocateMem();
    initialize();
    initializeSparse();

    std::ofstream os("tenHHRing_output.V.dat");
    while(t < 200.0f) {
        if(iT == 0) {
            glbSpkStim[0] = 0;
            glbSpkCntStim[0] = 1;
            pushStimSpikesToDevice();
        }
        stepTime();
        pullPop1StateFromDevice();

        os << t << " ";
        for (int j= 0; j < 10; j++) {
            os << VPop1[j] << " ";
        }
        os << std::endl;
    }
    os.close();
    return 0;
}
