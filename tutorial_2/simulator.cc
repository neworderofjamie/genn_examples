// Standard C++ includes
#include <fstream>

// tenHHRing simulation code
#include "tenHHRing_CODE/definitions.h"

int main()
{
    allocateMem();
    initialize();
    startSpikeStim[0] = 0;
    endSpikeStim[0] = 1;
    initializeSparse();

    allocatespikeTimesStim(1);
    spikeTimesStim[0] = 0.0f;
    pushspikeTimesStimToDevice(1);

    std::ofstream os("tenHHRing_output.V.dat");
    while(t < 200.0f) {
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
