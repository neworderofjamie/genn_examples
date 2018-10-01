// tenHHModel simulation code
#include "tenHHModel_CODE/definitions.h"

#include <fstream>

int main()
{
    allocateMem();
    initialize();
    ofstream os("tenHHModel_output.V.dat");
    while (t < 200.0f) {
#ifdef CPU_ONLY
        stepTimeCPU();
#else
        stepTimeGPU();

        pullPop1StateFromDevice();
#endif
        os << t << " ";
        for (int j= 0; j < 10; j++) {
            os << VPop1[j] << " ";
        }
        os << endl;
    }
    os.close();
    return 0;
}