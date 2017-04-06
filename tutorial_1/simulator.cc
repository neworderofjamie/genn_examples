// tenHHModel simulation code
#include "tenHHModel_CODE/definitions.h"

#include <fstream>

int main()
{
    allocateMem();
    initialize();
    ofstream os("tenHHModel_output.V.dat");
    for (int i= 0; i < 10000; i++) {
        stepTimeGPU();

        pullPop1StateFromDevice();
        os << t << " ";
        for (int j= 0; j < 10; j++) {
            os << VPop1[j] << " ";
        }
        os << endl;
    }
    os.close();
    return 0;
}