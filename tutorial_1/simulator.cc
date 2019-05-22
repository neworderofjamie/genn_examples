// tenHHModel simulation code
#include "tenHHModel_CODE/definitions.h"

#include <fstream>

int main()
{
    allocateMem();
    initialize();
    std::ofstream os("tenHHModel_output.V.dat");
    while (t < 1000.0f) {
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
