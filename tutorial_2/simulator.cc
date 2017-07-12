// tenHHRing simulation code
#include "tenHHRing_CODE/definitions.h"

#include <fstream>

int main()
{
    allocateMem();
    initialize();

    // define the connectivity
    int pre, post;
    for (int i= 0; i < 10; i++) {
        pre= i;
        post= (i+1)%10;
        gPop1self[pre*10+post]= -0.2;
    }
    pushPop1selfStateToDevice();

    // define stimuli connectivity
    gStimPop1[0]= -0.2;
    pushStimPop1StateToDevice();

    ofstream os("tenHHRing_output.V.dat");
    for (int i= 0; i < 10000; i++) {
        if(i == 0) {
            glbSpkStim[0] = 0;
            glbSpkCntStim[0] = 1;
            pushStimSpikesToDevice();
        }
        stepTimeGPU();

        pullPop1StateFromDevice();
        if(glbSpkCntPop1[0] > 0)
        {
            printf("GADZOOKS!\n");
        }
        os << t << " ";
        for (int j= 0; j < 10; j++) {
            os << VPop1[j] << " ";
        }
        os << endl;
    }
    os.close();
    return 0;
}