#include "adexp_curr_CODE/definitions.h"

#include "analogueRecorder.h"

int main()
{
    allocateMem();
    initialize();
    initializeSparse();

    AnalogueRecorder<float> voltageRecorder("voltages.csv", {VNeurons, WNeurons}, 1, ",");

    while(t < 200.0) {
        // Simulate
        stepTime();

        pullNeuronsStateFromDevice();

        voltageRecorder.record(t);
    }

    return EXIT_SUCCESS ;
}
