#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

#include "../common/connectors.h"
#include "../common/spike_csv_recorder.h"

#include "izhikevich_pavlovian_CODE/definitions.h"

int main()
{
    auto  allocStart = chrono::steady_clock::now();
    allocateMem();
    auto  allocEnd = chrono::steady_clock::now();
    printf("Allocation %ldms\n", chrono::duration_cast<chrono::milliseconds>(allocEnd - allocStart).count());

    auto  initStart = chrono::steady_clock::now();
    initialize();

    std::mt19937 gen;
    buildFixedProbabilityConnector(200, 200, 0.1, CII, &allocateII, gen);
    buildFixedProbabilityConnector(200, 800, 0.1, CIE, &allocateIE, gen);
    buildFixedProbabilityConnector(800, 800, 0.1, CEE, &allocateEE, gen);
    buildFixedProbabilityConnector(800, 200, 0.1, CEI, &allocateEI, gen);

    // Final setup
    initizhikevich_pavlovian();

    auto initEnd = chrono::steady_clock::now();
    printf("Init %ldms\n", chrono::duration_cast<chrono::milliseconds>(initEnd - initStart).count());

    // Open CSV output files
    SpikeCSVRecorder e_spikes("e_spikes.csv", glbSpkCntE, glbSpkE);
    SpikeCSVRecorder i_spikes("i_spikes.csv", glbSpkCntI, glbSpkI);

    auto simStart = chrono::steady_clock::now();

    // Create distribution to pick an input to apply thamalic input to
    std::uniform_real_distribution<> inputCurrent(-6.5, 6.5);

    // Loop through timesteps
    //unsigned int prevThalamicInputNeuron = 0;
    for(unsigned int t = 0; t < 1000; t++)
    {
        // Generate uniformly distributed numbers to fill host array
        // **TODO** move to GPU
        std::generate_n(IextE, 800,
            [&inputCurrent, &gen](){ return inputCurrent(gen); });
        std::generate_n(IextI, 200,
            [&inputCurrent, &gen](){ return inputCurrent(gen); });

        // Simulate
#ifndef CPU_ONLY
        // Upload random input currents to GPU
        CHECK_CUDA_ERRORS(cudaMemcpy(d_IextE, IextE, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_IextI, IextI, 200 * sizeof(scalar), cudaMemcpyHostToDevice));

        stepTimeGPU();

        // Download spikes from GPU
        pullECurrentSpikesFromDevice();
        pullICurrentSpikesFromDevice();
#else
        stepTimeCPU();
#endif

        // Record spikes
        e_spikes.record(t);
        i_spikes.record(t);
    }
    auto simEnd = chrono::steady_clock::now();
    printf("Simulation %ldms\n", chrono::duration_cast<chrono::milliseconds>(simEnd - simStart).count());

    return 0;
}
