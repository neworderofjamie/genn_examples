#include <algorithm>
#include <numeric>
#include <random>

#include "../common/connectors.h"
#include "../common/spike_csv_recorder.h"
#include "../common/timer.h"

#include "izhikevich_pavlovian_CODE/definitions.h"

#include "parameters.h"

int main()
{
    std::mt19937 gen;

    {
        Timer<> t("Allocation:");
        allocateMem();
    }

    {
        Timer<> t("Initialization:");
        initialize();
    }

    {
        Timer<> t("Building connectivity:");
        buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numInhibitory,
                                    Parameters::probabilityConnection, CII, &allocateII, gen);
        buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numExcitatory,
                                    Parameters::probabilityConnection, CIE, &allocateIE, gen);
        buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numExcitatory,
                                    Parameters::probabilityConnection, CEE, &allocateEE, gen);
        buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numInhibitory,
                                    Parameters::probabilityConnection, CEI, &allocateEI, gen);
    }

    // Final setup
    {
        Timer<> t("Sparse init:");
        initizhikevich_pavlovian();
    }

    std::vector<std::vector<unsigned int>> inputSets;
    {
        Timer<> t("Stimuli generation:");

        // Resize input sets vector
        inputSets.resize(Parameters::numStimuliSets);

        // Build array of neuron indices
        std::vector<unsigned int> neuronIndices(Parameters::numExcitatory + Parameters::numInhibitory);
        std::iota(neuronIndices.begin(), neuronIndices.end(), 0);

        // Loop through input sets
        for(auto &i : inputSets) {
            // Shuffle neuron indices
            std::shuffle(neuronIndices.begin(), neuronIndices.end(), gen);

            // Copy first set size indices into input set
            i.resize(Parameters::stimuliSetSize);
            std::copy_n(neuronIndices.begin(), Parameters::stimuliSetSize, i.begin());
        }
    }

    // Open CSV output files
    SpikeCSVRecorder e_spikes("e_spikes.csv", glbSpkCntE, glbSpkE);
    SpikeCSVRecorder i_spikes("i_spikes.csv", glbSpkCntI, glbSpkI);


    {
        Timer<> t("Simulation:");


         // Create distribution to pick an input to apply thamalic input to
        std::uniform_real_distribution<> inputCurrent(-6.5, 6.5);

        // Create distribution to pick inter stimuli intervals
        std::uniform_int_distribution<> interStimuliInterval((unsigned int)std::round(Parameters::minInterStimuliIntervalMs / Parameters::timestepMs),
                                                            (unsigned int)std::round(Parameters::maxInterStimuliIntervalMs / Parameters::timestepMs));

        std::uniform_int_distribution<> stimuliSet(0, Parameters::numStimuliSets - 1);

         // Draw time until first stimuli
        unsigned int nextStimuliTimestep = interStimuliInterval(gen);
        unsigned int nextStimuliSet = 0;

        // Loop through timesteps
        for(unsigned int t = 0; t < 1000; t++)
        {
            // Generate uniformly distributed numbers to fill host array
            // **TODO** move to GPU
            std::generate_n(IextE, Parameters::numExcitatory,
                [&inputCurrent, &gen](){ return inputCurrent(gen); });
            std::generate_n(IextI, Parameters::numInhibitory,
                [&inputCurrent, &gen](){ return inputCurrent(gen); });

            // If we should be applying input in
            if(t == nextStimuliTimestep) {
                std::cout << "Applying stimuli set " << nextStimuliSet << " at timestep " << t << std::endl;

                // Loop through neurons in input set and add stimuli current
                for(unsigned int n : inputSets[nextStimuliSet]) {
                    if(n < Parameters::numExcitatory) {
                        IextE[n] += (scalar)Parameters::stimuliCurrent;
                    }
                    else {
                        IextI[n] += (scalar)Parameters::stimuliCurrent;
                    }
                }

                // Pick time and set for next stimuli
                nextStimuliTimestep = t + interStimuliInterval(gen);
                nextStimuliSet = stimuliSet(gen);
            }

            // Simulate
#ifndef CPU_ONLY
            // Upload random input currents to GPU
            CHECK_CUDA_ERRORS(cudaMemcpy(d_IextE, IextE, Parameters::numExcitatory * sizeof(scalar), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERRORS(cudaMemcpy(d_IextI, IextI, Parameters::numInhibitory * sizeof(scalar), cudaMemcpyHostToDevice));

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
    }

    return 0;
}
