// Standard C++ includes
#include <iostream>
#include <random>

// GeNN user projects includes
#include "spikeRecorder.h"

// Auto-generated model code
#include "spike_source_array_CODE/definitions.h"

int main()
{
    try
    {
        const float rateHz = 10.0f;
        const float durationMs = 500.0f;

        const float scale = 1000.0f / (rateHz * DT);
        std::mt19937 rng;
        std::exponential_distribution<float> dist;
        std::vector<float> spikeTimes;
        std::vector<unsigned int> endIndices;
        endIndices.reserve(101);

        endIndices.push_back(0.0f);
        // Loop through neurons
        for(size_t n = 0; n < 100; n++) {
            // Generate poisson spike train
            float time = 0.0f;
            while(true) {
                time += scale * dist(rng);
                if(time >= durationMs) {
                    break;
                }
                else {
                    spikeTimes.push_back(time);
                }

            }

            // Add end index
            endIndices.push_back((unsigned int)spikeTimes.size());
        }

        allocateMem();
        allocateRecordingBuffers(500);
        initialize();

        std::copy_n(endIndices.cbegin(), 100, startSpikeSSA);
        std::copy_n(endIndices.cbegin() + 1, 100, endSpikeSSA);

        initializeSparse();

        // Allocate spikes and copy
        allocatespikeTimesSSA(spikeTimes.size());
        std::copy(spikeTimes.cbegin(), spikeTimes.cend(), spikeTimesSSA);
        pushspikeTimesSSAToDevice(spikeTimes.size());

        while(t < durationMs) {
            stepTime();
        }
        
        pullRecordingBuffersFromDevice();
        writeTextSpikeRecording("spikes.csv", recordSpkSSA, 100, 500, 1.0, ",", true);
    }
    catch(const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}