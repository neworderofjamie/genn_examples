// Standard C++ includes
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

// GeNN robotics includes
#include "timer.h"
#include "spikeRecorder.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "va_benchmark_CODE/definitions.h"

template <typename Generator, typename IndexType>
void buildFixedProbabilityConnector(unsigned int numPre, unsigned int numPost, float probability,
                                    unsigned int *rowLength, IndexType *ind, unsigned int maxRowLength, 
                                    Generator &gen)
{
    const double probabilityReciprocal = 1.0 / std::log(1.0f - probability);

    // Create RNG to draw probabilities
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Zero row lengths
    std::fill_n(rowLength, numPre, 0);

    // Loop through potential synapses
    const int64_t maxConnections = (int64_t)numPre * (int64_t)numPost;
    for(int64_t s = -1;;) {
        // Skip to next connection
        s += (1 + (int64_t)(std::log(dis(gen)) * probabilityReciprocal));

        // If we haven't skipped past end of matrix
        if(s < maxConnections) {
            // Convert synapse number to pre and post index
            const auto prePost = std::div(s, numPost);

            // Get pointer to start of this presynaptic neuron's connection row
            IndexType *rowIndices = &ind[prePost.quot * maxRowLength];

            // Add synapse
            rowIndices[rowLength[prePost.quot]++] = prePost.rem;
            assert(rowLength[prePost.quot] <= maxRowLength);
        }
        else {
            break;
        }
    }
}

int main()
{
    try
    {
        std::mt19937 gen;
        allocateMem();
        initialize();

        // Create connectivity
        buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numExcitatory, Parameters::probabilityConnection,
                                       rowLengthEE, indEE, maxRowLengthEE, gen);
        buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numInhibitory, Parameters::probabilityConnection,
                                       rowLengthEI, indEI, maxRowLengthEI, gen);
        buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numInhibitory, Parameters::probabilityConnection,
                                       rowLengthII, indII, maxRowLengthII, gen);
        buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numExcitatory, Parameters::probabilityConnection,
                                       rowLengthIE, indIE, maxRowLengthIE, gen);

        // Initialize membrane voltage
        std::uniform_real_distribution<float> vDist(Parameters::resetVoltage, Parameters::thresholdVoltage);
        std::generate_n(VE, Parameters::numExcitatory, [&vDist, &gen]() { return vDist(gen); });
        std::generate_n(VI, Parameters::numInhibitory, [&vDist, &gen]() { return vDist(gen); });

        initializeSparse();

        // Open CSV output files
        SpikeRecorder<SpikeWriterTextCached> spikes(&getECurrentSpikes, &getECurrentSpikeCount, "spikes.csv", ",", true);

        {
            Timer a("Simulation wall clock:");
            while(t < 10000.0) {
                // Simulate
                stepTime();

                pullECurrentSpikesFromDevice();


                spikes.record(t);
            }
        }

        spikes.writeCache();

        std::cout << "Init:" << initTime << std::endl;
        std::cout << "Init sparse:" << initSparseTime << std::endl;
        std::cout << "Neuron update:" << neuronUpdateTime << std::endl;
        std::cout << "Presynaptic update:" << presynapticUpdateTime << std::endl;
    }
    catch(const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
