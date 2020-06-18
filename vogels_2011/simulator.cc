// Standard C++ includes
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>
#include <iostream>

// GeNN userproject includes
#include "timer.h"
#include "spikeRecorder.h"

// Auto-generated model code
#include "vogels_2011_CODE/definitions.h"


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
        buildFixedProbabilityConnector(2000, 2000, 0.02,
                                       rowLengthEE, indEE, maxRowLengthEE, gen);
        buildFixedProbabilityConnector(2000, 500, 0.02,
                                       rowLengthEI, indEI, maxRowLengthEI, gen);
        buildFixedProbabilityConnector(500, 500, 0.02,
                                       rowLengthII, indII, maxRowLengthII, gen);
        buildFixedProbabilityConnector(500, 2000, 0.02,
                                       rowLengthIE, indIE, maxRowLengthIE, gen);

        // Initialize membrane voltage
        std::uniform_real_distribution<float> vDist(-60.0, -50.0);
        std::generate_n(VE, 2000, [&vDist, &gen]() { return vDist(gen); });
        std::generate_n(VI, 500, [&vDist, &gen]() { return vDist(gen); });

        initializeSparse();

        // Open CSV output files
        SpikeRecorder<SpikeWriterTextCached> spikes(&getECurrentSpikes, &getECurrentSpikeCount, "spikes.csv", ",", true);

        FILE *weights = fopen("weights.csv", "w");
        fprintf(weights, "Time(ms), Weight (nA)\n");

        {
            Timer b("Simulation:");
            // Loop through timesteps
            while(t < 10000.0f) {
                // Simulate
                stepTime();

                pullECurrentSpikesFromDevice();
                pullgIEFromDevice();

                spikes.record(t);

                float totalWeight = 0.0f;
                unsigned int numSynapses = 0;
                for(unsigned int i = 0; i < 500; i++) {
                    for(unsigned int s = 0; s < rowLengthIE[i]; s++) {
                        totalWeight += gIE[(i * maxRowLengthIE) + s];
                        numSynapses++;
                    }
                }

                // Calculate mean IE weights
                fprintf(weights, "%f, %f\n", 1.0 * t, totalWeight / (double)numSynapses);
            }
        }

        // Close files
        fclose(weights);

    }
    catch(const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
