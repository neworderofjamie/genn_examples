#include <algorithm>
#include <iostream>
#include <random>

#include <cassert>
#include <cstdint>

#include "connectors.h"
#include "spike_csv_recorder.h"

#include "bcpnn_simple_CODE/definitions.h"

namespace
{
void printPattern(const std::vector<bool> &p)
{
    for(bool a : p) {
        std::cout << a << ", ";
    }
    std::cout << std::endl;
}

template<typename R>
std::vector<bool> createActivePattern(size_t num, R &rng)
{
    std::vector<bool> p;
    p.reserve(num);

    std::uniform_int_distribution<int> distribution(0, 1);
    std::generate_n(std::back_inserter(p), num,
        [&rng, &distribution](){ return (distribution(rng) != 0); });
    return p;
}

std::vector<bool> createInactivePattern(size_t num)
{
    std::vector<bool> p(num, false);
    return p;
}

std::vector<bool> createUncorrelatedPattern(const std::vector<bool> &p)
{
    std::vector<bool> u;
    u.reserve(p.size());

    std::transform(std::begin(p), std::end(p), std::back_inserter(u),
                    [](bool p){ return !p; });
    return u;
}

std::vector<bool> createCorrelatedPattern(const std::vector<bool> &p)
{
    std::vector<bool> c(p);
    return c;
}

template<typename R, size_t N>
std::vector<bool> createSpikeVector(const std::vector<bool> (&data)[N], float fmax, size_t delayTimesteps, size_t patternTimesteps, R &rng)
{
    // Reserve spike vector
    std::vector<bool> spikes;

    // Add delay to start
    spikes.insert(spikes.end(), delayTimesteps, false);

    std::uniform_real_distribution<float> distribution(0, 1);

    // Loop through patterns
    for(const auto &p : data) {
        for(bool a : p) {
            // Get firing frequency for pattern and corresponding threshold
            const float freq = a ? fmax : 0.0f;
            const float thresh = freq * 0.001f;

            // Generate spike vector
            std::generate_n(std::back_inserter(spikes), patternTimesteps,
                            [thresh, &rng, &distribution](){ return (thresh > distribution(rng)); });
        }
    }

    return spikes;
}
}   // Anonymous namespace

int main()
{
    const size_t numPatterns = 10;
    std::default_random_engine generator;

    // Create patterns
    const auto corr1 = createActivePattern(numPatterns, generator);
    const auto corr2 = createCorrelatedPattern(corr1);

    const auto indep1 = createActivePattern(numPatterns, generator);
    const auto indep2 = createActivePattern(numPatterns, generator);

    const auto anti1 = createActivePattern(numPatterns, generator);
    const auto anti2 = createUncorrelatedPattern(anti1);

    const auto both1 = createInactivePattern(numPatterns);
    const auto both2 = createInactivePattern(numPatterns);

    const auto post1 = createActivePattern(numPatterns, generator);
    const auto post2 = createInactivePattern(numPatterns);

    // Create spike vectors
    const std::vector<bool> pre[] = {corr1, indep1, anti1, both1, post1};
    const std::vector<bool> post[] = {corr2, indep2, anti2, both2, post2};
    const auto preSpikeVector = createSpikeVector(pre, 50.0f, 20, 180, generator);
    const auto postSpikeVector = createSpikeVector(post, 50.0f, 20, 180, generator);
    assert(preSpikeVector.size() == postSpikeVector.size());

    allocateMem();

    initialize();

    buildOneToOneConnector(1, 1, CPreStimToPre, &allocatePreStimToPre);
    buildOneToOneConnector(1, 1, CPostStimToPost, &allocatePostStimToPost);
    buildOneToOneConnector(1, 1, CPreToPost, &allocatePreToPost);

    // Setup reverse connection indices for STDP
    initbcpnn_simple();

    // Load spike vectors
    std::cout << "Sim timesteps:" << preSpikeVector.size() << std::endl;

    SpikeCSVRecorder preSpikes("pre_spikes.csv", glbSpkCntPre, glbSpkPre);
    SpikeCSVRecorder postSpikes("post_spikes.csv", glbSpkCntPost, glbSpkPost);

    FILE *preTrace = fopen("pre_trace.csv", "w");
    FILE *postTrace = fopen("post_trace.csv", "w");

    // Loop through timesteps
    bool recordPreTrace = false;
    bool recordPostTrace = false;
    for(unsigned int i = 0; i < preSpikeVector.size(); i++)
    {
        // Apply input specified by spike vector
        glbSpkCntPreStim[0] = 0;
        glbSpkCntPostStim[0] = 0;
        if(preSpikeVector[i]) {
            glbSpkPreStim[glbSpkCntPreStim[0]++] = 0;
        }
        if(postSpikeVector[i]) {
            glbSpkPostStim[glbSpkCntPostStim[0]++] = 0;
        }

        // Simulate
#ifndef CPU_ONLY
        pushPreStimCurrentSpikesToDevice();
        pushPostStimCurrentSpikesToDevice();

        stepTimeGPU();

        pullPreToPostStateFromDevice();
        pullPreCurrentSpikesFromDevice();
        pullPostCurrentSpikesFromDevice();
#else
        stepTimeCPU();
#endif
        // Record spikes
        preSpikes.record(t);
        postSpikes.record(t);

        if (recordPreTrace) {
            fprintf(preTrace, "%f, %f, %f, %f\n", t, ZiStarPreToPost[0], PiStarPreToPost[0], gPreToPost[0]);
        }

        if (recordPostTrace) {
            fprintf(postTrace, "%f, %f, %f, %f\n", t, ZjStarPreToPost[0], PjStarPreToPost[0], gPreToPost[0]);
        }

        // Record pre and post traces next timestep if there was a spike this timestep
        recordPreTrace = (glbSpkCntPre[0] > 0);
        recordPostTrace = (glbSpkCntPost[0] > 0);
    }

    fclose(preTrace);
    fclose(postTrace);


    return 0;
}