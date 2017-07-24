// Standard C++ includes
#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>

// C standard includes
#include <cstdint>

// Common includes
#include "../common/connectors.h"
#include "../common/spike_csv_recorder.h"
#include "../common/timer.h"

// GeNN generated code includes
#include "ardin_webb_mb_CODE/definitions.h"

// Model includes
#include "parameters.h"

//------------------------------------------------------------------------
// Anonymous namespace
//------------------------------------------------------------------------
namespace
{
unsigned int convertMsToTimesteps(double ms)
{
    return (unsigned int)std::round(ms / Parameters::timestepMs);
}
}

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

        buildFixedNumberPreConnector(Parameters::numPN, Parameters::numKC,
                                     Parameters::numPNSynapsesPerKC, CpnToKC, &allocatepnToKC, gen);
    }

    // Final setup
    {
        Timer<> t("Sparse init:");
        initardin_webb_mb();
    }

    std::vector<scalar> stimuliCurrent(Parameters::numPN);
    {
        Timer<> t("Stimuli generation:");

        // Open file for binary IO
        std::ifstream input("test.data", std::ios::binary);
        if(!input.good()) {
            throw std::runtime_error("Cannot open test data");
        }

        // Read grayscale image bytes
        std::array<uint8_t, Parameters::numPN> data;
        input.read(reinterpret_cast<char*>(data.data()), Parameters::numPN);
        if(!input) {
            throw std::runtime_error("Couldn't read test data");
        }

        // Transform raw data into floating point and scale into current
        std::transform(data.begin(), data.end(), stimuliCurrent.begin(),
                       [](uint8_t d)
                       {
                           return (scalar)d * (scalar)(5250.0f / 255.0f);
                       });
    }

    // Open CSV output files
    SpikeCSVRecorder pnSpikes("pn_spikes.csv", glbSpkCntPN, glbSpkPN);
    SpikeCSVRecorder kcSpikes("kc_spikes.csv", glbSpkCntKC, glbSpkKC);
    SpikeCSVRecorder enSpikes("en_spikes.csv", glbSpkCntEN, glbSpkEN);

    std::ofstream synapticTagStream("kc_en_syn.csv");

    {
        Timer<> t("Simulation:");

        // Create normal distribution to generate background input current
        std::normal_distribution<scalar> inputCurrent(0.0f, 0.05f);

        // Convert simulation regime parameters to timesteps
        const unsigned int duration = convertMsToTimesteps(Parameters::durationMs);
        const unsigned int rewardTimestep = convertMsToTimesteps(Parameters::rewardTimeMs);
        const unsigned int presentDuration = convertMsToTimesteps(Parameters::presentDurationMs);

        // Loop through timesteps
        for(unsigned int t = 0; t < duration; t++)
        {
            // Generate background input currents
            // **TODO** move to GPU
            std::generate_n(IextPN, Parameters::numPN,
                [&inputCurrent, &gen](){ return inputCurrent(gen); });
            std::generate_n(IextKC, Parameters::numKC,
                [&inputCurrent, &gen](){ return inputCurrent(gen); });
            std::generate_n(IextEN, Parameters::numEN,
                [&inputCurrent, &gen](){ return inputCurrent(gen); });

            // If we should still be applying image, add stimuli current to projection neuron input
            if(t < presentDuration) {
                std::transform(stimuliCurrent.begin(), stimuliCurrent.end(), IextPN, IextPN,
                               std::plus<scalar>());
            }

            // If we should reward in this timestep, inject dopamine
            if(t == rewardTimestep) {
                std::cout << "\tApplying reward at timestep " << t << std::endl;
                injectDopaminekcToEN = true;
            }

#ifndef CPU_ONLY
            // Upload random input currents to GPU
            CHECK_CUDA_ERRORS(cudaMemcpy(d_IextPN, IextPN, Parameters::numPN * sizeof(scalar), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERRORS(cudaMemcpy(d_IextKC, IextKC, Parameters::numKC * sizeof(scalar), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERRORS(cudaMemcpy(d_IextEN, IextEN, Parameters::numEN * sizeof(scalar), cudaMemcpyHostToDevice));

            // Simulate on GPU
            stepTimeGPU();

            // Download spikes
            pullPNCurrentSpikesFromDevice();
            pullKCCurrentSpikesFromDevice();
            pullENCurrentSpikesFromDevice();

            // Download synaptic weights and tags
            CHECK_CUDA_ERRORS(cudaMemcpy(ckcToEN, d_ckcToEN, Parameters::numKC * Parameters::numEN * sizeof(scalar), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERRORS(cudaMemcpy(gkcToEN, d_gkcToEN, Parameters::numKC * Parameters::numEN * sizeof(scalar), cudaMemcpyDeviceToHost));
#else
            // Simulate on CPU
            stepTimeCPU();
#endif
            // If a dopamine spike has been injected this timestep
            if(t == rewardTimestep) {
                const scalar tMs =  (scalar)t * DT;

                // Decay global dopamine traces
                dkcToEN = dkcToEN * std::exp(-tMs / Parameters::tauD);

                // Add effect of dopamine spike
                dkcToEN += 0.5f;

                // Update last reward time
                tDkcToEN = tMs;

                // Clear dopamine injection flags
                injectDopaminekcToEN = false;
            }

            for(unsigned int i = 0; i < Parameters::numKC * Parameters::numEN; i++) {
                synapticTagStream << t << "," << i << "," << ckcToEN[i] << "," << gkcToEN[i] << std::endl;
            }
            // Record spikes
            pnSpikes.record(t);
            kcSpikes.record(t);
            enSpikes.record(t);
        }
    }

    return 0;
}
