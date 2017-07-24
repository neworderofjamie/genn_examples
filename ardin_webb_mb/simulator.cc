// Standard C++ includes
#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>

// C standard includes
#include <cstdint>

// POSIX C includes
extern "C"
{
#include <glob.h>
}

// Common includes
#include "../common/connectors.h"
#include "../common/png_to_float.h"
#include "../common/spike_csv_recorder.h"
#include "../common/timer.h"

// GeNN generated code includes
#include "ardin_webb_mb_CODE/definitions.h"

// Model includes
#include "parameters.h"

//#define RECORD_SYNAPSE_STATE
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

        /*allocatekcToEN(Parameters::numKC);
        for(unsigned int i = 0; i < Parameters::numKC; i++) {
            CkcToEN.indInG[i] = i;
            CkcToEN.ind[i] = 0;
        }
        CkcToEN.indInG[Parameters::numKC] = Parameters::numKC;

        std::fill_n(gkcToEN, Parameters::numKC, Parameters::kcToENWeight);
        std::fill_n(ckcToEN, Parameters::numKC, 0.0f);
        std::fill_n(tCkcToEN, Parameters::numKC, 0.0f);*/
    }

    // Final setup
    {
        Timer<> t("Sparse init:");
        initardin_webb_mb();
    }

    //std::vector<scalar> stimuliCurrent(Parameters::numPN);
    float *stimuliCurrent = nullptr;
    float *d_stimuliCurrent = nullptr;
    unsigned int numStimuli = 0;

    {
        Timer<> t("Stimuli generation:");

        glob_t globBuffer;
        glob("ant2_data/test*.png", GLOB_TILDE, nullptr, &globBuffer);
        std::cout << globBuffer.gl_pathc << " test images found" << std::endl;

        try
        {
            const size_t stimuliSize = Parameters::numPN * (2 + globBuffer.gl_pathc);
            numStimuli = globBuffer.gl_pathc + 1;

#ifdef CPU_ONLY
            stimuliCurrent = new float[stimuliSize];
#else
            CHECK_CUDA_ERRORS(cudaMallocHost(&stimuliCurrent, stimuliSize * sizeof(float)));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_stimuliCurrent, stimuliSize * sizeof(float)));
#endif

            std::fill_n(&stimuliCurrent[0], Parameters::numPN, 0.0f);

            // Load training image
            read_png("ant2_data/train.png", Parameters::inputCurrentScale, false, &stimuliCurrent[Parameters::numPN]);

            // Load testing images
            for(size_t i = 0; i < globBuffer.gl_pathc; i++) {
                read_png(globBuffer.gl_pathv[i], Parameters::inputCurrentScale, false, &stimuliCurrent[(i + 2) * Parameters::numPN]);
            }

#ifndef CPU_ONLY
            // Upload data to GPU
            CHECK_CUDA_ERRORS(cudaMemcpy(d_stimuliCurrent, stimuliCurrent, stimuliSize * sizeof(float), cudaMemcpyHostToDevice));
#endif

            // Set correct image pointer
#ifdef CPU_ONLY
            IextPN = stimuliCurrent;
#else
            IextPN = d_stimuliCurrent;
#endif

        }
        catch(...)
        {
            if(globBuffer.gl_pathc > 0) {
                globfree(&globBuffer);
            }
            throw;
        }
    }

    dkcToEN = 0.0f;

    // Open CSV output files
    SpikeCSVRecorder pnSpikes("pn_spikes.csv", glbSpkCntPN, glbSpkPN);
    SpikeCSVRecorder kcSpikes("kc_spikes.csv", glbSpkCntKC, glbSpkKC);
    SpikeCSVRecorder enSpikes("en_spikes.csv", glbSpkCntEN, glbSpkEN);

#ifdef RECORD_SYNAPSE_STATE
    std::ofstream synapticTagStream("kc_en_syn.csv");
#endif  // RECORD_SYNAPSE_STATE

    {
        Timer<> t("Simulation:");

        // Create normal distribution to generate background input current
        std::normal_distribution<scalar> inputCurrent(0.0f, 0.05f);

        // Convert simulation regime parameters to timesteps
        const unsigned int interStimuliDuration = convertMsToTimesteps(Parameters::interStimuliDurationMs);
        const unsigned int rewardTimestep = convertMsToTimesteps(Parameters::rewardTimeMs);
        const unsigned int presentDuration = convertMsToTimesteps(Parameters::presentDurationMs);

        const unsigned int stimuliDuration = interStimuliDuration + presentDuration;
        const unsigned int duration = stimuliDuration * numStimuli;

        std::cout << "Simulating for " << duration << " timesteps" << std::endl;

        // Loop through timesteps
        for(unsigned int t = 0; t < duration; t++)
        {
            // Generate background input currents
            // **TODO** move to GPU
            /*std::generate_n(IoffsetPN, Parameters::numPN,
                [&inputCurrent, &gen](){ return inputCurrent(gen); });
            std::generate_n(IoffsetKC, Parameters::numKC,
                [&inputCurrent, &gen](){ return inputCurrent(gen); });
            std::generate_n(IoffsetEN, Parameters::numEN,
                [&inputCurrent, &gen](){ return inputCurrent(gen); });*/

            // If we are in stimuli period where we should be presenting an image
            const auto tStimuli = std::div((long)t, (long)stimuliDuration);
            if(tStimuli.rem < presentDuration) {
                if(tStimuli.rem == 0) {
                    std::cout << "\tShowing stimuli:" << tStimuli.quot << std::endl;
                }

                // Update offset to point to correct block of pixel data
                IextOffsetPN = Parameters::numPN * (1 + tStimuli.quot);
            }
            // Otherwise update offset to point to block of zeros
            else {
                IextOffsetPN = 0;
            }

            // If we should reward in this timestep, inject dopamine
            if(t == rewardTimestep) {
                std::cout << "\tApplying reward at timestep " << t << std::endl;
                injectDopaminekcToEN = true;
            }

#ifndef CPU_ONLY
            // Upload random input currents to GPU
            /*CHECK_CUDA_ERRORS(cudaMemcpy(d_IoffsetPN, IoffsetPN, Parameters::numPN * sizeof(scalar), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERRORS(cudaMemcpy(d_IoffsetKC, IoffsetKC, Parameters::numKC * sizeof(scalar), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERRORS(cudaMemcpy(d_IoffsetEN, IoffsetEN, Parameters::numEN * sizeof(scalar), cudaMemcpyHostToDevice));*/

            // Simulate on GPU
            stepTimeGPU();

            // Download spikes
            pullPNCurrentSpikesFromDevice();
            pullKCCurrentSpikesFromDevice();
            pullENCurrentSpikesFromDevice();

#ifdef RECORD_SYNAPSE_STATE
            // Download synaptic weights and tags
            CHECK_CUDA_ERRORS(cudaMemcpy(ckcToEN, d_ckcToEN, Parameters::numKC * Parameters::numEN * sizeof(scalar), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERRORS(cudaMemcpy(gkcToEN, d_gkcToEN, Parameters::numKC * Parameters::numEN * sizeof(scalar), cudaMemcpyDeviceToHost));
#endif  // RECORD_SYNAPSE_STATE
#else
            // Simulate on CPU
            stepTimeCPU();
#endif
            //for(unsigned int i = 0; i < Parameters::numKC; i++) {
            //    kcStateStream << t << "," << i << "," << VKC[i] << "," << UKC[i] << std::endl;
            //}

            // If a dopamine spike has been injected this timestep
            if(t == rewardTimestep) {
                const scalar tMs =  (scalar)t * DT;

                // Decay global dopamine traces
                dkcToEN = dkcToEN * std::exp(-(tMs - tDkcToEN) / Parameters::tauD);

                // Add effect of dopamine spike
                dkcToEN += Parameters::dopamineStrength;

                // Update last reward time
                tDkcToEN = tMs;

                // Clear dopamine injection flags
                injectDopaminekcToEN = false;
            }

#ifdef RECORD_SYNAPSE_STATE
            for(unsigned int i = 0; i < Parameters::numKC * Parameters::numEN; i++) {
                synapticTagStream << t << "," << i << "," << ckcToEN[i] << "," << gkcToEN[i] << std::endl;
            }
#endif  // RECORD_SYNAPSE_STATE

            // Record spikes
            pnSpikes.record(t);
            kcSpikes.record(t);
            enSpikes.record(t);
        }
    }

    return 0;
}
