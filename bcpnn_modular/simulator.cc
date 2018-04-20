// Standard C++ includes
#include <memory>
#include <vector>

// Standard C includes
#include <cassert>

// CUDA includes
#include <cuda_runtime.h>

// GeNN robotics includes
#include "spike_csv_recorder.h"
#include "timer.h"

// Common includes
#include "../common/shared_library_model.h"

// Model parameters
#include "parameters.h"


#define CHECK_CUDA_ERRORS(call) {\
    cudaError_t error = call;\
    if (error != cudaSuccess) {\
        fprintf(stderr, "%s: %i: cuda error %i: %s\n", __FILE__, __LINE__, (int) error, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

int main()
{
    const double dt = 1.0;
    const double trainingStimTime = 100.0;
    const int numTrainingEpochs = 50;
    const int numStimTimesteps = (int)(trainingStimTime / dt);
    const int numTimesteps = numStimTimesteps * Parameters::numMCPerHC * numTrainingEpochs;

    SharedLibraryModelFloat model("bcpnn_modular");
    {
        Timer<> timer("Allocation:");

        model.allocateMem();
    }
    {
        Timer<> timer("Initialization:");
        model.initialize();
    }

    // Final setup
    {
        Timer<> timer("Sparse init:");
        model.initializeSparse();
    }

    const float expMinusLambdaOn = std::exp(-(Parameters::fmax / 1000.0) * dt);
    const float expMinusLambdaOff = 1.0f;

    // Build array of stimuli
    std::vector<float> stimPoissonExpMinusLambda;
    stimPoissonExpMinusLambda.reserve(Parameters::numMCPerHC * Parameters::numMCPerHC);
    for(unsigned int i = 0; i < Parameters::numMCPerHC; i++) {
        stimPoissonExpMinusLambda.insert(stimPoissonExpMinusLambda.end(), i, expMinusLambdaOff);
        stimPoissonExpMinusLambda.push_back(expMinusLambdaOn);
        stimPoissonExpMinusLambda.insert(stimPoissonExpMinusLambda.end(), Parameters::numMCPerHC - i - 1, expMinusLambdaOff);
    }
    assert(stimPoissonExpMinusLambda.size() == (Parameters::numMCPerHC * Parameters::numMCPerHC));

#ifndef CPU_ONLY
    float *d_stimPoissonExpMinusLambda = nullptr;
    CHECK_CUDA_ERRORS(cudaMalloc(&d_stimPoissonExpMinusLambda, Parameters::numMCPerHC * Parameters::numMCPerHC * sizeof(float)));

    CHECK_CUDA_ERRORS(cudaMemcpy(d_stimPoissonExpMinusLambda, stimPoissonExpMinusLambda.data(), Parameters::numMCPerHC * Parameters::numMCPerHC * sizeof(float), cudaMemcpyHostToDevice));
#endif


    //
    std::vector<std::unique_ptr<SpikeCSVRecorderCached>> spikeRecorders;
    spikeRecorders.reserve(Parameters::numHC);

#ifndef CPU_ONLY
    std::vector<SharedLibraryModelFloat::VoidFunction> pullCurrentSpikesFunctions;
    pullCurrentSpikesFunctions.reserve(Parameters::numHC);
#endif

    std::vector<float**> hcuStimPoissonExpMinusLambdaPointers;
    hcuStimPoissonExpMinusLambdaPointers.reserve(Parameters::numHC);

    // Loop through hypercolumns
    for(unsigned int i = 0; i < Parameters::numHC; i++) {
        const std::string name = "E_" + std::to_string(i);

        // Get spike pull function
#ifndef CPU_ONLY
        pullCurrentSpikesFunctions.push_back(
            (SharedLibraryModelFloat::VoidFunction)model.getSymbol("pull" + name + "CurrentSpikesFromDevice"));
#endif

        // Get extra global variable used to point to this HCUs stimuli
        hcuStimPoissonExpMinusLambdaPointers.push_back((float**)model.getSymbol("stimPoissonExpMinusLambda" + name));

        // Get spike count and spikes
        unsigned int **spikeCount = (unsigned int**)model.getSymbol("glbSpkCnt" + name);
        unsigned int **spikes = (unsigned int**)model.getSymbol("glbSpk" + name);

        // Add spike recorder
        spikeRecorders.emplace_back(new SpikeCSVRecorderCached((name + ".csv").c_str(), *spikeCount, *spikes));
    }

    {
        Timer<> timer("Simulation:");

        // Loop through timesteps
        for(int i = 0; i < numTimesteps; i++)
        {
            // If we are at the start of a stimuli
            const auto stim = std::div(i, numStimTimesteps);
            if(stim.rem == 0) {
                const auto stimSequence = std::div(stim.quot, Parameters::numMCPerHC);
                const int activeMC = stimSequence.rem;
                const int trainingEpoch = stimSequence.quot;
                if(activeMC == 0) {
                    std::cout << "Training epoch " << trainingEpoch << std::endl;
                }

                // Loop through HCUs and assign correct input pointer
                for(float **pointer : hcuStimPoissonExpMinusLambdaPointers) {
                    *pointer = &d_stimPoissonExpMinusLambda[activeMC * Parameters::numMCPerHC];
                }

            }
            // Simulate
#ifndef CPU_ONLY
            model.stepTimeGPU();

            // Pull current spikes from all populations
            for(auto p : pullCurrentSpikesFunctions) {
                p();
            }
#else
            model.stepTimeCPU();
#endif
            // Record spikes
            for(auto &s : spikeRecorders) {
                s->record(model.getT());
            }
        }
    }

    // Free memory
#ifndef CPU_ONLY
    CHECK_CUDA_ERRORS(cudaFree(d_stimPoissonExpMinusLambda));
#endif

    return 0;
}
