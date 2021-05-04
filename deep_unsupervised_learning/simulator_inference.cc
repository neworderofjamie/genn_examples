// Standard C++ includes
#include <algorithm>
#include <iostream>
#include <vector>

// GeNN userproject includes
#include "spikeRecorder.h"

// Auto-generated model code
#include "deep_unsupervised_learning_inference_CODE/definitions.h"

// Model parameters
#include "parameters.h"
#include "sim_utils.h"

int main()
{
    using namespace Parameters;

    allocateMem();
    allocateRecordingBuffers(1000);
    initialize();

    // Read kernels from disk
    {
        std::ifstream conv1("kernels/conv1_kernel.bin", std::ios_base::binary);
        std::ifstream conv2("kernels/conv2_kernel.bin", std::ios_base::binary);
        
        conv1.read(reinterpret_cast<char*>(gInput_Conv1), InputConv1::kernelSize * sizeof(scalar));
        conv2.read(reinterpret_cast<char*>(gConv1_Conv2), Conv1Conv2::kernelSize * sizeof(scalar));
    }
    initializeSparse();
    
        // Load grid data
    const unsigned int numTrainingImages = loadImageData("green_grid.bin", datasetInput, Input::numNeurons,
                                                         &allocatedatasetInput, &pushdatasetInputToDevice);                                                    
    
    // Simulate
    for(unsigned int i = 0; i < 1000; i++) {
        stepTime();
    }
    
     // Save spikes
    pullRecordingBuffersFromDevice();
    writeTextSpikeRecording("input_inference_spikes_0.csv", recordSpkInput,
                            Input::numNeurons, 1000, timestepMs,
                            ",", true);

    writeTextSpikeRecording("conv1_inference_spikes_0.csv", recordSpkConv1,
                            Conv1::numNeurons, 1000, timestepMs,
                            ",", true);

    writeTextSpikeRecording("conv2_inference_spikes_0.csv", recordSpkConv2,
                            Conv2::numNeurons, 1000, timestepMs,
                            ",", true);

    writeTextSpikeRecording("kc_inference_spikes_0.csv", recordSpkKC,
                            KC::numNeurons, 1000, timestepMs,
                            ",", true);

    return EXIT_SUCCESS;
}
