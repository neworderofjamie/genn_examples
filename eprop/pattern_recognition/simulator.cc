#include "pattern_recognition_CODE/definitions.h"

#include <iostream>
#include <random>

// GeNN userproject includes
#include "analogueRecorder.h"
#include "spikeRecorder.h"
#include "timer.h"

// Batch-learning includes
#include "batch_learning.h"
#include "deep_r.h"

// EProp includes
#include "cuda_timer.h"
#include "parameters.h"

int main()
{
    try 
    {
        std::mt19937 rng;

        allocateMem();
        allocateRecordingBuffers(1000);
        initialize();

        // Use CUDA to calculate initial transpose of feedforward recurrent->output weights
        BatchLearning::transposeCUDA(d_gRecurrentOutput, d_gOutputRecurrent, 
                                     Parameters::numRecurrentNeurons, Parameters::numOutputNeurons);

        initializeSparse();

#ifdef USE_DEEP_R
        CUDATimer recurrentRecurrentTimer;
        BatchLearning::DeepR recurrentRecurrentdeepR(Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons, maxRowLengthRecurrentRecurrent,
                                                     rowLengthRecurrentRecurrent, d_rowLengthRecurrentRecurrent, d_indRecurrentRecurrent,
                                                     d_DeltaGRecurrentRecurrent, d_MRecurrentRecurrent, d_VRecurrentRecurrent, 
                                                     d_gRecurrentRecurrent, d_eFilteredRecurrentRecurrent);
#endif
        AnalogueRecorder<float> outputRecorder("output.csv", {YOutput, YStarOutput}, Parameters::numOutputNeurons, ",");
        float learningRate = 0.003f;
        {
            Timer a("Simulation wall clock:");

            // Loop through trials
            for(unsigned int trial = 0; trial <= 1000; trial++) {
                if((trial % 100) == 0) {
                    // if this isn't the first trial, reduce learning rate
                    if(trial != 0) {
                        learningRate *= 0.7f;
                    }

                    std::cout << "Trial " << trial << " (learning rate " << learningRate << ")" << std::endl;
                }
                // Loop through timesteps within trial
                for(unsigned int i = 0; i < 1000; i++) {
                    stepTime();

                    if((trial % 100) == 0) {
                        // Download state
                        pullYOutputFromDevice();
                        pullYStarOutputFromDevice();

                        // Record
                        outputRecorder.record(t);
                    }
                }

                if((trial % 100) == 0) {
                    pullRecordingBuffersFromDevice();
                    writeTextSpikeRecording("input_spikes_" + std::to_string(trial) + ".csv", recordSpkInput,
                                            Parameters::numInputNeurons, 1000, 1.0, ",", true);
                    writeTextSpikeRecording("recurrent_spikes_" + std::to_string(trial) + ".csv", recordSpkRecurrent,
                                            Parameters::numRecurrentNeurons, 1000, 1.0, ",", true);
                }
                
                // Apply learning
                const scalar firstMomentScale = 1.0f / (1.0f - std::pow(Parameters::beta1, trial + 1));
                const scalar secondMomentScale = 1.0f / (1.0f - std::pow(Parameters::beta2, trial + 1));
                alphaInputRecurrentWeightOptimiser = learningRate;
                alphaRecurrentRecurrentWeightOptimiser = learningRate;
                alphaRecurrentOutputWeightOptimiser = learningRate;
                firstMomentScaleInputRecurrentWeightOptimiser = firstMomentScale;
                firstMomentScaleRecurrentRecurrentWeightOptimiser = firstMomentScale;
                firstMomentScaleRecurrentOutputWeightOptimiser = firstMomentScale;
                secondMomentScaleInputRecurrentWeightOptimiser = secondMomentScale;
                secondMomentScaleRecurrentRecurrentWeightOptimiser = secondMomentScale;
                secondMomentScaleRecurrentOutputWeightOptimiser = secondMomentScale;
                updateGradientLearn();
#ifdef USE_DEEP_R
                recurrentRecurrentdeepR.update(trial, learningRate);

                if(Parameters::timingEnabled) {
                    recurrentRecurrentdeepR.updateTimers();
                }
#endif
            }
        }
        
        if(Parameters::timingEnabled) {
            std::cout << "GeNN:" << std::endl;
            std::cout << "\tInit:" << initTime << std::endl;
            std::cout << "\tInit sparse:" << initSparseTime << std::endl;
            std::cout << "\tNeuron update:" << neuronUpdateTime << std::endl;
            std::cout << "\tPresynaptic update:" << presynapticUpdateTime << std::endl;
            std::cout << "\tSynapse dynamics:" << synapseDynamicsTime << std::endl;
            std::cout << "\tGradient learning custom update:" << updateGradientLearnTime << std::endl;
#ifdef USE_DEEP_R
            std::cout << "\tRecurrent->recurrent Deep-R learning:" << std::endl;
            std::cout << "\t\tTotal:" << recurrentRecurrentdeepR.getHostUpdateTime() << std::endl;
            std::cout << "\t\tFirst pass kernel:" << recurrentRecurrentdeepR.getFirstPassKernelTime() << std::endl;
            std::cout << "\t\tSecond pass kernel:" << recurrentRecurrentdeepR.getSecondPassKernelTime() << std::endl;
#endif
        }
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
    }
}
