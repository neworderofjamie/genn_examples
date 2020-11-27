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
        BatchLearning::DeepR recurrentRecurrentdeepR(Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons, maxRowLengthRecurrentRecurrent,
                                                     rowLengthRecurrentRecurrent, d_rowLengthRecurrentRecurrent, d_indRecurrentRecurrent,
                                                     d_DeltaGRecurrentRecurrent, d_MRecurrentRecurrent, d_VRecurrentRecurrent, 
                                                     d_gRecurrentRecurrent, d_eFilteredRecurrentRecurrent);
#endif
        AnalogueRecorder<float> outputRecorder("output.csv", {YOutput, YStarOutput}, Parameters::numOutputNeurons, ",");

        CUDATimer inputRecurrentTimer;
#ifndef USE_DEEP_R
        CUDATimer recurrentRecurrentTimer;
#endif
        CUDATimer recurrentOutputTimer;

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
                if(Parameters::timingEnabled) {
                    inputRecurrentTimer.start();
                }
                BatchLearning::adamOptimizerCUDA(d_DeltaGInputRecurrent, d_MInputRecurrent, d_VInputRecurrent, d_gInputRecurrent, 
                                                 Parameters::numInputNeurons, Parameters::numRecurrentNeurons, 
                                                 trial, learningRate);
                if(Parameters::timingEnabled) {
                    inputRecurrentTimer.stop();
                }
#ifdef USE_DEEP_R
                recurrentRecurrentdeepR.update(trial, learningRate);
#else
                if(Parameters::timingEnabled) {
                    recurrentRecurrentTimer.start();
                }
                BatchLearning::adamOptimizerCUDA(d_DeltaGRecurrentRecurrent, d_MRecurrentRecurrent, d_VRecurrentRecurrent, d_gRecurrentRecurrent, 
                                                 Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons, 
                                                 trial, learningRate);
                if(Parameters::timingEnabled) {
                    recurrentRecurrentTimer.stop();
                }
#endif
                if(Parameters::timingEnabled) {
                    recurrentOutputTimer.start();
                }
                BatchLearning::adamOptimizerTransposeCUDA(d_DeltaGRecurrentOutput, d_MRecurrentOutput, d_VRecurrentOutput, d_gRecurrentOutput, d_gOutputRecurrent, 
                                                          Parameters::numRecurrentNeurons, Parameters::numOutputNeurons, 
                                                          trial, learningRate);
                if(Parameters::timingEnabled) {
                    recurrentOutputTimer.stop();

                    // Wait for last timer to complete
                    recurrentOutputTimer.synchronize();

                    // Update counters
                    inputRecurrentTimer.update();
#ifdef USE_DEEP_R
                    recurrentRecurrentdeepR.updateTimers();
#else
                    recurrentRecurrentTimer.update();
#endif
                    recurrentOutputTimer.update();
                }
            }
        }
        
        if(Parameters::timingEnabled) {
            std::cout << "GeNN:" << std::endl;
            std::cout << "\tInit:" << initTime << std::endl;
            std::cout << "\tInit sparse:" << initSparseTime << std::endl;
            std::cout << "\tNeuron update:" << neuronUpdateTime << std::endl;
            std::cout << "\tPresynaptic update:" << presynapticUpdateTime << std::endl;
            std::cout << "\tSynapse dynamics:" << synapseDynamicsTime << std::endl;

            std::cout << "Batch learning:" << std::endl;
            std::cout << "\tInput->recurrent learning:" << inputRecurrentTimer.getTotalTime() << std::endl;
#ifdef USE_DEEP_R
            std::cout << "\tRecurrent->recurrent Deep-R learning:" << std::endl;
            std::cout << "\t\tTotal:" << recurrentRecurrentdeepR.getHostUpdateTime() << std::endl;
            std::cout << "\t\tFirst pass kernel:" << recurrentRecurrentdeepR.getFirstPassKernelTime() << std::endl;
            std::cout << "\t\tSecond pass kernel:" << recurrentRecurrentdeepR.getSecondPassKernelTime() << std::endl;
#else
            std::cout << "\tRecurrent->recurrent learning:" << recurrentRecurrentTimer.getTotalTime() << std::endl;
#endif
            std::cout << "\tRecurrent->output learning:" << recurrentOutputTimer.getTotalTime() << std::endl;
        }
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
    }
}
