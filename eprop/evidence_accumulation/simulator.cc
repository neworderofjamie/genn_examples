// Standard C++ includes
#include <fstream>
#include <iostream>
#include <random>

// MPI includes
#include <mpi.h>

// NCCL includes
#include <nccl.h>

// GeNN userproject includes
#include "analogueRecorder.h"
#include "spikeRecorder.h"

// Auto-generated model code
#include "evidence_accumulation_CODE/definitions.h"

// Batch-learning includes
#include "batch_learning.h"

// Model parameters
#include "parameters.h"

#define CHECK_MPI_ERRORS(call) {\
    int error = call;\
    if (error != MPI_SUCCESS) {\
        throw std::runtime_error(__FILE__": " + std::to_string(__LINE__) + ": MPI error " + std::to_s\
tring(error));\
    }\
}

#define CHECK_NCCL_ERRORS(call) {\
    ncclResult_t error = call;\
    if (error != ncclSuccess) {\
        throw std::runtime_error(__FILE__": " + std::to_string(__LINE__) + ": NCCL error " + std::to_s\
tring(error) + ": " + ncclGetErrorString(error));\
    }\
}


namespace
{
void setInputRates(float left = Parameters::inactiveRateHz, float right = Parameters::inactiveRateHz, 
                   float decision = Parameters::inactiveRateHz, float background = Parameters::backgroundRateHz)
{
    // Fill rates
    std::fill_n(&rateInput[0 * Parameters::numInputPopulationNeurons], Parameters::numInputPopulationNeurons, left);
    std::fill_n(&rateInput[1 * Parameters::numInputPopulationNeurons], Parameters::numInputPopulationNeurons, right);
    std::fill_n(&rateInput[2 * Parameters::numInputPopulationNeurons], Parameters::numInputPopulationNeurons, decision);
    std::fill_n(&rateInput[3 * Parameters::numInputPopulationNeurons], Parameters::numInputPopulationNeurons, background);

    // Upload to GPU
    pushrateInputToDevice();
}
}

int main(int argc, char *argv[])
{
    try
    {
        // Initialize MPI
        int rank = -1;
        int numRanks = -1;
        CHECK_MPI_ERRORS(MPI_Init(&argc, &argv));
        CHECK_MPI_ERRORS(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        CHECK_MPI_ERRORS(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

	// Due to laziness, check number of trials can be divided by number of ranks
	assert((64 % numRanks) == 0);
	
	// Allocate a unique NCCL ID on first rank
        ncclUniqueId ncclID;
        ncclComm_t ncclCommunicator;
        if(rank == 0) {
            CHECK_NCCL_ERRORS(ncclGetUniqueId(&ncclID));
        }

	// Broadcast NCCL ID to all nodes
	// **NOTE** this always sends value from first rank
	CHECK_MPI_ERRORS(MPI_Bcast((void*)&ncclID, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));

        allocateMem();

	// Create NCCL communicator
	CHECK_NCCL_ERRORS(ncclCommInitRank(&ncclCommunicator, numRanks, ncclID, rank));

	initialize();
        
        // Use CUDA to calculate initial transpose of feedforward recurrent->output weights
        BatchLearning::transposeCUDA(d_gRecurrentLIFOutput, d_gOutputRecurrentLIF, 
                                     Parameters::numRecurrentNeurons, Parameters::numOutputNeurons);
        BatchLearning::transposeCUDA(d_gRecurrentALIFOutput, d_gOutputRecurrentALIF, 
                                     Parameters::numRecurrentNeurons, Parameters::numOutputNeurons);
        initializeSparse();

        std::ofstream performance("performance_" + std::to_string(rank) + ".csv");
        performance << "Epoch, Number of cues, Number correct" << std::endl;
        
        std::mt19937 rng;
	{
	    uint32_t seedData[std::mt19937::state_size];
	    std::random_device seedSource;
	    for(int i = 0; i < std::mt19937::state_size; i++) {
	      seedData[i] = seedSource();
	    }
	    std::seed_seq seeds(std::begin(seedData), std::end(seedData));
	    rng.seed(seeds);
	}
        std::uniform_int_distribution<unsigned int> delayTimestepsDistribution(Parameters::minDelayTimesteps, Parameters::maxDelayTimesteps);

        const std::mt19937::result_type midRNG = std::mt19937::min() + ((std::mt19937::max() - std::mt19937::min()) / 2);

        float learningRate = 0.005f;

        // Start with a single cue
        unsigned int numCues = 1;
        for(unsigned int epoch = 0;; epoch++) {
            // Loop through trials
            unsigned int numCorrect = 0;
            for(unsigned int trial = 0; trial < (64 / numRanks); trial++) {
                // Calculate number of timesteps per cue in this trial
                const unsigned int cueTimesteps = (Parameters::cuePresentTimesteps + Parameters::cueDelayTimesteps) * numCues;

                // Pick delay time
                const unsigned int delayTimesteps = delayTimestepsDistribution(rng);

                // Loop through trial timesteps
                unsigned int numLeftCues = 0;
                unsigned int numRightCues = 0;
                float leftOutput = 0.0f;
                float rightOutput = 0.0f;
                const unsigned int trialTimesteps = cueTimesteps + delayTimesteps + Parameters::decisionTimesteps;
                for(unsigned int timestep = 0; timestep < trialTimesteps; timestep++) {
                    // Cue
                    if(timestep < cueTimesteps) {
                        // Figure out what timestep within the cue we're in
                        const unsigned int cueTimestep = timestep % (Parameters::cuePresentTimesteps + Parameters::cueDelayTimesteps);

                        // If this is the first timestep of the cue
                        if(cueTimestep == 0) {
                            // Activate either left or right neuron
                            if(rng() < midRNG) {
                                numLeftCues++;
                                setInputRates(Parameters::activeRateHz, Parameters::inactiveRateHz);
                            }
                            else {
                                numRightCues++;
                                setInputRates(Parameters::inactiveRateHz, Parameters::activeRateHz);
                            }
                        }
                        // Otherwise, if this is the last timestep of the cue
                        else if(cueTimestep == Parameters::cuePresentTimesteps) {
                            setInputRates();
                        }
                    }
                    // Delay
                    else if(timestep == cueTimesteps) {
                        setInputRates();
                    }
                    // Decision
                    else if(timestep == (cueTimesteps + delayTimesteps)){
                        // Activate correct output neuron depending on which cue was presented more often
                        if(numLeftCues > numRightCues) {
                            PiStarOutput[0] = 1.0f;  PiStarOutput[1] = 0.0f;
                        }
                        else {
                            PiStarOutput[0] = 0.0f;  PiStarOutput[1] = 1.0f;
                        }
                        pushPiStarOutputToDevice();
                        setInputRates(Parameters::inactiveRateHz, Parameters::inactiveRateHz, Parameters::activeRateHz);
                    }
                    stepTime();
    
                    // If we're in decision state
                    if(timestep >= (cueTimesteps + delayTimesteps)){ 
                        // Download output
                        pullPiOutputFromDevice();

			// Accummulate left and right output
                        leftOutput += PiOutput[0];
                        rightOutput += PiOutput[1];
                    }
                }

                // Turn off both outputs
                // **HACK** negative value turns off accumulation of gradients other than during decision
                PiStarOutput[0] = -10.0f;  PiStarOutput[1] = -10.0f;
                pushPiStarOutputToDevice();
                
                // If output was correct this trial, 
                if((leftOutput > rightOutput) == (numLeftCues > numRightCues)) {
                    numCorrect++;
                }
            }
                        
            // Update weights
            #define ADAM_OPTIMIZER_CUDA(POP_NAME, NUM_SRC_NEURONS, NUM_TRG_NEURONS) \
	      CHECK_NCCL_ERRORS(ncclAllReduce(d_DeltaG##POP_NAME, d_DeltaG##POP_NAME, NUM_SRC_NEURONS * NUM_TRG_NEURONS, ncclFloat, ncclSum, ncclCommunicator, 0)); \
	        BatchLearning::adamOptimizerCUDA(d_DeltaG##POP_NAME, d_M##POP_NAME, d_V##POP_NAME, d_g##POP_NAME, NUM_SRC_NEURONS, NUM_TRG_NEURONS, epoch, learningRate)
             
            ADAM_OPTIMIZER_CUDA(InputRecurrentLIF, Parameters::numInputNeurons, Parameters::numRecurrentNeurons);
            ADAM_OPTIMIZER_CUDA(InputRecurrentALIF, Parameters::numInputNeurons, Parameters::numRecurrentNeurons);
            ADAM_OPTIMIZER_CUDA(LIFLIFRecurrent, Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons);
            ADAM_OPTIMIZER_CUDA(ALIFLIFRecurrent, Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons);
            ADAM_OPTIMIZER_CUDA(LIFALIFRecurrent, Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons);
            ADAM_OPTIMIZER_CUDA(ALIFALIFRecurrent, Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons);

	    CHECK_NCCL_ERRORS(ncclAllReduce(d_DeltaGRecurrentLIFOutput, d_DeltaGRecurrentLIFOutput, Parameters::numRecurrentNeurons * Parameters::numOutputNeurons, ncclFloat, ncclSum, ncclCommunicator, 0));
            BatchLearning::adamOptimizerTransposeCUDA(d_DeltaGRecurrentLIFOutput, d_MRecurrentLIFOutput, d_VRecurrentLIFOutput, d_gRecurrentLIFOutput, d_gOutputRecurrentLIF, 
                                                      Parameters::numRecurrentNeurons, Parameters::numOutputNeurons, 
                                                      epoch, learningRate);

			      CHECK_NCCL_ERRORS(ncclAllReduce(d_DeltaGRecurrentALIFOutput, d_DeltaGRecurrentALIFOutput, Parameters::numRecurrentNeurons * Parameters::numOutputNeurons, ncclFloat, ncclSum, ncclCommunicator, 0));
	    BatchLearning::adamOptimizerTransposeCUDA(d_DeltaGRecurrentALIFOutput, d_MRecurrentALIFOutput, d_VRecurrentALIFOutput, d_gRecurrentALIFOutput, d_gOutputRecurrentALIF, 
                                                      Parameters::numRecurrentNeurons, Parameters::numOutputNeurons, 
                                                      epoch, learningRate);
                                                          
            // Update biases
			      CHECK_NCCL_ERRORS(ncclAllReduce(d_DeltaBOutput, d_DeltaBOutput, Parameters::numOutputNeurons, ncclFloat, ncclSum, ncclCommunicator, 0));
            BatchLearning::adamOptimizerCUDA(d_DeltaBOutput, d_MOutput, d_VOutput, d_BOutput,
                                             Parameters::numOutputNeurons, 1,
                                             epoch, learningRate);

	    // Use MPI to sum number of correct trials across ranks
	    CHECK_MPI_ERRORS(MPI_Allreduce(&numCorrect, &numCorrect, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD));
	    
            // Display performance in this epoch	    
            std::cout << "(" << rank << ") Epoch " << epoch << " (" << numCues << " cues): " << numCorrect << "/64 correct" << std::endl;
// Write performance to file
            performance << epoch << ", " << numCues << ", " << numCorrect << std::endl;
            
            // If enough trials were correct
            if(numCorrect > 58) {
                // Advance to next stage of curriculum
                // **NOTE** only odd numbers of cues have a clear winner
                numCues += 2;
                
                // Stop if curriculum is complete
                if(numCues > 7) {
                    break;
                }
            }
        }

	// Free memory allocated by GeNN
	freeMem();

	// Destroy NCCL communicator and finalize MPI
	CHECK_NCCL_ERRORS(ncclCommDestroy(ncclCommunicator));
        CHECK_MPI_ERRORS(MPI_Finalize());
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
