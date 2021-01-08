// Standard C++ includes
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

// MPI includes
#include <mpi.h>

// NCCL includes
#include <nccl.h>

// GeNN userproject includes
#include "analogueRecorder.h"
#include "spikeRecorder.h"

// Auto-generated model code
#include "s_mnist_CODE/definitions.h"

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
//----------------------------------------------------------------------------
// Anonynous namespace
//----------------------------------------------------------------------------
namespace
{
uint32_t readBigEndian(std::ifstream &data)
{
    union
    {
        char b[4];
        uint32_t w;
    } swizzle;

    // Read data into swizzle union
    data.read(&swizzle.b[0], 4);

    // Swap endianess
    std::swap(swizzle.b[0], swizzle.b[3]);
    std::swap(swizzle.b[1], swizzle.b[2]);
    return swizzle.w;
}

unsigned int loadImageData(const std::string &imageDatafilename, uint8_t *&egp,
                           void (*allocateEGPFn)(unsigned int), void (*pushEGPFn)(unsigned int))
{
    // Open binary file
    std::ifstream imageData(imageDatafilename, std::ifstream::binary);
    assert(imageData.good());

    // Read header words
    const uint32_t magic = readBigEndian(imageData);
    const uint32_t numImages = readBigEndian(imageData);
    const uint32_t numRows = readBigEndian(imageData);
    const uint32_t numCols = readBigEndian(imageData);

    // Validate header words
    assert(magic == 0x803);
    assert(numRows == Parameters::inputHeight);
    assert(numCols == Parameters::inputWidth);

    // Allocate EGP for data
    allocateEGPFn(numRows * numCols * numImages);

    // Read data into EGP
    imageData.read(reinterpret_cast<char *>(egp), numRows * numCols * numImages);

    // Push EGP
    pushEGPFn(numRows * numCols * numImages);

    return numImages;
}

void loadLabelData(const std::string &labelDataFilename, unsigned int desiredNumLabels, uint8_t *&egp,
                   void (*allocateEGPFn)(unsigned int), void (*pushEGPFn)(unsigned int))
{
    // Open binary file
    std::ifstream labelData(labelDataFilename, std::ifstream::binary);
    assert(labelData.good());

    // Read header words
    const uint32_t magic = readBigEndian(labelData);
    const uint32_t numLabels = readBigEndian(labelData);

    // Validate header words
    assert(magic == 0x801);
    assert(numLabels == desiredNumLabels);

    // Allocate EGP for data
    allocateEGPFn(numLabels);

    // Read data into EGP
    labelData.read(reinterpret_cast<char *>(egp), numLabels);

    // Push EGP
    pushEGPFn(numLabels);
}
}   // Anonymous namespace

int main()
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
        assert((Parameters::batchSize % numRanks) == 0);
        const unsigned int rankBatchSize = Parameters::batchSize / numRanks;
        std::cout << "(" << rank << ") Batch size:" << rankBatchSize << std::endl;
        
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

        allocateRecordingBuffers(rankBatchSize * Parameters::trialTimesteps);

        // Create NCCL communicator
        CHECK_NCCL_ERRORS(ncclCommInitRank(&ncclCommunicator, numRanks, ncclID, rank));

        initialize();
        
        // Load training data and labels
        const unsigned int totalNumTrainingImages = loadImageData("mnist/train-images.idx3-ubyte", datasetInput, &allocatedatasetInput, &pushdatasetInputToDevice);
        loadLabelData("mnist/train-labels.idx1-ubyte", totalNumTrainingImages, labelsOutput, &allocatelabelsOutput, &pushlabelsOutputToDevice);

        // Calculate number of batches this equates to
        const unsigned int numBatches = ((totalNumTrainingImages + Parameters::batchSize - 1) / Parameters::batchSize);
        
        // Calculate number of training images that should be processed on this rank
        const unsigned int maxNumTrainingImages = (totalNumTrainingImages + numRanks - 1) / numRanks;
        const unsigned int numTrainingImages = (rank == (numRanks - 1)) ? ((totalNumTrainingImages - 1) % maxNumTrainingImages) + 1 : maxNumTrainingImages;

        std::cout << "(" << rank << ") Num training images:" << numTrainingImages << std::endl;
        
        // Allocate indices buffer and initialize host indices
        allocateindicesInput(numTrainingImages);
        allocateindicesOutput(numTrainingImages);
        std::iota(&indicesInput[0], &indicesInput[numTrainingImages], maxNumTrainingImages * rank);


        // Use CUDA to calculate initial transpose of feedforward recurrent->output weights
        BatchLearning::transposeCUDA(d_gRecurrentALIFOutput, d_gOutputRecurrentALIF, 
                                     Parameters::numRecurrentNeurons, Parameters::numOutputNeurons);
        initializeSparse();

        std::ofstream performance;
        if(rank == 0) {
            performance.open("performance.csv");
            performance << "Epoch, Batch, Num trials, Number correct" << std::endl;
        }
        AnalogueRecorder<float> outputRecorder("output" + std::to_string(rank) + ".csv", {PiOutput, EOutput}, Parameters::numOutputNeurons, ",");

        float learningRate = 0.005f;

        // Loop through epochs
        for(unsigned int epoch = 0; epoch < 1; epoch++) {
            std::cout << "(" << rank << ") Epoch " << epoch << std::endl;

            // Shuffle indices, duplicate to output and upload
            // **TODO** some sort of shared pointer business
            std::random_shuffle(&indicesInput[0], &indicesInput[numTrainingImages]);
            std::copy_n(indicesInput, numTrainingImages, indicesOutput);
            pushindicesInputToDevice(numTrainingImages);
            pushindicesOutputToDevice(numTrainingImages);

            // Loop through batches in epoch
            unsigned int i = 0;
            for(unsigned int batch = 0; batch < numBatches; batch++) {
                std::cout << "\t(" << rank << ") Batch " << batch << "/" << numBatches << std::endl;

                const unsigned int numTrialsInBatch = (batch == (numBatches - 1)) ? ((numTrainingImages - 1) % rankBatchSize) + 1 : rankBatchSize;
                
                std::cout << "\t(" << rank << ") Batch " << batch << "/" << numBatches << "(" << numTrialsInBatch << ")" << std::endl;
                
                // Loop through trials
                unsigned int numCorrect = 0;
                for(unsigned int trial = 0; trial < numTrialsInBatch; trial++) {
                    // Loop through timesteps
                    std::array<scalar, 10> output{0};
                    for(unsigned int timestep = 0; timestep < Parameters::trialTimesteps; timestep++) {
                        stepTime();

                        // If we're in the cue region
                        if(timestep > (Parameters::inputWidth * Parameters::inputHeight * Parameters::inputRepeats)) {
                            // Download network output
                            pullPiOutputFromDevice();
                            pullEOutputFromDevice();

                            // Record outputs
                            outputRecorder.record(t);

                            // Add output to total
                            std::transform(output.begin(), output.end(), PiOutput, output.begin(),
                                           [](scalar a, scalar b) { return a + b; });
                        }
                    }

                    // If maximum output matches label, increment counter
                    const auto classification = std::distance(output.cbegin(), std::max_element(output.cbegin(), output.cend()));
                    if(classification == labelsOutput[indicesOutput[i]]) {
                        numCorrect++;
                    }

                    // Advance to next stimuli
                    i++;
                }

                pullRecordingBuffersFromDevice();
                const std::string filenameSuffix = std::to_string(epoch) + "_" + std::to_string(batch) + "_" + std::to_string(rank);
                writeTextSpikeRecording("input_spikes_" + filenameSuffix + ".csv", recordSpkInput,
                                        Parameters::numInputNeurons, Parameters::batchSize * Parameters::trialTimesteps, Parameters::timestepMs,
                                        ",", true);
                writeTextSpikeRecording("recurrent_alif_spikes_" + filenameSuffix + ".csv", recordSpkRecurrentALIF,
                                        Parameters::numRecurrentNeurons, Parameters::batchSize * Parameters::trialTimesteps, Parameters::timestepMs,
                                        ",", true);
                // Update weights
                #define ADAM_OPTIMIZER_CUDA(POP_NAME, NUM_SRC_NEURONS, NUM_TRG_NEURONS) \
                    CHECK_NCCL_ERRORS(ncclAllReduce(d_DeltaG##POP_NAME, d_DeltaG##POP_NAME, NUM_SRC_NEURONS * NUM_TRG_NEURONS, ncclFloat, ncclSum, ncclCommunicator, 0)); \
                    BatchLearning::adamOptimizerCUDA(d_DeltaG##POP_NAME, d_M##POP_NAME, d_V##POP_NAME, d_g##POP_NAME, NUM_SRC_NEURONS, NUM_TRG_NEURONS, epoch, learningRate)
             
                ADAM_OPTIMIZER_CUDA(InputRecurrentALIF, Parameters::numInputNeurons, Parameters::numRecurrentNeurons);
                ADAM_OPTIMIZER_CUDA(ALIFALIFRecurrent, Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons);

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
            
                // Calculate number of trials in this batch
                const unsigned int totalNumTrialsInBatch = (batch == (numBatches - 1)) ? ((totalNumTrainingImages - 1) % Parameters::batchSize) + 1 : Parameters::batchSize;

                // Display performance in this epoch
                std::cout << "\t\t(" << rank << ") " << numCorrect << "/" << totalNumTrialsInBatch << "  correct" << std::endl;
            
                // Write performance to file
                if(rank == 0) {
                    performance << epoch << ", " << batch << ", " << totalNumTrialsInBatch << ", " << numCorrect << std::endl;
                }
            }
        }
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
