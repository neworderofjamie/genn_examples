#include "optical_flow_CODE/definitions.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <vector>

// Standard C includes
#include <cassert>
#include <cstdlib>

// Common example includes
#include "../common/analogue_csv_recorder.h"
#include "../common/spike_csv_recorder.h"

// Optical flow includes
#include "parameters.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
typedef void (*allocateFn)(unsigned int);

unsigned int getNeuronIndex(unsigned int resolution, unsigned int x, unsigned int y)
{
    return x + (y * resolution);
}

void print_sparse_matrix(unsigned int pre_resolution, const SparseProjection &projection)
{
    const unsigned int pre_size = pre_resolution * pre_resolution;
    for(unsigned int i = 0; i < pre_size; i++)
    {
        std::cout << i << ":";

        for(unsigned int j = projection.indInG[i]; j < projection.indInG[i + 1]; j++)
        {
            std::cout << projection.ind[j] << ",";
        }

        std::cout << std::endl;
    }
}

void build_centre_to_macro_connection(SparseProjection &projection, allocateFn allocate)
{
    // Allocate centre_size * centre_size connections
    allocate(Parameters::centreSize * Parameters::centreSize);

    // Calculate start and end of border on each row
    const unsigned int near_border = (Parameters::inputSize - Parameters::centreSize) / 2;
    const unsigned int far_border = near_border + Parameters::centreSize;

    // Loop through rows of pixels in centre
    unsigned int s = 0;
    unsigned int i = 0;
    for(unsigned int yi = 0; yi < Parameters::inputSize; yi++)
    {
        for(unsigned int xi = 0; xi < Parameters::inputSize; xi++)
        {
            projection.indInG[i++] = s;

            // If we're in the centre
            if(xi >= near_border && xi < far_border && yi >= near_border && yi < far_border) {
                const unsigned int yj = (yi - near_border) / Parameters::kernelSize;
                const unsigned int xj = (xi - near_border) / Parameters::kernelSize;
                projection.ind[s++] = getNeuronIndex(Parameters::macroPixelSize, xj, yj);
            }
        }
    }

    // Add ending entry to data structure
    projection.indInG[i] = s;

    // Check
    assert(s == (Parameters::centreSize * Parameters::centreSize));
    assert(i == (Parameters::inputSize * Parameters::inputSize));
}

template<typename Generator, typename ShuffleEngine>
void readCalibrateInput(Generator &gen, ShuffleEngine &engine)
{
    // Create array containing coordinates of all pixels within a macrocell
    std::vector<std::pair<unsigned int, unsigned int>> macroCellIndices;
    macroCellIndices.reserve(Parameters::kernelSize * Parameters::kernelSize);
    for(unsigned int x = 0; x < Parameters::kernelSize; x++)
    {
        for(unsigned int y = 0; y < Parameters::kernelSize; y++)
        {
            macroCellIndices.push_back(std::make_pair(x, y));
        }
    }

    spikeCount_DVS = 0;

    const unsigned int nearBorder = (Parameters::inputSize - Parameters::centreSize) / 2;

    // Loop through macropixels
    for(unsigned int yi = 0; yi < Parameters::macroPixelSize; yi++)
    {
        // Create binomial distribution of probability of neuron in kernel firing based on y coordinate
        std::binomial_distribution<> d(Parameters::kernelSize * Parameters::kernelSize,
                                       0.1 * ((double)yi / (double)Parameters::macroPixelSize));
        for(unsigned int xi = 0; xi < Parameters::macroPixelSize; xi++)
        {
            // Shuffle the order of the macrocell indices
            std::shuffle(macroCellIndices.begin(), macroCellIndices.end(), engine);

            // Draw number of active pixels from binomial
            const unsigned int numActiveNeurons = d(gen);
            for(unsigned int i = 0; i < numActiveNeurons; i++)
            {
                unsigned int a = getNeuronIndex(Parameters::inputSize,
                                                (xi * Parameters::kernelSize) + nearBorder + macroCellIndices[i].first,
                                                (yi * Parameters::kernelSize) + nearBorder + macroCellIndices[i].second);
                assert(a < (Parameters::inputSize * Parameters::inputSize));
                spike_DVS[spikeCount_DVS++] = a;

            }
        }
    }
    // 9 spikes per ms
    // 8 spikes per ms
    // 7 spikes per ms
    // 6 spikes per ms
    // 5 spikes
}

unsigned int read_p_input(std::ifstream &stream, std::vector<unsigned int> &indices)
{
    // Read lines into string
    std::string line;
    std::getline(stream, line);

    if(line.empty()) {
        return std::numeric_limits<unsigned int>::max();
    }

    // Create string stream from line
    std::stringstream lineStream(line);

    // Read time from start of line
    std::string nextTimeString;
    std::getline(lineStream, nextTimeString, ';');
    unsigned int nextTime = (unsigned int)std::stoul(nextTimeString);

    // Clear existing times
    indices.clear();

    while(lineStream.good()) {
        // Read input spike index
        std::string inputIndexString;
        std::getline(lineStream, inputIndexString, ',');
        indices.push_back(std::atoi(inputIndexString.c_str()));
    }

    return nextTime;
}
}

int main(int argc, char *argv[])
{
    std::ifstream spikeInput(argv[1]);
    //assert(spikeInput.good());

    allocateMem();
    initialize();

    build_centre_to_macro_connection(CDVS_MacroPixel, &allocateDVS_MacroPixel);

    //print_sparse_matrix(Parameters::inputSize, CDVS_MacroPixel);
    initoptical_flow();

    std::random_device rd;
    std::mt19937 gen(rd());

    std::default_random_engine engine;

    // Read first line of input
    std::vector<unsigned int> inputIndices;
    //unsigned int nextInputTime = read_p_input(spikeInput, inputIndices);

    SpikeCSVRecorder dvsPixelSpikeRecorder("dvs_pixel_spikes.csv", glbSpkCntDVS, glbSpkDVS);
    SpikeCSVRecorder macroPixelSpikeRecorder("macro_pixel_spikes.csv", glbSpkCntMacroPixel, glbSpkMacroPixel);

    // Loop through timesteps until there is no more import
    for(unsigned int i = 0; i < 100/*nextInputTime < std::numeric_limits<unsigned int>::max()*/; i++)
    {
        //spikeCount_DVS = 1;
        //spike_DVS[0] = i;
        readCalibrateInput(gen, engine);
        // If we should supply input this timestep
        //if(nextInputTime == i) {
            // Copy into spike source
            //spikeCount_DVS = inputIndices.size();
            //std::copy(inputIndices.cbegin(), inputIndices.cend(), &spike_DVS[0]);

#ifndef CPU_ONLY
            // Copy to GPU
            pushPCurrentSpikesToDevice();
#endif

            // Read NEXT input
            //nextInputTime = read_p_input(spikeInput, inputIndices);
        //}

        dvsPixelSpikeRecorder.record(t);

        // Simulate
#ifndef CPU_ONLY
        stepTimeGPU();
#else
        stepTimeCPU();
#endif

        macroPixelSpikeRecorder.record(t);
    }

    return 0;
}