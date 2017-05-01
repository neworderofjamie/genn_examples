#include "optical_flow_CODE/definitions.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

// Standard C includes
#include <cassert>
#include <csignal>
#include <cstdlib>

// Common example includes
#include "../common/dvs_128.h"
#include "../common/spike_csv_recorder.h"
#include "../common/timer.h"

// Optical flow includes
#include "parameters.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
typedef void (*allocateFn)(unsigned int);

volatile std::sig_atomic_t g_SignalStatus;

void signalHandler(int status)
{
    g_SignalStatus = status;
}


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

void buildDetectors(SparseProjection &excitatoryProjection, SparseProjection &inhibitoryProjection,
                    allocateFn allocateExcitatory, allocateFn allocateInhibitory)
{
    allocateExcitatory(Parameters::detectorSize * Parameters::detectorSize * Parameters::DetectorMax);
    allocateInhibitory(Parameters::detectorSize * Parameters::detectorSize * Parameters::DetectorMax);

    // Loop through macro cells
    unsigned int sExcitatory = 0;
    unsigned int iExcitatory = 0;
    unsigned int sInhibitory = 0;
    unsigned int iInhibitory = 0;
    for(unsigned int yi = 0; yi < Parameters::macroPixelSize; yi++)
    {
        for(unsigned int xi = 0; xi < Parameters::macroPixelSize; xi++)
        {
            excitatoryProjection.indInG[iExcitatory++] = sExcitatory;
            inhibitoryProjection.indInG[iInhibitory++] = sInhibitory;

            // If we're not in border region
            if(xi >= 1 && xi < (Parameters::macroPixelSize - 1)
                && yi >= 1 && yi < (Parameters::macroPixelSize - 1))
            {
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;

                // Add excitatory synapses
                excitatoryProjection.ind[sExcitatory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorLeft, yj);
                excitatoryProjection.ind[sExcitatory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorRight, yj);
                excitatoryProjection.ind[sExcitatory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorLeft, yj);
                excitatoryProjection.ind[sExcitatory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorRight, yj);
            }


            if(xi < (Parameters::macroPixelSize - 2)
                && yi >= 1 && yi < (Parameters::macroPixelSize - 1))
            {
                // Connect to 'left' detector associated with macropixel one to right
                const unsigned int xj = (xi - 1 + 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;
                inhibitoryProjection.ind[sInhibitory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                        xj + Parameters::DetectorLeft, yj);
            }

            if(xi >= 2
                && yi >= 1 && yi < (Parameters::macroPixelSize - 1))
            {
                // Connect to 'right' detector associated with macropixel one to right
                const unsigned int xj = (xi - 1 - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;
                inhibitoryProjection.ind[sInhibitory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                        xj + Parameters::DetectorRight, yj);
            }

            if(xi >= 1 && xi < (Parameters::macroPixelSize - 1)
                && yi < (Parameters::macroPixelSize - 2))
            {
                // Connect to 'up' detector associated with macropixel one below
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1 + 1;
                inhibitoryProjection.ind[sInhibitory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                        xj + Parameters::DetectorUp, yj);
            }

            if(xi >= 1 && xi < (Parameters::macroPixelSize - 1)
                && yi >= 2)
            {
                // Connect to 'down' detector associated with macropixel one above
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1 - 1;
                inhibitoryProjection.ind[sInhibitory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                        xj + Parameters::DetectorDown, yj);
            }

        }
    }

    // Add ending entry to data structure
    excitatoryProjection.indInG[iExcitatory] = sExcitatory;
    inhibitoryProjection.indInG[iInhibitory] = sInhibitory;

    // Check
    assert(sExcitatory == (Parameters::detectorSize * Parameters::detectorSize * Parameters::DetectorMax));
    assert(iExcitatory == (Parameters::macroPixelSize * Parameters::macroPixelSize));
    assert(sInhibitory == (Parameters::detectorSize * Parameters::detectorSize * Parameters::DetectorMax));
    assert(iInhibitory == (Parameters::macroPixelSize * Parameters::macroPixelSize));
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
    // Create DVS 128 device
    DVS128 dvs(DVS128::Polarity::On);
    
    allocateMem();
    initialize();

    build_centre_to_macro_connection(CDVS_MacroPixel, &allocateDVS_MacroPixel);
    buildDetectors(CMacroPixel_Output_Excitatory, CMacroPixel_Output_Inhibitory,
                   &allocateMacroPixel_Output_Excitatory, &allocateMacroPixel_Output_Inhibitory);
    //print_sparse_matrix(Parameters::inputSize, CDVS_MacroPixel);
    initoptical_flow();

    std::random_device rd;
    std::mt19937 gen(rd());

    std::default_random_engine engine;

    // Catch interrupt (ctrl-c) signals
    std::signal(SIGINT, signalHandler);

    SpikeCSVRecorder dvsPixelSpikeRecorder("dvs_pixel_spikes.csv", glbSpkCntDVS, glbSpkDVS);
    SpikeCSVRecorder macroPixelSpikeRecorder("macro_pixel_spikes.csv", glbSpkCntMacroPixel, glbSpkMacroPixel);
    SpikeCSVRecorder outputSpikeRecorder("output_spikes.csv", glbSpkCntOutput, glbSpkOutput);

    // Start revieving DVS events
    dvs.start();

    double record = 0.0;
    double dvsGet = 0.0;
    double step = 0.0;

    const auto dtDuration = std::chrono::duration<double, std::milli>{DT};

    // Loop through timesteps until there is no more import
    unsigned int i = 0;
    std::chrono::duration<double, std::milli> sleepTime{0};
    std::chrono::duration<double, std::milli> overrunTime{0};
    for(i = 0; g_SignalStatus == 0; i++)
    {
        auto tickStart = std::chrono::high_resolution_clock::now();
    
        {
            TimerAccumulate<std::milli> timer(dvsGet);
            dvs.readEvents(spikeCount_DVS, spike_DVS);
        }

        {
            TimerAccumulate<std::milli> timer(record);
            dvsPixelSpikeRecorder.record(t);
        }

        {
            TimerAccumulate<std::milli> timer(step);

            // Simulate
#ifndef CPU_ONLY
            stepTimeGPU();
#else
            stepTimeCPU();
#endif
        }

        {
            TimerAccumulate<std::milli> timer(record);

            macroPixelSpikeRecorder.record(t);
            outputSpikeRecorder.record(t);
        }

        auto tickEnd = std::chrono::high_resolution_clock::now();
        auto tickDuration = tickEnd - tickStart;
        if(tickDuration < dtDuration) {
            auto tickSleep = dtDuration - tickDuration;
            sleepTime += tickSleep;
            std::this_thread::sleep_for(tickSleep);
        }
        else {
            overrunTime += (tickDuration - dtDuration);
        }
    }

    dvs.stop();

    std::cout << "Ran for " << i << " " << DT << "ms timesteps, overan for " << overrunTime.count() << "ms, slept for " << sleepTime.count() << "ms" << std::endl;
    std::cout << "DVS:" << dvsGet << "ms, Step:" << step << "ms, Record:" << record << std::endl;
    return 0;
}