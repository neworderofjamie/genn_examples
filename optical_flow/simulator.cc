// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

// Standard C includes
#include <cassert>
#include <csignal>
#include <cstdlib>

// OpenCV includes
#include <opencv2/highgui/highgui.hpp>

// Common example includes
#ifdef LIVE
    #include "../common/dvs_128.h"
#endif
#include "../common/spike_image_renderer.h"
#include "../common/timer.h"

// Optical flow includes
#include "parameters.h"

// Auto-generated simulation code
#include "optical_flow_CODE/definitions.h"

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

void buildCentreToMacroConnection(SparseProjection &projection, allocateFn allocate)
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
            // Mark start of 'synaptic row'
            excitatoryProjection.indInG[iExcitatory++] = sExcitatory;
            inhibitoryProjection.indInG[iInhibitory++] = sInhibitory;

            // If we're not in border region
            if(xi >= 1 && xi < (Parameters::macroPixelSize - 1)
                && yi >= 1 && yi < (Parameters::macroPixelSize - 1))
            {
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;

                // Add excitatory synapses to all detectors
                excitatoryProjection.ind[sExcitatory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorLeft, yj);
                excitatoryProjection.ind[sExcitatory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorRight, yj);
                excitatoryProjection.ind[sExcitatory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorUp, yj);
                excitatoryProjection.ind[sExcitatory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorDown, yj);
            }


            // Create inhibitory connection to 'left' detector associated with macropixel one to right
            if(xi < (Parameters::macroPixelSize - 2)
                && yi >= 1 && yi < (Parameters::macroPixelSize - 1))
            {
                const unsigned int xj = (xi - 1 + 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;
                inhibitoryProjection.ind[sInhibitory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorLeft, yj);
            }

            // Create inhibitory connection to 'right' detector associated with macropixel one to right
            if(xi >= 2
                && yi >= 1 && yi < (Parameters::macroPixelSize - 1))
            {
                const unsigned int xj = (xi - 1 - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;
                inhibitoryProjection.ind[sInhibitory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorRight, yj);
            }

            // Create inhibitory connection to 'up' detector associated with macropixel one below
            if(xi >= 1 && xi < (Parameters::macroPixelSize - 1)
                && yi < (Parameters::macroPixelSize - 2))
            {
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1 + 1;
                inhibitoryProjection.ind[sInhibitory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorUp, yj);
            }

            // Create inhibitory connection to 'down' detector associated with macropixel one above
            if(xi >= 1 && xi < (Parameters::macroPixelSize - 1)
                && yi >= 2)
            {
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

bool readPInput(std::ifstream &stream, std::vector<unsigned int> &indices, unsigned int &nextTime)
{
    // Read lines into string
    std::string line;
    std::getline(stream, line);

    if(line.empty()) {
        return false;
    }

    // Create string stream from line
    std::stringstream lineStream(line);

    // Read time from start of line
    std::string nextTimeString;
    std::getline(lineStream, nextTimeString, ';');
    nextTime = (unsigned int)std::stoul(nextTimeString);

    // Clear existing times
    indices.clear();

    while(lineStream.good()) {
        // Read input spike index
        std::string inputIndexString;
        std::getline(lineStream, inputIndexString, ',');
        indices.push_back(std::atoi(inputIndexString.c_str()));
    }

    return true;
}

void displayThreadHandler(std::mutex &inputMutex, const cv::Mat &inputImage,
                          std::mutex &outputMutex, const float (&output)[Parameters::detectorSize][Parameters::detectorSize][2])
{
    cv::namedWindow("Input", CV_WINDOW_NORMAL);
    cv::resizeWindow("Input", Parameters::inputSize * Parameters::inputScale,
                     Parameters::inputSize * Parameters::inputScale);

    // Create output image
    const unsigned int outputImageSize = Parameters::detectorSize * Parameters::outputScale;
    cv::Mat outputImage(outputImageSize, outputImageSize, CV_8UC3);

#ifdef JETSON_POWER
    std::ifstream powerStream("/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device/in_power0_input");
    std::ifstream gpuPowerStream("/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device/in_power1_input");
    std::ifstream cpuPowerStream("/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device/in_power2_input");
#endif  // JETSON_POWER

    while(g_SignalStatus == 0)
    {
        // Clear background
        outputImage.setTo(cv::Scalar::all(0));

        {
            std::lock_guard<std::mutex> lock(outputMutex);

            // Loop through output coordinates
            for(unsigned int x = 0; x < Parameters::detectorSize; x++)
            {
                for(unsigned int y = 0; y < Parameters::detectorSize; y++)
                {
                    const cv::Point start(x * Parameters::outputScale, y * Parameters::outputScale);
                    const cv::Point end = start + cv::Point(Parameters::outputVectorScale * output[x][y][0],
                                                            Parameters::outputVectorScale * output[x][y][1]);

                    cv::line(outputImage, start, end,
                             CV_RGB(0xFF, 0xFF, 0xFF));
                }
            }
        }


#ifdef JETSON_POWER
        // Read all power measurements
        unsigned int power, cpuPower, gpuPower;
        powerStream >> power;
        cpuPowerStream >> cpuPower;
        gpuPowerStream >> gpuPower;

        // Clear all stream flags (EOF gets set)
        powerStream.clear();
        cpuPowerStream.clear();
        gpuPowerStream.clear();

        char power[255];
        sprintf(power, "Power:%umW, GPU power:%umW", power, gpuPower);
        cv::putText(outputImage, power, cv::Point(0, outputImageSize - 20),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 0xFF));
        sprintf(power, "CPU power:%umW", cpuPower);
        cv::putText(outputImage, power, cv::Point(0, outputImageSize - 5),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 0xFF));
#endif

        cv::imshow("Output", outputImage);

        {
            std::lock_guard<std::mutex> lock(inputMutex);
            cv::imshow("Input", inputImage);
        }


        cv::waitKey(33);
    }
}

void applyOutputSpikes(unsigned int outputSpikeCount, const unsigned int *outputSpikes, float (&output)[Parameters::detectorSize][Parameters::detectorSize][2])
{
    // Loop through output spikes
    for(unsigned int s = 0; s < outputSpikeCount; s++)
    {
        // Convert spike ID to x, y, detector
        const unsigned int spike = outputSpikes[s];
        const auto spikeCoord = std::div((int)spike, (int)Parameters::detectorSize * Parameters::DetectorMax);
        const int spikeY = spikeCoord.quot;
        const auto xCoord = std::div(spikeCoord.rem, (int)Parameters::DetectorMax);
        const int spikeX =  xCoord.quot;

        // Apply spike to correct axis of output pixel based on detector it was emitted by
        switch(xCoord.rem)
        {
            case Parameters::DetectorLeft:
                output[spikeX][spikeY][0] -= 1.0f;
                break;

            case Parameters::DetectorRight:
                output[spikeX][spikeY][0] += 1.0f;
                break;

            case Parameters::DetectorUp:
                output[spikeX][spikeY][1] -= 1.0f;
                break;

            case Parameters::DetectorDown:
                output[spikeX][spikeY][1] += 1.0f;
                break;

        }
    }

    // Decay output
    for(unsigned int x = 0; x < Parameters::detectorSize; x++)
    {
        for(unsigned int y = 0; y < Parameters::detectorSize; y++)
        {
            output[x][y][0] *= Parameters::spikePersistence;
            output[x][y][1] *= Parameters::spikePersistence;
        }
    }
}
}

int main(int argc, char *argv[])
{
    allocateMem();
    initialize();

    buildCentreToMacroConnection(CDVS_MacroPixel, &allocateDVS_MacroPixel);
    buildDetectors(CMacroPixel_Output_Excitatory, CMacroPixel_Output_Inhibitory,
                   &allocateMacroPixel_Output_Excitatory, &allocateMacroPixel_Output_Inhibitory);
    //print_sparse_matrix(Parameters::inputSize, CDVS_MacroPixel);
    initoptical_flow();

#ifdef LIVE
     // Create DVS 128 device
    DVS128 dvs(DVS128::Polarity::On);

    double dvsGet = 0.0;
#else
    assert(argc > 1);

    std::ifstream spikeInput(argv[1]);
    assert(spikeInput.good());

    double read = 0.0;

    // Read first line of input
    std::vector<unsigned int> inputIndices;
    unsigned int nextInputTime;
    if(!readPInput(spikeInput, inputIndices, nextInputTime)) {
        std::cerr << "No spikes to input" << std::endl;
        return 1;
    }
#endif

    double step = 0.0;
    double render = 0.0;


    std::mutex inputMutex;
    cv::Mat inputImage(Parameters::inputSize, Parameters::inputSize, CV_32F);

    std::mutex outputMutex;
    float output[Parameters::detectorSize][Parameters::detectorSize][2] = {0};
    std::thread displayThread(displayThreadHandler,
                              std::ref(inputMutex), std::ref(inputImage),
                              std::ref(outputMutex), std::ref(output));

    // Convert timestep to a duration
    const auto dtDuration = std::chrono::duration<double, std::milli>{DT};

    // Duration counters
    std::chrono::duration<double, std::milli> sleepTime{0};
    std::chrono::duration<double, std::milli> overrunTime{0};
    unsigned int i = 0;

     // Catch interrupt (ctrl-c) signals
    std::signal(SIGINT, signalHandler);

    for(i = 0; g_SignalStatus == 0; i++)
    {
        auto tickStart = std::chrono::high_resolution_clock::now();
#ifdef LIVE
        {
            TimerAccumulate<std::milli> timer(dvsGet);
            dvs.readEvents(spikeCount_DVS, spike_DVS);

#ifndef CPU_ONLY
            // Copy to GPU
            pushDVSCurrentSpikesToDevice();
#endif
        }
#else
        {
            TimerAccumulate<std::milli> timer(read);

            // If we should supply input this timestep
            if(nextInputTime == i) {
                // Copy into spike source
                spikeCount_DVS = inputIndices.size();
                std::copy(inputIndices.cbegin(), inputIndices.cend(), &spike_DVS[0]);

                // Read NEXT input
                if(!readPInput(spikeInput, inputIndices, nextInputTime)) {
                    g_SignalStatus = 1;
                }
            }

#ifndef CPU_ONLY
            // Copy to GPU
            pushDVSCurrentSpikesToDevice();
#endif
        }
#endif
        {
            TimerAccumulate<std::milli> timer(render);
            {
                std::lock_guard<std::mutex> lock(inputMutex);
                renderSpikeImage(spikeCount_DVS, spike_DVS, Parameters::inputSize,
                                 Parameters::spikePersistence, inputImage);
            }
        }

        {
            TimerAccumulate<std::milli> timer(step);

            // Simulate
#ifndef CPU_ONLY
            stepTimeGPU();
            pullOutputCurrentSpikesFromDevice();
#else
            stepTimeCPU();
#endif
        }

        {
            TimerAccumulate<std::milli> timer(render);
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                applyOutputSpikes(spikeCount_Output, spike_Output, output);
            }
        }

        // Get time of tick start
        auto tickEnd = std::chrono::high_resolution_clock::now();

        // If there we're ahead of real-time pause
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

    // Wait for display thread to die
    displayThread.join();

    // Stop DVS
    std::cout << "Ran for " << i << " " << DT << "ms timesteps, overan for " << overrunTime.count() << "ms, slept for " << sleepTime.count() << "ms" << std::endl;

#ifdef LIVE
    dvs.stop();
    std::cout << "DVS:" << dvsGet << "ms, Step:" << step << "ms, Render:" << render << std::endl;
#else
    std::cout << "Read:" << read << "ms, Step:" << step << "ms, Render:" << render << std::endl;
#endif

    return 0;
}