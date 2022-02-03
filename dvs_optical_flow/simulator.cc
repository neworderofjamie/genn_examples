// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>

// Standard C includes
#include <cassert>
#include <csignal>
#include <cstdlib>

// OpenCV includes
#include <opencv2/opencv.hpp>

// GeNN userproject includes
#include "timer.h"

// Common includes
#include "../common/dvs_128.h"


// Model includes
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

void buildCentreToMacroConnection(unsigned int *rowLength, unsigned int *ind)
{
    // Calculate start and end of border on each row
    const unsigned int near_border = (Parameters::inputSize - Parameters::centreSize) / 2;
    const unsigned int far_border = near_border + Parameters::centreSize;

    // Loop through rows of pixels in centre
    unsigned int i = 0;
    for(unsigned int yi = 0; yi < Parameters::inputSize; yi++)
    {
        for(unsigned int xi = 0; xi < Parameters::inputSize; xi++)
        {
            // If we're in the centre
            if(xi >= near_border && xi < far_border && yi >= near_border && yi < far_border) {
                const unsigned int yj = (yi - near_border) / Parameters::kernelSize;
                const unsigned int xj = (xi - near_border) / Parameters::kernelSize;
                
                ind[i] = getNeuronIndex(Parameters::macroPixelSize, xj, yj);
                rowLength[i++] = 1;
            }
            else {
                rowLength[i++] = 0;
            }
        }
    }

    // Check
    assert(i == (Parameters::inputSize * Parameters::inputSize));
}

void buildDetectors(unsigned int *excitatoryRowLength, unsigned int *excitatoryInd,
                    unsigned int *inhibitoryRowLength, unsigned int *inhibitoryInd)
{
    // Loop through macro cells
    unsigned int iExcitatory = 0;
    unsigned int iInhibitory = 0;
    for(unsigned int yi = 0; yi < Parameters::macroPixelSize; yi++)
    {
        for(unsigned int xi = 0; xi < Parameters::macroPixelSize; xi++)
        {
            // Get index of start of row
            unsigned int sExcitatory = (iExcitatory * Parameters::DetectorMax);
            unsigned int sInhibitory = (iInhibitory * Parameters::DetectorMax);
            
            // If we're not in border region
            if(xi >= 1 && xi < (Parameters::macroPixelSize - 1)
                && yi >= 1 && yi < (Parameters::macroPixelSize - 1))
            {
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;

                // Add excitatory synapses to all detectors
                excitatoryInd[sExcitatory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorLeft, yj);
                excitatoryInd[sExcitatory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorRight, yj);
                excitatoryInd[sExcitatory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorUp, yj);
                excitatoryInd[sExcitatory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                                         xj + Parameters::DetectorDown, yj);
                excitatoryRowLength[iExcitatory++] = 4;
            }
            else {
                excitatoryRowLength[iExcitatory++] = 0;
            }


            // Create inhibitory connection to 'left' detector associated with macropixel one to right
            inhibitoryRowLength[iInhibitory] = 0;
            if(xi < (Parameters::macroPixelSize - 2)
                && yi >= 1 && yi < (Parameters::macroPixelSize - 1))
            {
                const unsigned int xj = (xi - 1 + 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;
                inhibitoryInd[sInhibitory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                              xj + Parameters::DetectorLeft, yj);
                inhibitoryRowLength[iInhibitory]++;
            }

            // Create inhibitory connection to 'right' detector associated with macropixel one to right
            if(xi >= 2
                && yi >= 1 && yi < (Parameters::macroPixelSize - 1))
            {
                const unsigned int xj = (xi - 1 - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;
                inhibitoryInd[sInhibitory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                              xj + Parameters::DetectorRight, yj);
                inhibitoryRowLength[iInhibitory]++;
            }

            // Create inhibitory connection to 'up' detector associated with macropixel one below
            if(xi >= 1 && xi < (Parameters::macroPixelSize - 1)
                && yi < (Parameters::macroPixelSize - 2))
            {
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1 + 1;
                inhibitoryInd[sInhibitory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                              xj + Parameters::DetectorUp, yj);
                inhibitoryRowLength[iInhibitory]++;
            }

            // Create inhibitory connection to 'down' detector associated with macropixel one above
            if(xi >= 1 && xi < (Parameters::macroPixelSize - 1)
                && yi >= 2)
            {
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1 - 1;
                inhibitoryInd[sInhibitory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                              xj + Parameters::DetectorDown, yj);
                inhibitoryRowLength[iInhibitory]++;
            }
            iInhibitory++;

        }
    }

    // Check
    assert(iExcitatory == (Parameters::macroPixelSize * Parameters::macroPixelSize));
    assert(iInhibitory == (Parameters::macroPixelSize * Parameters::macroPixelSize));
}

void displayThreadHandler(std::mutex &inputMutex, const cv::Mat &inputImage,
                          std::mutex &outputMutex, const float (&output)[Parameters::detectorSize][Parameters::detectorSize][2])
{
    cv::namedWindow("Input", cv::WINDOW_NORMAL);
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

int main()
{
    allocateMem();
    initialize();

    buildCentreToMacroConnection(rowLengthDVS_MacroPixel, indDVS_MacroPixel);
    buildDetectors(rowLengthMacroPixel_Output_Excitatory, indMacroPixel_Output_Excitatory,
                   rowLengthMacroPixel_Output_Inhibitory, indMacroPixel_Output_Inhibitory);

    initializeSparse();

    // Create DVXplorer device
    DVS::DVXplorer dvs(DVS::Polarity::On);
    dvs.start();
    
    double dvsGet = 0.0;
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

        {
            TimerAccumulate timer(dvsGet);
            dvs.readEvents(spikeCount_DVS, spike_DVS);

            // Copy to GPU
            //pushDVSCurrentSpikesToDevice();
        }

        {
            TimerAccumulate timer(render);
            std::lock_guard<std::mutex> lock(inputMutex);

            {
                // Loop through spikes
                for(unsigned int s = 0; s < spikeCount_DVS; s++)
                {
                    // Convert spike ID to x, y
                    const unsigned int spike = spike_DVS[s];
                    const auto spikeCoord = std::div((int)spike, (int)Parameters::inputSize);

                    // Set pixel to be white
                    inputImage.at<float>(spikeCoord.quot, spikeCoord.rem) += 1.0f;
                }

                // Decay image
                inputImage *= Parameters::spikePersistence;
            }
        }

        {
            TimerAccumulate timer(step);

            // Simulate
            //stepTime();
            //pullOutputCurrentSpikesFromDevice();
        }

        {
            TimerAccumulate timer(render);
            {
                //std::lock_guard<std::mutex> lock(outputMutex);
                //applyOutputSpikes(spikeCount_Output, spike_Output, output);
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
    dvs.stop();

    std::cout << "Ran for " << i << " " << DT << "ms timesteps, overan for " << overrunTime.count() << "ms, slept for " << sleepTime.count() << "ms" << std::endl;
    std::cout << "DVS:" << dvsGet << "ms, Step:" << step << "ms, Render:" << render << std::endl;

    return 0;
}
