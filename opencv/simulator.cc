// Standard C++ includes
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <iostream>
#include <ratio>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

// Standard C includes
#include <cassert>
#include <cstdlib>

// OpenCV includes
#include <opencv2/highgui/highgui.hpp>

// Common example code
#include "../common/analogue_csv_recorder.h"
#include "../common/opencv_dvs.h"

#include "opencv_CODE/definitions.h"

typedef std::chrono::high_resolution_clock sim_clock;

void gennThreadHandler(const std::atomic<bool> &exit)
{
    typedef std::chrono::duration<double, std::milli> double_ms;

    sim_clock::duration totalSleepTime;
    sim_clock::duration totalOverrunTime;

    const double_ms dtDurationMs{DT};
    const sim_clock::duration dtDuration = std::chrono::duration_cast<sim_clock::duration>(dtDurationMs);

    unsigned int i = 0;
    for(;!exit; i++)
    {
        // Get time step started at
        const auto stepStart = sim_clock::now();

        // Simulate
#ifndef CPU_ONLY
        stepTimeGPU();

        pullPSFORtateFromDevice();
#else
        stepTimeCPU();
#endif

        const auto stepEnd = sim_clock::now();
        const auto stepLength = stepEnd - stepStart;

        if(stepLength > dtDuration) {
            totalOverrunTime += (stepLength - dtDuration);
        }
        else {
            const auto sleepTime = dtDuration - stepLength;
            std::this_thread::sleep_for(sleepTime);
            totalSleepTime += sleepTime;
        }

    }

    std::cout << "Ran for " << i << " " << double(DT) <<  "ms timesteps and overran by:" << double_ms(totalOverrunTime).count() << "ms and slept for " << double_ms(totalSleepTime).count() << "ms" << std::endl;
}

void cameraThreadHandler(OpenCVDVS &dvs, std::atomic<bool> &exit)
{
    typedef std::chrono::duration<double> double_s;

    // Configure windows which will be used to show down-sampled images
    cv::namedWindow("Downsampled frame", CV_WINDOW_NORMAL);
    cv::namedWindow("Frame difference", CV_WINDOW_NORMAL);
    cv::namedWindow("P Membrane voltage", CV_WINDOW_NORMAL);
    cv::resizeWindow("Downsampled frame", 320, 320);
    cv::resizeWindow("Frame difference", 320, 320);
    cv::resizeWindow("P Membrane voltage", 320, 320);

    const auto cameraBegin = sim_clock::now();

    unsigned int i;
    for(i = 1;; i++)
    {
        // Read DVS state and put result into GeNN
        tie(inputCurrentsP, stepP) = dvs.update(i);

        // Show raw frame and difference with previous
        dvs.showDownsampledFrame("Downsampled frame", i);
        dvs.showFrameDifference("Frame difference");

        cv::Mat wrappedVoltage(32, 32, CV_32FC1, VP);
        cv::imshow("P Membrane voltage", wrappedVoltage);

        // **YUCK** required for OpenCV GUI to do anything
        if(cv::waitKey(1) == 27) {
            exit = true;
            break;
        }
    }

    const auto cameraEnd = sim_clock::now();
    const auto cameraTimeS = double_s(cameraEnd - cameraBegin);
    std::cout << (double)(i - 1) / cameraTimeS.count() << " FPS" << std::endl;
}

int main(int argc, char *argv[])
{
     const unsigned int device = (argc > 1) ? std::atoi(argv[1]) : 0;
#ifndef CPU_ONLY
    OpenCVDVSGPU dvs(device, 32);
#else
    OpenCVDVSCPU dvs(device, 32);
#endif

    allocateMem();
    initialize();
    
    initopencv();

    // Read DVS state and put result into GeNN
    tie(inputCurrentsP, stepP) = dvs.update(0);


    std::atomic<bool> exit(false);
    std::thread gennThread(gennThreadHandler, std::ref(exit));
    std::thread cameraThread(cameraThreadHandler, std::ref(dvs), std::ref(exit));

    gennThread.join();
    cameraThread.join();
    return 0;
}