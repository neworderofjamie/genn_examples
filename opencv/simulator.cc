// Standard C++ includes
#include <fstream>
#include <limits>
#include <iostream>
#include <set>
#include <sstream>
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



int main(int argc, char *argv[])
{
    OpenCVDVS dvs((argc > 1) ? std::atoi(argv[1]) : 0, 32);
    
    // Configure windows which will be used to show down-sampled images
    cv::namedWindow("Downsampled frame", CV_WINDOW_NORMAL);
    cv::namedWindow("Frame difference", CV_WINDOW_NORMAL);
    cv::namedWindow("P Membrane voltage", CV_WINDOW_NORMAL);
    cv::resizeWindow("Downsampled frame", 320, 320);
    cv::resizeWindow("Frame difference", 320, 320);
    cv::resizeWindow("P Membrane voltage", 320, 320);
    
    allocateMem();
    initialize();
    
    initopencv();

    for(unsigned int i = 0;; i++)
    {
        // Read DVS state and put result into GeNN
        tie(inputCurrentsP, stepP) = dvs.update(i);

        // Show raw frame and difference with previous
        dvs.showDownsampledFrame("Downsampled frame", i);
        dvs.showFrameDifference("Frame difference");

        // Simulate
#ifndef CPU_ONLY
        stepTimeGPU();

        pullPStateFromDevice();
#else
        stepTimeCPU();
#endif
        
        cv::Mat wrappedVoltage(32, 32, CV_32FC1, VP);
        cv::imshow("P Membrane voltage", wrappedVoltage);
        // **YUCK** required for OpenCV GUI to do anything
        cv::waitKey(1);
    }


    return 0;
}