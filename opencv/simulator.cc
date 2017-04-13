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
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "opencv_CODE/definitions.h"

// Common example includes
#include "../common/analogue_csv_recorder.h"


int main()
{
    // Create video capture object to read from USB camera (1)
    cv::VideoCapture camera(1);
    if(!camera.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return 1;
    }
    
    // Get frame dimensions
    const unsigned int width = camera.get(CV_CAP_PROP_FRAME_WIDTH);
    const unsigned int height = camera.get(CV_CAP_PROP_FRAME_HEIGHT);
    std::cout << "Width:" << width << ", height:" << height << std::endl;
    
    const unsigned int margin = (width - height) / 2;
    const cv::Rect cameraSquare(cv::Point(margin, 0), cv::Point(width - margin, height));
    
    // Create two downsampled frame
    cv::Mat downSampledFrames[2];
    downSampledFrames[0].create(32, 32, CV_32FC1);
    downSampledFrames[1].create(32, 32, CV_32FC1);
    downSampledFrames[0].setTo(0);
    downSampledFrames[1].setTo(0);
    
    // Create 3rd matrix to hold difference
    cv::Mat frameDifference;
    frameDifference.create(32, 32, CV_32FC1);
    frameDifference.setTo(0);

#ifndef CPU_ONLY
    cv::gpu::GpuMat frameDifferenceGPU;
#endif
    
    // Read first frame from camera
    cv::Mat rawFrame;
    if(!camera.read(rawFrame)) {
        std::cerr << "Cannot read first frame" << std::endl;
        return 1;
    }
    
    // Create square Region of Interest within this
    cv::Mat squareFrame = rawFrame(cameraSquare);
    cv::Mat greyscaleFrame;
    
    // Configure windows which will be used to show down-sampled images
    cv::namedWindow("Downsampled frame", CV_WINDOW_NORMAL);
    cv::namedWindow("Frame difference", CV_WINDOW_NORMAL);
    cv::namedWindow("P Membrane voltage", CV_WINDOW_NORMAL);
    cv::resizeWindow("Downsampled frame", height, height);
    cv::resizeWindow("Frame difference", height, height);
    cv::resizeWindow("P Membrane voltage", height, height);
    
    allocateMem();
    initialize();
    
    initopencv();

    for(unsigned int i = 0;; i++)
    {
        // Get references to current and previous down-sampled frame
        cv::Mat &curDownSampledFrame = downSampledFrames[i % 2];
        cv::Mat &prevDownSampledFrame = downSampledFrames[(i + 1) % 2];
 
        // Convert square frame to greyscale
        cv::cvtColor(squareFrame, greyscaleFrame, CV_BGR2GRAY);
        greyscaleFrame.convertTo(greyscaleFrame, CV_32FC1, 1.0 / 255.0);
        
        // Resample greyscale camera output into current down-sampled frame
        cv::resize(greyscaleFrame, curDownSampledFrame, cv::Size(32, 32));
        
         // Show raw frame
        //cv::imshow("Raw frame", squareFrame);
        //cv::imshow("Greyscale frame", greyscaleFrame);
        cv::imshow("Downsampled frame", curDownSampledFrame);
        
        // If this isn't first frame
        if(i > 0) {
            // Calculate difference with previous frame
            frameDifference = curDownSampledFrame - prevDownSampledFrame;
            assert(frameDifference.type() == CV_32FC1);
            cv::imshow("Frame difference", frameDifference);
        }
    
#ifndef CPU_ONLY
        // Upload frame difference to GPU
        frameDifferenceGPU.upload(frameDifference);
        
        auto frameDifferencePtrStep = (cv::gpu::PtrStep<float>)frameDifferenceGPU;
        inputCurrentsP = frameDifferencePtrStep.data;
        stepP = frameDifferencePtrStep.step / sizeof(float);
#else
        assert(frameDifference.isContinuous());
        inputCurrentsP = (float*)frameDifference.data;
#endif
        // Read next frame
        if(!camera.read(rawFrame)) {
            std::cerr << "Cannot read frame" << std::endl;
            return 1;
        }

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