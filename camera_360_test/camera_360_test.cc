// Standard C++ includes
#include <iostream>

// Standard C includes
#include <cassert>
#include <cmath>

// OpenCV includes
#include <opencv2/superres/optical_flow.hpp>

// Common includes
#include "../common/camera_360.h"

void renderOpticalFlow(const cv::Mat &flowX, const cv::Mat &flowY, cv::Mat &flow, int scale)
{
    assert(flow.cols == flowX.cols * scale);
    assert(flow.rows == flowX.rows * scale);
    
    flow.setTo(cv::Scalar::all(0));
    
    // Loop through output coordinates
    for(unsigned int x = 0; x < flowX.cols; x++)
    {
        for(unsigned int y = 0; y < flowX.rows; y++)
        {
            const cv::Point start(x * scale, y * scale);
            const cv::Point end = start + cv::Point((float)scale * flowX.at<float>(y, x),
                                                    (float)scale * flowY.at<float>(y, x));

            cv::line(flow, start, end,
                     CV_RGB(0xFF, 0xFF, 0xFF));
        }
    }
}
int main(int argc, char *argv[])
{
    const unsigned int device = (argc > 1) ? std::atoi(argv[1]) : 0;
    
    std::cout << "CV version:" << CV_VERSION << std::endl;
    
    Camera360 camera(device, cv::Size(640, 480), cv::Size(90, 10),
                     0.5, 0.416, 0.173, 0.377);
    
    // Create motor
    cv::namedWindow("Unwrapped", CV_WINDOW_NORMAL);
    cv::resizeWindow("Unwrapped", 900, 100);
    
    cv::namedWindow("Original", CV_WINDOW_NORMAL);
    cv::resizeWindow("Original", 640, 480);

    // Create two grayscale frames
    cv::Mat frames[2];
    frames[0].create(10, 90, CV_8UC1);
    frames[1].create(10, 90, CV_8UC1);
    
    cv::Ptr<cv::superres::FarnebackOpticalFlow> opticalFlow = cv::superres::createOptFlow_Farneback();
    cv::Mat flowX;
    cv::Mat flowY;
    cv::Mat flowImage(100, 900, CV_8UC1);
    for(unsigned int i = 0;; i++)
    {
        if(!camera.read()) {
            return 1;
        }

        // Show frame difference
        cv::imshow("Original", camera.getOriginalImage());
        cv::imshow("Unwrapped", camera.getUnwrappedImage());
        
        // Convert frame to grayscale and store in array
        const unsigned int currentFrame = i % 2;
        cv::cvtColor(camera.getUnwrappedImage(), frames[currentFrame], CV_BGR2GRAY);
        
        // If this isn't the first frame, calculate optical flow
        if(i > 0) {
            const unsigned int prevFrame = (i - 1) % 2;
            opticalFlow->calc(frames[prevFrame], frames[currentFrame], flowX, flowY);
            
            renderOpticalFlow(flowX, flowY, flowImage, 10);
            
            cv::imshow("Optical flow", flowImage);
            //std::cout << flowX.cols << ", " << flowX.rows << ", " << flowX.channels() << std::endl;
        }
        if(cv::waitKey(1) == 27) {
            break;
        }
    }
    
    return 0;
}


