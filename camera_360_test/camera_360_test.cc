// Standard C++ includes
#include <iostream>

// Standard C includes
#include <cassert>
#include <cmath>

// OpenCV includes
#include <opencv2/superres/optical_flow.hpp>

// Common includes
#include "../common/camera_360.h"

// Anonymous namespace
namespace
{
constexpr float pi = 3.141592653589793238462643383279502884f;

void renderOpticalFlow(const cv::Mat &flowX, const cv::Mat &flowY, cv::Mat &flow, int scale)
{
    assert(flow.cols == flowX.cols * scale);
    assert(flow.rows == flowX.rows * scale);

    // Clear image
    flow.setTo(cv::Scalar::all(0));

    // Loop through output coordinates
    for(unsigned int x = 0; x < flowX.cols; x++)
    {
        for(unsigned int y = 0; y < flowX.rows; y++)
        {
            // Draw line showing direction of optical flow
            const cv::Point start(x * scale, y * scale);
            const cv::Point end = start + cv::Point((float)scale * flowX.at<float>(y, x),
                                                    (float)scale * flowY.at<float>(y, x));
            cv::line(flow, start, end,
                     CV_RGB(0xFF, 0xFF, 0xFF));
        }
    }
}

void buildFilter(cv::Mat &filter, float preferredAngle) {
    // Loop through columns
    for(unsigned int x = 0; x < filter.cols; x++) {
        // Convert column to angle
        const float th = (((float)x / (float)filter.cols) * 2.0f * pi) - pi;

        // Write filter with sin of angle
        filter.at<float>(0, x) = sin(th - preferredAngle);

        std::cout << x << " = " << filter.at<float>(0, x) << std::endl;
    }
}
}   // Anonymous namespace

int main(int argc, char *argv[])
{
    const unsigned int device = (argc > 1) ? std::atoi(argv[1]) : 0;

    std::cout << "CV version:" << CV_VERSION << std::endl;

    Camera360 camera(device, cv::Size(640, 480), cv::Size(90, 10),
                     0.5, 0.416, 0.173, 0.377, -pi);

    // Create motor
    cv::namedWindow("Unwrapped", CV_WINDOW_NORMAL);
    cv::resizeWindow("Unwrapped", 900, 100);

    cv::namedWindow("Original", CV_WINDOW_NORMAL);
    cv::resizeWindow("Original", 640, 480);

    // Create two grayscale frames to hold optical flow
    cv::Mat frames[2];
    frames[0].create(10, 90, CV_8UC1);
    frames[1].create(10, 90, CV_8UC1);

    // Build a velocity filter whose preferred angle is going straighj
    cv::Mat velocityFilter(1, 90, CV_32FC1);
    buildFilter(velocityFilter, 0.0f);

    cv::Ptr<cv::superres::FarnebackOpticalFlow> opticalFlow = cv::superres::createOptFlow_Farneback();
    cv::Mat flowX;
    cv::Mat flowY;
    cv::Mat flowImage(100, 900, CV_8UC1);
    cv::Mat flowXSum(1, 90, CV_32FC1);
    cv::Mat flowSum(1, 1, CV_32FC1);

    for(unsigned int i = 0;; i++)
    {
        if(!camera.read()) {
            return EXIT_FAILURE;
        }

        // Show frame difference
        cv::imshow("Original", camera.getOriginalImage());
        cv::imshow("Unwrapped", camera.getUnwrappedImage());

        // Convert frame to grayscale and store in array
        const unsigned int currentFrame = i % 2;
        cv::cvtColor(camera.getUnwrappedImage(), frames[currentFrame], CV_BGR2GRAY);

        // If this isn't the first frame
        if(i > 0) {
            // Calculate optical flow
            const unsigned int prevFrame = (i - 1) % 2;
            opticalFlow->calc(frames[prevFrame], frames[currentFrame], flowX, flowY);

            // Render optical flow
            renderOpticalFlow(flowX, flowY, flowImage, 10);

            // Reduce horizontal flow - summing along columns
            cv::reduce(flowX, flowXSum, 0, CV_REDUCE_SUM);

            // Multiply summed flow by filters
            cv::multiply(flowXSum, velocityFilter, flowXSum);

            // Reduce filtered flow - summing along rows
            cv::reduce(flowXSum, flowSum, 1, CV_REDUCE_SUM);

            char textBuffer[256];
            sprintf(textBuffer, "%f", flowSum.at<float>(0, 0));
            cv::putText(flowImage, textBuffer, cv::Point(0, 90), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0xFF, 0xFF, 0xFF));


            // Show flow
            cv::imshow("Optical flow", flowImage);
        }
        if(cv::waitKey(1) == 27) {
            break;
        }
    }

    return EXIT_SUCCESS;
}


