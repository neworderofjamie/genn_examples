// Standard C++ includes
#include <iostream>

// Standard C includes
#include <cmath>

// Common includes
#include "../common/camera_360.h"

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

    for(unsigned int i = 0;; i++)
    {
        if(!camera.read()) {
            return 1;
        }

        // Show frame difference
        cv::imshow("Original", camera.getOriginalImage());
        cv::imshow("Unwrapped", camera.getUnwrappedImage());

        if(cv::waitKey(1) == 27) {
            break;
        }
    }
    
    return 0;
}


