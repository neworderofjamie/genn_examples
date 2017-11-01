#pragma once

// Standard C includes
#include <cassert>
#include <cmath>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

//----------------------------------------------------------------------------
// Camera360
//----------------------------------------------------------------------------
class Camera360
{
public:
    Camera360(int device, const cv::Size &cameraResolution, const cv::Size &unwrappedResolution,
              double centreX = 0.5, double centreY = 0.5, double inner = 0.1, double outer = 0.5)
    {
        if(!open(device, cameraResolution, unwrappedResolution, 
            centreX, centreY, inner, outer)) 
        {
            throw std::runtime_error("Unable to open camera");
        }
    }
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool open(int device, const cv::Size &cameraResolution, const cv::Size &unwrappedResolution,
              double centreX = 0.5, double centreY = 0.5, double inner = 0.1, double outer = 0.5)
    {
        // Open capture device
        if(!m_Capture.open(device)) {
            std::cerr << "Unable to open device" << std::endl;
            return false;
        }
        
        assert(cameraResolution.width == m_Capture.get(cv::CAP_PROP_FRAME_WIDTH));
        assert(cameraResolution.height == m_Capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        // Set camera resolution
        /*if(!m_Capture.set(cv::CAP_PROP_FRAME_WIDTH, cameraResolution.width)) {
            std::cerr << "Unable to set width to " << cameraResolution.width << std::endl;
            return false;
        }
        if(!m_Capture.set(cv::CAP_PROP_FRAME_HEIGHT, cameraResolution.height)) {
            std::cerr << "Unable to set height to " << cameraResolution.height << std::endl;
            return false;
        }*/
        
        // convert relative (0.0 to 1.0) to absolute pixel values
        const int centreXPixel = (int)round((double)cameraResolution.width * centreX);
        const int centreYPixel = (int)round((double)cameraResolution.height * centreY);
        const int innerPixel = (int)round((double)cameraResolution.height * inner); 
        const int outerPixel = (int)round((double)cameraResolution.height * outer); 

        // Create original image
        m_OriginalImage.create(cameraResolution, CV_8UC3);
        
        // Create x and y pixel maps
        m_UnwrapMapX.create(unwrappedResolution, CV_32FC1);
        m_UnwrapMapY.create(unwrappedResolution, CV_32FC1);
        
        // Create output image
        m_OutputImage.create(unwrappedResolution, CV_8UC3);
        
        // Build unwrap maps
        const float pi = 3.141592653589793238462643383279502884f;
        for (int i = 0; i < unwrappedResolution.height; i++) {
            for (int j = 0; j < unwrappedResolution.width; j++) {
                const float r = ((float)i / (float)unwrappedResolution.height) * (outerPixel - innerPixel) + innerPixel;
                const float th = ((float)j / (float)unwrappedResolution.width) * 2.0f * pi;
                const float x = centreXPixel - r * sin(th);
                const float y = centreYPixel + r * cos(th);
                m_UnwrapMapX.at<float>(i, j) = x;
                m_UnwrapMapY.at<float>(i, j) = y;
            }
        }
        
        return true;
    }
    
    bool read()
    {
        // Capture frame
        if(!m_Capture.read(m_OriginalImage)) {
            return false;
        }
        
        // Unwrap captured image and return
        cv::remap(m_OriginalImage, m_OutputImage, m_UnwrapMapX, m_UnwrapMapY, cv::INTER_NEAREST);
        return true;
    }
    
    const cv::Mat &getOriginalImage() const { return m_OriginalImage; }
    const cv::Mat &getUnwrappedImage() const { return m_OutputImage; }
    
private:
    //------------------------------------------------------------------------
    // Private members
    //------------------------------------------------------------------------
    cv::VideoCapture m_Capture;
    
    cv::Mat m_OriginalImage;
    cv::Mat m_UnwrapMapX;
    cv::Mat m_UnwrapMapY;
    cv::Mat m_OutputImage;
    
};