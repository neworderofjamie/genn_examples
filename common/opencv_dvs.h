#pragma once

// Standard C++ includes
#include <iostream>

// OpenCV includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

//----------------------------------------------------------------------------
// OpenCVDVS
//----------------------------------------------------------------------------
//! Uses OpenCV video capture interface to provide low-resolution, square 
//! Image consisting of difference between frames:
//! pipe into a layer of neurons and bob's your cheap DVS uncle
class OpenCVDVS
{
public:
    OpenCVDVS(unsigned int device, unsigned int resolution)
        : m_Camera(device), m_Resolution(resolution)
    {
        // Check camera has opened correctly
        if(!m_Camera.isOpened()) {
            throw std::runtime_error("Cannot open camera");
        }
        
        // Get frame dimensions
        const unsigned int width = m_Camera.get(CV_CAP_PROP_FRAME_WIDTH);
        const unsigned int height = m_Camera.get(CV_CAP_PROP_FRAME_HEIGHT);
        std::cout << "Width:" << width << ", height:" << height << std::endl;
        
        const unsigned int margin = (width - height) / 2;
        const cv::Rect cameraSquare(cv::Point(margin, 0), cv::Point(width - margin, height));
        
        // Initialize and zero the two downsampled image
        m_DownsampledFrames[0].create(m_Resolution, m_Resolution, CV_32FC1);
        m_DownsampledFrames[1].create(m_Resolution, m_Resolution, CV_32FC1);
        m_DownsampledFrames[0].setTo(0);
        m_DownsampledFrames[1].setTo(0);
        
        // Create 3rd image to hold output
        m_FrameDifference.create(m_Resolution, m_Resolution, CV_32FC1);
        m_FrameDifference.setTo(0);

        // Read first frame from camera
        readFrame();
        
        // Create square Region of Interest within raw frame
        m_SquareROI = m_RawFrame(cameraSquare);
    }
    
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    std::pair<float*, unsigned int> update(unsigned int i)
    {
        // Get references to current and previous down-sampled frame
        cv::Mat &curDownSampledFrame = m_DownsampledFrames[i % 2];
        cv::Mat &prevDownSampledFrame = m_DownsampledFrames[(i + 1) % 2];
 
        // Convert square frame to greyscale and then to floating-point
        cv::cvtColor(m_SquareROI, m_GreyscaleFrame, CV_BGR2GRAY);
        m_GreyscaleFrame.convertTo(m_GreyscaleFrame, CV_32FC1, 1.0 / 255.0);
        
        // Resample greyscale camera output into current down-sampled frame
        cv::resize(m_GreyscaleFrame, curDownSampledFrame, 
                   cv::Size(m_Resolution, m_Resolution));
  
        // If this isn't first frame, calculate difference with previous frame
        if(i > 0) {
            m_FrameDifference = curDownSampledFrame - prevDownSampledFrame;
        }
    
        // Read next frame
        readFrame();
        
#ifndef CPU_ONLY
        // Upload frame difference to GPU
        m_FrameDifferenceGPU.upload(m_FrameDifference);
        
        // Get low-level structure containing device pointer and stride and return
        auto frameDifferencePtrStep = (cv::gpu::PtrStep<float>)m_FrameDifferenceGPU;
        return std::make_pair(frameDifferencePtrStep.data,
                              frameDifferencePtrStep.step / sizeof(float));
#else
        // Return frame difference data directly
        return std::make_pair(reinterpret_cast<float*>(m_FrameDifference.data),
                              m_Resolution);
#endif
    }
    
    void showDownsampledFrame(const char *name, unsigned int i)
    {
        cv::imshow(name, m_DownsampledFrames[i % 2]);
    }
    
    void showFrameDifference(const char *name)
    {
        cv::imshow(name, m_FrameDifference);
    }
    
    void showRawFrame(const char *name)
    {
        cv::imshow(name, m_RawFrame);
    }

    void showGreyscaleFrame(const char *name)
    {
        cv::imshow(name, m_GreyscaleFrame);
    }
    
    
private:
    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    void readFrame()
    {
        if(!m_Camera.read(m_RawFrame)) {
            throw std::runtime_error("Cannot read first frame");
        }
    }
    
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    cv::VideoCapture m_Camera;
    
    const unsigned int m_Resolution;
    
    // Downsampled frames to calculate output from
    cv::Mat m_DownsampledFrames[2];
    
    cv::Mat m_FrameDifference;

    // Full resolution, colour frame read directly from camera
    cv::Mat m_RawFrame;
    
    // Square region of interest within m_RawFrame used for subsequent processing
    cv::Mat m_SquareROI;
    
    // Intermediate full-resolution, greyscale frame 
    cv::Mat m_GreyscaleFrame;
    
#ifndef CPU_ONLY
    cv::gpu::GpuMat m_FrameDifferenceGPU;
#endif  // CPU_ONLY
};