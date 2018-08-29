#pragma once

// Standard C++ includes
#include <iostream>
#include <mutex>

// OpenCV includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifndef CPU_ONLY
#include <opencv2/core/cuda.hpp>
#endif  // CPU_ONLY

//----------------------------------------------------------------------------
// OpenCVDVS
//----------------------------------------------------------------------------
//! Uses OpenCV video capture interface to provide low-resolution, square 
//! Image consisting of difference between frames:
//! pipe into a layer of neurons and bob's your cheap DVS uncle
class OpenCVDVS
{
public:
    OpenCVDVS(unsigned int device, unsigned int resolution, bool absolute)
        : m_Camera(device), m_Resolution(resolution), m_Absolute(absolute)
    {
        // Check camera has opened correctly
        if(!m_Camera.isOpened()) {
            throw std::runtime_error("Cannot open camera");
        }
        
         // Read first frame from camera
        readFrame();
        
        // Create square Region of Interest within raw frame
        m_SquareROI = m_RawFrame(getCameraSquare());       
    }
    
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::pair<float*, unsigned int> update(unsigned int i) = 0;
    virtual void showDownsampledFrame(const char *name, unsigned int i) = 0;
    virtual void showFrameDifference(const char *name) = 0;
    virtual void showGreyscaleFrame(const char *name) = 0;
    
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void showRawFrame(const char *name)
    {
        cv::imshow(name, m_RawFrame);
    }
    
protected:
    //----------------------------------------------------------------------------
    // Protected methods
    //----------------------------------------------------------------------------
    void readFrame()
    {
        if(!m_Camera.read(m_RawFrame)) {
            throw std::runtime_error("Cannot read first frame");
        }
    }
    
    cv::Rect getCameraSquare()
    {
        // Get frame dimensions
        const unsigned int width = m_Camera.get(CV_CAP_PROP_FRAME_WIDTH);
        const unsigned int height = m_Camera.get(CV_CAP_PROP_FRAME_HEIGHT);
         
        const unsigned int margin = (width - height) / 2;
        return cv::Rect(cv::Point(margin, 0), cv::Point(width - margin, height));
    }
    
    unsigned int getResolution() const
    {
        return m_Resolution;
    }

    bool isAbsolute() const
    {
        return m_Absolute;
    }
    
    const cv::Mat &getSquareROI() const
    {
        return m_SquareROI;
    }
    
private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    cv::VideoCapture m_Camera;
    
    // Square resolution DVS operates at
    const unsigned int m_Resolution;

    // Should frame difference be absolute
    const bool m_Absolute;
    
    // Full resolution, colour frame read directly from camera
    cv::Mat m_RawFrame;
    
    // Square region of interest within m_RawFrame used for subsequent processing
    cv::Mat m_SquareROI;
};

//----------------------------------------------------------------------------
// OpenCVDVSCPU
//----------------------------------------------------------------------------
class OpenCVDVSCPU : public OpenCVDVS
{
public:
    OpenCVDVSCPU(unsigned int device, unsigned int resolution, bool absolute=false)
        : OpenCVDVS(device, resolution, absolute)
    {
        // Initialize and zero the two downsampled image
        m_DownsampledFrames[0].create(getResolution(), getResolution(), CV_32FC1);
        m_DownsampledFrames[1].create(getResolution(), getResolution(), CV_32FC1);
        m_DownsampledFrames[0].setTo(0);
        m_DownsampledFrames[1].setTo(0);
        
        // Create 3rd image to hold output
        m_FrameDifference.create(getResolution(), getResolution(), CV_32FC1);
        m_FrameDifference.setTo(0);
    }
    
    //----------------------------------------------------------------------------
    // OpenCVDVS virtuals
    //----------------------------------------------------------------------------
    virtual std::pair<float*, unsigned int> update(unsigned int i) override
    {
        // Get references to current and previous down-sampled frame
        auto &curDownSampledFrame = m_DownsampledFrames[i % 2];
        auto &prevDownSampledFrame = m_DownsampledFrames[(i + 1) % 2];
        
        // Convert square frame to floating-point using CPU
        cv::cvtColor(getSquareROI(), m_GreyscaleFrame, CV_BGR2GRAY);
        
        // Convert greyscale frame to floating point
        m_GreyscaleFrame.convertTo(m_GreyscaleFrame, CV_32FC1, 1.0 / 255.0);

        // Resample greyscale camera output into current down-sampled frame
        cv::resize(m_GreyscaleFrame, curDownSampledFrame, 
                   cv::Size(getResolution(), getResolution()));
        
        // If this isn't first frame, calculate difference with previous frame
        if(i > 0) {
            if(isAbsolute()) {
                cv::absdiff(curDownSampledFrame, prevDownSampledFrame, m_FrameDifference);
            }
            else {
                cv::subtract(curDownSampledFrame, prevDownSampledFrame, m_FrameDifference);
            }
        }
    
        // Read next frame
        readFrame();

        // Return frame difference data directly
        return std::make_pair(reinterpret_cast<float*>(m_FrameDifference.data),
                              getResolution());
    }
    
    virtual void showDownsampledFrame(const char *name, unsigned int i) override
    {
        cv::imshow(name, m_DownsampledFrames[i % 2]);
    }
    
    virtual void showFrameDifference(const char *name) override
    {
        cv::imshow(name, m_FrameDifference);
    }

    virtual void showGreyscaleFrame(const char *name) override
    {
        cv::imshow(name, m_GreyscaleFrame);
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    const cv::Mat &getFrameDifference() const{ return m_FrameDifference; }
    
private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    cv::Mat m_GreyscaleFrame;
    
    cv::Mat m_DownsampledFrames[2];
    cv::Mat m_FrameDifference;
};

//----------------------------------------------------------------------------
// OpenCVDVSGPU
//----------------------------------------------------------------------------
#ifndef CPU_ONLY
class OpenCVDVSGPU : public OpenCVDVS
{
public:
    OpenCVDVSGPU(unsigned int device, unsigned int resolution, bool absolute=false)
        : OpenCVDVS(device, resolution, absolute)
    {
        // Create GPU matrix to upload squared camera input into
        auto cameraSquare = getCameraSquare();
        m_SquareROIGPU.create(cameraSquare.width, cameraSquare.height, CV_8UC3);
        
        // Initialize and zero the two downsampled image
        m_DownsampledFrames[0].create(getResolution(), getResolution(), CV_32FC1);
        m_DownsampledFrames[1].create(getResolution(), getResolution(), CV_32FC1);
        m_DownsampledFrames[0].setTo(0);
        m_DownsampledFrames[1].setTo(0);
        
        // Create 3rd image to hold output
        m_FrameDifference.create(getResolution(), getResolution(), CV_32FC1);
        m_FrameDifference.setTo(0);
    }
    
    //----------------------------------------------------------------------------
    // OpenCVDVS virtuals
    //----------------------------------------------------------------------------
    virtual std::pair<float*, unsigned int> update(unsigned int i, std::mutex &mutex) override
    {
        // Get references to current and previous down-sampled frame
        auto &curDownSampledFrame = m_DownsampledFrames[i % 2];
        auto &prevDownSampledFrame = m_DownsampledFrames[(i + 1) % 2];
    
        // Upload camera data to GPU
        m_SquareROIGPU.upload(getSquareROI());
        
        // Convert square frame to floating-point using GPU
        cv::cuda::cvtColor(m_SquareROIGPU, m_GreyscaleFrame, CV_BGR2GRAY);
        
        // Convert greyscale frame to floating point
        m_GreyscaleFrame.convertTo(m_GreyscaleFrame, CV_32FC1, 1.0 / 255.0);

        // Resample greyscale camera output into current down-sampled frame
        cv::cuda::resize(m_GreyscaleFrame, curDownSampledFrame,
                        cv::Size(getResolution(), getResolution()));
        
        // If this isn't first frame, calculate difference with previous frame
        if(i > 0) {
            std::lock_guard<std::mutex> lock(outputMutex);
            if(isAbsolute()) {
                cv::cuda::absdiff(curDownSampledFrame, prevDownSampledFrame, m_FrameDifference);
            }
            else {
                cv::cuda::subtract(curDownSampledFrame, prevDownSampledFrame, m_FrameDifference);
            }
        }
    
        // Read next frame
        readFrame();
        
        // Get low-level structure containing device pointer and stride and return
        auto frameDifferencePtrStep = (cv::cuda::PtrStep<float>)m_FrameDifference;
        return std::make_pair(frameDifferencePtrStep.data,
                              frameDifferencePtrStep.step / sizeof(float));
    }
    
    virtual void showDownsampledFrame(const char *name, unsigned int i) override
    {
        cv::Mat downsampledFrame;
        m_DownsampledFrames[i % 2].download(downsampledFrame);
        
        cv::imshow(name, downsampledFrame);
    }
    
    virtual void showFrameDifference(const char *name) override
    {
        cv::Mat frameDifference;
        m_FrameDifference.download(frameDifference);
        
        cv::imshow(name, frameDifference);
    }

    virtual void showGreyscaleFrame(const char *name) override
    {
        cv::Mat greyscaleFrame;
        m_GreyscaleFrame.download(greyscaleFrame);
        
        cv::imshow(name, greyscaleFrame);
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    const cv::cuda::GpuMat &getFrameDifference() const{ return m_FrameDifference; }
    
private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    // GPU matrix m_SquareROI is uploaded to 
    cv::cuda::GpuMat m_SquareROIGPU;
    
    cv::cuda::GpuMat m_GreyscaleFrame;
    
    // Downsampled frames to calculate output from
    cv::cuda::GpuMat m_DownsampledFrames[2];
    cv::cuda::GpuMat m_FrameDifference;
};
#endif  // CPU_ONLY
