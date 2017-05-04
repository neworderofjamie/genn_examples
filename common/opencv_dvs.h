#pragma once

// Standard C++ includes
#include <iostream>

// OpenCV includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifndef CPU_ONLY
#include <opencv2/gpu/gpu.hpp>
#else
#include <math.h>
#endif  // CPU_ONLY

void centre_surround_kernel(unsigned int const size, float const ctr_std_dev,
                            float const srr_std_dev, cv::Mat &kernel,
                            bool const is_on_centre=true){

    if (is_on_centre){
        kernel = cv::getGaussianKernel(size, ctr_std_dev, CV_32F) - \
                 cv::getGaussianKernel(size, srr_std_dev, CV_32F);
    }
    else{
        kernel = cv::getGaussianKernel(size, srr_std_dev, CV_32F) - \
                 cv::getGaussianKernel(size, ctr_std_dev, CV_32F);
    }
    kernel -= cv::mean(kernel);
    cv::Mat square = cv::Mat::zeros(size, size, CV_32F);
    cv::multiply(kernel, kernel, square);
    cv::divide(cv::sum(square), kernel, kernel);
}
//----------------------------------------------------------------------------
// OpenCVDVS
//----------------------------------------------------------------------------
//! Uses OpenCV video capture interface to provide low-resolution, square
//! Image consisting of difference between frames:
//! pipe into a layer of neurons and bob's your cheap DVS uncle
class OpenCVDVS
{
public:
    OpenCVDVS(unsigned int device, unsigned int resolution, bool absolute, float threshold)
        : m_Camera(device), m_Resolution(resolution), m_Absolute(absolute), m_Threshold(threshold),\
          m_Size(resolution*resolution)
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
    virtual std::tuple<float*, float*, unsigned int, unsigned int> updateMagno(unsigned int i) = 0;
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

    unsigned int getSize() const{
        return m_Size;
    }

    float getThreshold() const{
        return m_Threshold;
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
    const float m_Threshold;
    const unsigned int m_Size;

};

//----------------------------------------------------------------------------
// OpenCVDVSCPU
//----------------------------------------------------------------------------
class OpenCVDVSCPU : public OpenCVDVS
{
public:
    OpenCVDVSCPU(unsigned int device, unsigned int resolution, bool absolute=false, float threshold=0.05f)
        : OpenCVDVS(device, resolution, absolute, threshold)
    {
        // Initialize and zero the two downsampled image
        m_DownsampledFrames[0].create(getResolution(), getResolution(), CV_32FC1);
        m_DownsampledFrames[1].create(getResolution(), getResolution(), CV_32FC1);
        m_DownsampledFrames[0].setTo(0);
        m_DownsampledFrames[1].setTo(0);

        // Create 3rd image to hold output
        m_FrameDifference.create(getResolution(), getResolution(), CV_32FC1);
        m_FrameDifference.setTo(0);

        m_FrameDifferenceM.create(getResolutionM(), getResolutionM(), CV_32FC1);

        m_AbsFrameDifference.create(getResolution(), getResolution(), CV_32FC1);
        m_AbsFrameDifference.setTo(0);

        m_FrameDifferenceMask.create(getResolution(), getResolution(), CV_8UC1);
        m_FrameDifferenceMask.setTo(0);

        m_RefFrame.create(getResolution(), getResolution(), CV_32FC1);
        m_RefFrame.setTo(0);

        m_RefFrame.create(getResolution(), getResolution(), CV_32FC1);
        m_RefFrame.setTo(threshold);

        m_OnCentre3.create(3, 3, CV_32FC1);
        //based on basab's thesis
        centre_surround_kernel(3, 0.8f, 0.8f*6.7f, m_OnCentre3);
    }

    //----------------------------------------------------------------------------
    // OpenCVDVS virtuals
    //----------------------------------------------------------------------------
    virtual std::pair<float*, unsigned int> update(unsigned int i) override
    {

        updateCommon(i);

        // Read next frame
        readFrame();

        // Return frame difference data directly
        return std::make_pair(reinterpret_cast<float*>(m_FrameDifference.data),
                              getResolution());



    }

    virtual std::tuple<float*, float*, unsigned int, unsigned int> updateMagno(unsigned int i) override
    {
        updateCommon(i);

        cv::resize(m_FrameDifference, m_FrameDifferenceM,
                   cv::Size(getResolutionM(), getResolutionM()), cv::INTER_NEAREST);
        cv::filter2D(m_FrameDifferenceM, m_FrameDifferenceM, m_FrameDifferenceM.depth(),
                     m_OnCentre3);

        // Read next frame
        readFrame();


        return std::make_tuple(reinterpret_cast<float*>(m_FrameDifference.data),
                               reinterpret_cast<float*>(m_FrameDifferenceM.data),
                               getResolution(),
                               getResolutionM());
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
    void updateCommon(unsigned int i)
    {
        // Get references to current and previous down-sampled frame
        auto &curDownSampledFrame = m_DownsampledFrames[i % 2];
        auto &prevDownSampledFrame = m_DownsampledFrames[(i + 1) % 2];

        // Convert square frame to floating-point using CPU
        cv::cvtColor(getSquareROI(), m_GreyscaleFrame, CV_BGR2GRAY);

        // Convert greyscale frame to floating point
        m_GreyscaleFrame.convertTo(m_GreyscaleFrame, CV_32FC1, m_imgValScale);

        // Resample greyscale camera output into current down-sampled frame
        cv::resize(m_GreyscaleFrame, curDownSampledFrame,
                   cv::Size(getResolution(), getResolution()),
                   cv::INTER_NEAREST);

        cv::subtract(curDownSampledFrame, m_RefFrame, m_FrameDifference);
        m_AbsFrameDifference = cv::abs(m_FrameDifference);

        cv::threshold(m_AbsFrameDifference, m_AbsFrameDifference, getThreshold(),\
                      10.0f, cv::THRESH_TOZERO);
        m_AbsFrameDifference.convertTo(m_FrameDifferenceMask, CV_8UC1, 10);

        // cv::add(m_RefFrame*0.99f, m_FrameDifference, m_RefFrame, m_FrameDifferenceMask);
        cv::add(m_RefFrame, m_FrameDifference, m_RefFrame, m_FrameDifferenceMask);
        cv::filter2D(m_FrameDifference, m_FrameDifference, m_FrameDifference.depth(),
                     m_OnCentre3);

    }

    unsigned int getResolutionM() const{
        return m_ResolutionM;
    }
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    cv::Mat m_GreyscaleFrame;
    cv::Mat m_RefFrame;
    cv::Mat m_DownsampledFrames[2];
    cv::Mat m_FrameDifference;
    cv::Mat m_AbsFrameDifference;
    cv::Mat m_FrameDifferenceMask;
    cv::Mat m_OnCentre3;
    float m_MinThresh = 0.02;
    float m_MaxThresh = 0.1;
    cv::Mat m_ThresholdFrame;

    cv::Mat m_FrameDifferenceM;
    const unsigned int m_ResolutionM = getResolution()/2;
    const unsigned int m_SizeM = m_ResolutionM*m_ResolutionM;

    float m_pixDifference;
    float m_imgValScale = 1.0f / 255.0f;


};

//----------------------------------------------------------------------------
// OpenCVDVSGPU
//----------------------------------------------------------------------------
#ifndef CPU_ONLY
class OpenCVDVSGPU : public OpenCVDVS
{
public:
    OpenCVDVSGPU(unsigned int device, unsigned int resolution, bool absolute=false, float threshold=0.05)
        : OpenCVDVS(device, resolution, absolute, threshold)
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
    virtual std::pair<float*, unsigned int> update(unsigned int i) override
    {
        // Get references to current and previous down-sampled frame
        auto &curDownSampledFrame = m_DownsampledFrames[i % 2];
        auto &prevDownSampledFrame = m_DownsampledFrames[(i + 1) % 2];

        // Upload camera data to GPU
        m_SquareROIGPU.upload(getSquareROI());

        // Convert square frame to floating-point using GPU
        cv::gpu::cvtColor(m_SquareROIGPU, m_GreyscaleFrame, CV_BGR2GRAY);

        // Convert greyscale frame to floating point
        m_GreyscaleFrame.convertTo(m_GreyscaleFrame, CV_32FC1, 1.0 / 255.0);

        // Resample greyscale camera output into current down-sampled frame
        cv::gpu::resize(m_GreyscaleFrame, curDownSampledFrame,
                        cv::Size(getResolution(), getResolution()));

        // If this isn't first frame, calculate difference with previous frame
        if(i > 0) {
            if(isAbsolute()) {
                cv::gpu::absdiff(curDownSampledFrame, prevDownSampledFrame, m_FrameDifference);
            }
            else {
                cv::gpu::subtract(curDownSampledFrame, prevDownSampledFrame, m_FrameDifference);
            }
        }

        // Read next frame
        readFrame();

        // Get low-level structure containing device pointer and stride and return
        auto frameDifferencePtrStep = (cv::gpu::PtrStep<float>)m_FrameDifference;
        return std::make_pair(frameDifferencePtrStep.data,
                              frameDifferencePtrStep.step / sizeof(float));
    }

    virtual std::tuple<float*, float*, unsigned int, unsigned int> updateMagno(unsigned int i) override
    {
        auto output = update(i);
        return std::make_tuple(output.first, NULL, output.second, 0);
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
    const cv::gpu::GpuMat &getFrameDifference() const{ return m_FrameDifference; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    // GPU matrix m_SquareROI is uploaded to
    cv::gpu::GpuMat m_SquareROIGPU;

    cv::gpu::GpuMat m_GreyscaleFrame;

    // Downsampled frames to calculate output from
    cv::gpu::GpuMat m_DownsampledFrames[2];
    cv::gpu::GpuMat m_FrameDifference;
};
#endif  // CPU_ONLY
