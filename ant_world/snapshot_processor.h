#pragma once

// OpenCV includes
#include <opencv2/opencv.hpp>

//----------------------------------------------------------------------------
// SnapshotProcessor
//----------------------------------------------------------------------------
class SnapshotProcessor
{
public:
    SnapshotProcessor(unsigned int intermediateWidth, unsigned int intermediateHeight,
                      unsigned int outputWidth, unsigned int outputHeight);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    // Process input snapshot (probably at screen resolution)
    // and return GPU data pointer and step
    std::tuple<float*, unsigned int> process(const cv::Mat &snapshot);

private:
    //------------------------------------------------------------------------
    // Private members
    //------------------------------------------------------------------------
    // Dimensions of intermediate image
    const unsigned int m_IntermediateWidth;
    const unsigned int m_IntermediateHeight;

    // Dimensions of final output
    const unsigned int m_OutputWidth;
    const unsigned int m_OutputHeight;

    // Host OpenCV array to hold intermediate resolution colour snapshot
    cv::Mat m_IntermediateSnapshot;

    // Host OpenCV array to hold intermediate resolution greyscale snapshot
    cv::Mat m_IntermediateSnapshotGreyscale;

    // Host OpenCV array to hold final resolution greyscale snapshot
    cv::Mat m_FinalSnapshot;

    cv::Mat m_FinalSnapshotFloat;

    // GPU OpenCV array to hold
    cv::cuda::GpuMat m_FinalSnapshotFloatGPU;

    // CLAHE algorithm for histogram normalization
    cv::Ptr<cv::CLAHE> m_Clahe;
};