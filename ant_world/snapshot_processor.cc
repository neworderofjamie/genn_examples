#include "snapshot_processor.h"

//----------------------------------------------------------------------------
// SnapshotProcessor
//----------------------------------------------------------------------------
SnapshotProcessor::SnapshotProcessor(unsigned int displayScale, unsigned int intermediateWidth, unsigned int intermediateHeight,
                                     unsigned int outputWidth, unsigned int outputHeight)
:   m_DisplayScale(displayScale), m_IntermediateWidth(intermediateWidth), m_IntermediateHeight(intermediateHeight),
    m_OutputWidth(outputWidth), m_OutputHeight(outputHeight),
    m_IntermediateSnapshotGreyscale(intermediateHeight, intermediateWidth, CV_8UC1),
    m_FinalSnapshot(outputHeight, outputWidth, CV_8UC1),
    m_FinalSnapshotFloat(outputHeight, outputWidth, CV_32FC1),
    m_FinalSnapshotFloatGPU(outputHeight, outputWidth, CV_32FC1),
    m_Clahe(cv::createCLAHE(40.0, cv::Size(8, 8)))
{
    // Check that the display scale is a multiple of 4
    assert(m_DisplayScale % 4 == 0);
}
//----------------------------------------------------------------------------
std::tuple<float*, unsigned int> SnapshotProcessor::process(const cv::Mat &snapshot)
{
    // **TODO** theoretically this processing could all be done on the GPU but
    // a) we're currently starting from a snapshot in host memory
    // b) CLAHE seems broken for GPU matrices

    // Check snapshot is expected size
    assert((unsigned int)snapshot.rows == m_IntermediateHeight * m_DisplayScale);
    assert((unsigned int)snapshot.cols == m_IntermediateWidth * m_DisplayScale);

    // Calculate start and end offset of resize kernel
    const unsigned int kernelStart = m_DisplayScale / 4;
    const unsigned int kernelEnd = m_DisplayScale - kernelStart;
    const unsigned int kernelPixels = kernelStart * m_DisplayScale;

    // Loop through intermediate image rows
    // **NOTE** this technique for downsampling the image is taken from the Matlab
    // and MASSIVELY improves performance over standard cv::imresize algorithms
    for(unsigned int y = 0; y < m_IntermediateHeight; y++) {
        // Calculate rows averaging kernel should operate over
        const unsigned int kernelStartY = (y * m_DisplayScale) + kernelStart;
        const unsigned int kernelEndY = (y * m_DisplayScale) + kernelEnd;

        // Loop through intermediate image columns
        for(unsigned int x = 0; x < m_IntermediateWidth; x++) {
            // Calculate columns averaging kernel should operate over
            const unsigned int kernelStartX = (x * m_DisplayScale) + kernelStart;
            const unsigned int kernelEndX = (x * m_DisplayScale) + kernelEnd;

            // Loop over snapshot pixels in kernel and sum their green components
            unsigned int sum = 0;
            for(unsigned i = kernelStartY; i < kernelEndY; i++) {
                for(unsigned int j = kernelStartX; j < kernelEndX; j++) {
                    sum += snapshot.at<cv::Vec3b>(i, j)[1];
                }
            }

            // Divide sum by number of pixels in kernel to compute average and write to intermediate snapshot
            m_IntermediateSnapshotGreyscale.at<uint8_t>(y, x) = sum / kernelPixels;
        }
    }

    // Invert image
    cv::subtract(255, m_IntermediateSnapshotGreyscale, m_IntermediateSnapshotGreyscale);

    // Apply histogram normalization
    // http://answers.opencv.org/question/15442/difference-of-clahe-between-opencv-and-matlab/
    m_Clahe->apply(m_IntermediateSnapshotGreyscale, m_IntermediateSnapshotGreyscale);

    // Finally resample down to final size
    cv::resize(m_IntermediateSnapshotGreyscale, m_FinalSnapshot,
                cv::Size(m_OutputWidth, m_OutputHeight),
                0.0, 0.0, CV_INTER_CUBIC);
    m_FinalSnapshot.convertTo(m_FinalSnapshotFloat, CV_32FC1, 1.0 / 255.0);

    cv::imwrite("snapshot.png", m_FinalSnapshot);

    // Normalise snapshot using L2 norm
    cv::normalize(m_FinalSnapshotFloat, m_FinalSnapshotFloat);

    // Upload final snapshot to GPU
    m_FinalSnapshotFloatGPU.upload(m_FinalSnapshotFloat);

    // Extract device pointers and step; and return
    auto finalSnapshotPtrStep = (cv::cuda::PtrStep<float>)m_FinalSnapshotFloatGPU;
    return std::make_tuple(finalSnapshotPtrStep.data, finalSnapshotPtrStep.step / sizeof(float));
}