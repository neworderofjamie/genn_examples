#include "snapshot_processor.h"

//----------------------------------------------------------------------------
// SnapshotProcessor
//----------------------------------------------------------------------------
SnapshotProcessor::SnapshotProcessor(unsigned int intermediateWidth, unsigned int intermediateHeight,
                                     unsigned int outputWidth, unsigned int outputHeight)
:   m_IntermediateWidth(intermediateWidth), m_IntermediateHeight(intermediateHeight),
    m_OutputWidth(outputWidth), m_OutputHeight(outputHeight),
    m_IntermediateSnapshotGreyscale(intermediateHeight, intermediateWidth, CV_8UC1),
    m_FinalSnapshot(outputHeight, outputWidth, CV_8UC1),
    m_FinalSnapshotFloat(outputHeight, outputWidth, CV_32FC1),
    m_FinalSnapshotFloatGPU(outputHeight, outputWidth, CV_32FC1),
    m_Clahe(cv::createCLAHE(40.0, cv::Size(8, 8)))
{
}
//----------------------------------------------------------------------------
std::tuple<float*, unsigned int> SnapshotProcessor::process(const cv::Mat &snapshot)
{
    // **TODO** theoretically this processing could all be done on the GPU but
    // a) we're currently starting from a snapshot in host memory
    // b) CLAHE seems broken for GPU matrices
    assert((unsigned int)snapshot.rows == m_IntermediateHeight * 4);
    assert((unsigned int)snapshot.cols == m_IntermediateWidth * 4);
    std::cout << snapshot.cols << "," << snapshot.rows << "," << snapshot.depth() << std::endl;
    std::cout << m_IntermediateSnapshotGreyscale.cols << "," << m_IntermediateSnapshotGreyscale.rows << "," << m_IntermediateSnapshotGreyscale.depth() << std::endl;

    // Perform weird averaging used in Matlab code
    for(unsigned int y = 0; y < m_IntermediateHeight; y++) {
        for(unsigned int x = 0; x < m_IntermediateWidth; x++) {
            const unsigned int a = snapshot.at<cv::Vec3b>((y * 4) + 1, (x * 4) + 1)[1];
            const unsigned int b = snapshot.at<cv::Vec3b>((y * 4) + 2, (x * 4) + 1)[1];
            const unsigned int c = snapshot.at<cv::Vec3b>((y * 4) + 2, (x * 4) + 2)[1];
            const unsigned int d = snapshot.at<cv::Vec3b>((y * 4) + 1, (x * 4) + 2)[1];

            //std::cout << a << "," << b << "," << c << "," << d << "," << (a + b + c + d) / 4 << std::endl;
            m_IntermediateSnapshotGreyscale.at<uint8_t>(y, x) = (a + b + c + d) / 4;
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