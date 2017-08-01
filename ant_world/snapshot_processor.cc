#include "snapshot_processor.h"

//----------------------------------------------------------------------------
// SnapshotProcessor
//----------------------------------------------------------------------------
SnapshotProcessor::SnapshotProcessor(unsigned int intermediateWidth, unsigned int intermediateHeight,
                                     unsigned int outputWidth, unsigned int outputHeight)
:   m_IntermediateWidth(intermediateWidth), m_IntermediateHeight(intermediateHeight),
    m_OutputWidth(outputWidth), m_OutputHeight(outputHeight),
    m_IntermediateSnapshot(intermediateWidth, intermediateHeight, CV_8UC3),
    m_IntermediateSnapshotGreyscale(intermediateWidth, intermediateHeight, CV_8UC1),
    m_FinalSnapshot(outputWidth, outputHeight, CV_8UC1),
    m_FinalSnapshotGPU(outputWidth, outputHeight, CV_8UC1),
    m_Clahe(cv::createCLAHE(0.01 * 255.0, cv::Size(8, 8)))
{
}
//----------------------------------------------------------------------------
std::tuple<uint8_t*, unsigned int> SnapshotProcessor::process(const cv::Mat &snapshot)
{
    // Downsample to intermediate size
    cv::resize(snapshot, m_IntermediateSnapshot,
               cv::Size(m_IntermediateWidth, m_IntermediateHeight));

    // Convert to greyscale
    cv::cvtColor(m_IntermediateSnapshot, m_IntermediateSnapshotGreyscale, CV_BGR2GRAY);

    // Invert image
    cv::subtract(255, m_IntermediateSnapshotGreyscale, m_IntermediateSnapshotGreyscale);

    // Apply histogram normalization
    m_Clahe->apply(m_IntermediateSnapshotGreyscale, m_IntermediateSnapshotGreyscale);

    // Finally resample down to final size
    cv::resize(m_IntermediateSnapshotGreyscale, m_FinalSnapshot,
                cv::Size(m_OutputWidth, m_OutputHeight),
                0.0, 0.0, CV_INTER_CUBIC);

    //cv::imwrite("snapshot.png", m_FinalSnapshot);

    // Upload final snapshot to GPU
    m_FinalSnapshotGPU.upload(m_FinalSnapshot);

    // Extract device pointers and step; and return
    auto finalSnapshotPtrStep = (cv::cuda::PtrStep<uint8_t>)m_FinalSnapshotGPU;
    return std::make_tuple(finalSnapshotPtrStep.data, finalSnapshotPtrStep.step);
}