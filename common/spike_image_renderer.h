#pragma once

// Standard C includes
#include <cstdlib>

// OpenCV includes
#include <opencv2/core/mat.hpp>

inline void renderSpikeImage(unsigned int spikeCount, const unsigned int *spikes,
                             unsigned int width, float persistence, cv::Mat &image)
{
    // Loop through spikes
    for(unsigned int s = 0; s < spikeCount; s++)
    {
        // Convert spike ID to x, y
        const unsigned int spike = spikes[s];
        const auto spikeCoord = std::div((int)spike, (int)width);

        // Set pixel to be white
        image.at<float>(spikeCoord.quot, spikeCoord.rem) += 1.0f;
    }

    // Decay image
    image *= persistence;
}