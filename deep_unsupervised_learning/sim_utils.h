#pragma once

// Standard C++ includes
#include <fstream>

int loadImageData(const std::string &imageDatafilename, uint8_t *&egp, unsigned int numNeurons,
                  void (*allocateEGPFn)(unsigned int), void (*pushEGPFn)(unsigned int))
{
    // Open binary file
    std::ifstream imageData(imageDatafilename, std::ifstream::binary);
    assert(imageData.good());

    // Get file length
    imageData.seekg(0, std::ios_base::end);
    const auto fileBytes = imageData.tellg();
    imageData.seekg(0, std::ios_base::beg);

    // Determine how many images this equates to
    const auto numImages = std::div(long long{fileBytes}, long long{numNeurons});
    assert(numImages.rem == 0);


    // Allocate EGP for data
    allocateEGPFn(numNeurons * numImages.quot);

    // Read data into EGP
    imageData.read(reinterpret_cast<char*>(egp), numNeurons * numImages.quot);

    // Push EGP
    pushEGPFn(numNeurons * numImages.quot);

    return numImages.quot;
}