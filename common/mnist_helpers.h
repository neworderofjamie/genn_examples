#pragma once

// Standard C++ includes
#include <fstream>

// Standard C includes
#include <cassert>

inline uint32_t readBigEndian(std::ifstream &data)
{
    union
    {
        char b[4];
        uint32_t w;
    } swizzle;

    // Read data into swizzle union
    data.read(&swizzle.b[0], 4);

    // Swap endianess
    std::swap(swizzle.b[0], swizzle.b[3]);
    std::swap(swizzle.b[1], swizzle.b[2]);
    return swizzle.w;
}

inline unsigned int loadImageData(const std::string &imageDatafilename, uint8_t *&egp, 
                                  void (*allocateEGPFn)(unsigned int), void (*pushEGPFn)(unsigned int))
{
    // Open binary file
    std::ifstream imageData(imageDatafilename, std::ifstream::binary);
    assert(imageData.good());

    // Read header words
    const uint32_t magic = readBigEndian(imageData);
    const uint32_t numImages = readBigEndian(imageData);
    const uint32_t numRows = readBigEndian(imageData);
    const uint32_t numCols = readBigEndian(imageData);

    // Validate header words
    assert(magic == 0x803);
    assert(numRows == 28);
    assert(numCols == 28);

    // Allocate EGP for data
    allocateEGPFn(numRows * numCols * numImages);

    // Read data into EGP
    imageData.read(reinterpret_cast<char*>(egp), numRows * numCols * numImages);

    // Push EGP
    pushEGPFn(numRows * numCols * numImages);

    return numImages;
}

inline void loadLabelData(const std::string &labelDataFilename, unsigned int desiredNumLabels, uint8_t *egp)
{
    // Open binary file
    std::ifstream labelData(labelDataFilename, std::ifstream::binary);
    assert(labelData.good());

    // Read header words
    const uint32_t magic = readBigEndian(labelData);
    const uint32_t numLabels = readBigEndian(labelData);

    // Validate header words
    assert(magic == 0x801);
    assert(numLabels == desiredNumLabels);
    
    // Read data into EGP
    labelData.read(reinterpret_cast<char *>(egp), numLabels);
}

inline void loadLabelData(const std::string &labelDataFilename, unsigned int desiredNumLabels, uint8_t *&egp,
                          void (*allocateEGPFn)(unsigned int), void (*pushEGPFn)(unsigned int))
{
    // Allocate EGP for data
    allocateEGPFn(desiredNumLabels);

    // Load label data
    loadLabelData(labelDataFilename, desiredNumLabels, egp);

    // Push EGP
    pushEGPFn(desiredNumLabels);
}

inline void loadDense(const std::string &weightFilename, scalar *weights, unsigned int count)
{
    std::ifstream file(weightFilename, std::ifstream::binary);    
    file.read(reinterpret_cast<char*>(weights), sizeof(scalar) * count);
}

inline void saveDense(const std::string &weightFilename, const scalar *weights, unsigned int count)
{
    std::ofstream file(weightFilename, std::ifstream::binary);    
    file.write(reinterpret_cast<const char*>(weights), sizeof(scalar) * count);
}