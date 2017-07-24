#pragma once

// Standard C++ includes
#include <iostream>
#include <string>
#include <vector>

// Standard C includes
#include <cstdio>

// Libpng includes
#include <png.h>


void read_png(const std::string &filename, float scale, bool rowMajor, float *data)
{
    std::cout << "Loading:" << filename << std::endl;
    // open file and test for it being a png
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error(filename + " could not be opened for reading");
    }

    png_byte header[8];    // 8 is the maximum size that can be checked
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8)) {
        throw std::runtime_error(filename + " is not recognized as a PNG file");
    }

    // initialize stuff
    png_structp pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

    if (!pngPtr) {
        throw std::runtime_error("png_create_read_struct failed");
    }

    png_infop infoPtr = png_create_info_struct(pngPtr);
    if (!infoPtr) {
        throw std::runtime_error("png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(pngPtr))) {
        throw std::runtime_error("Error during init_io");
    }

    png_init_io(pngPtr, fp);
    png_set_sig_bytes(pngPtr, 8);

    png_read_info(pngPtr, infoPtr);

    // Check format is compatible
    if(png_get_color_type(pngPtr, infoPtr) != PNG_COLOR_TYPE_GRAY) {
        throw std::runtime_error("Only greyscale images are supported");
    }
    if(png_get_bit_depth(pngPtr, infoPtr) != 8) {
        throw std::runtime_error("Only images with 8-bit per channel are supported");
    }

    // Read dimensions
    const int width = png_get_image_width(pngPtr, infoPtr);
    const int height = png_get_image_height(pngPtr, infoPtr);
    std::cout << "\tImage width:" << width << ", height:" << height << std::endl;

    // read file
    png_read_update_info(pngPtr, infoPtr);
    if (setjmp(png_jmpbuf(pngPtr))) {
        throw std::runtime_error("Error during read_image");
    }

    // Allocate rows
    png_bytepp rowPointers = new png_bytep[height];
    for (int y = 0; y < height; y++) {
        rowPointers[y] = new png_byte[png_get_rowbytes(pngPtr, infoPtr)];
    }

    // Read image
    png_read_image(pngPtr, rowPointers);

    // Loop through rows
    for (int y = 0; y < height; y++) {
        png_bytep row = rowPointers[y];
        for (int x = 0; x < width; x++) {
            // Get scaled pixel value
            float v = (scale / 255.0) * (float)row[x];

            // Write data to output pointer
            if(rowMajor) {
                data[(y * width) + x] = v;
            }
            else {
                data[(x * height) + y] = v;
            }
        }
    }

    // Delete rows
    for (int y = 0; y < height; y++) {
        delete [] rowPointers[y];
    }
    delete [] rowPointers;

    // Close file
    fclose(fp);
}