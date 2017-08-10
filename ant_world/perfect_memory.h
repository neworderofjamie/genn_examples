#pragma once

#include "opencv2/opencv.hpp"

// Whether to log algorithm's output
#define PM_LOG

// Where to store log files
#define PM_LOG_DIR "pm_dump/"

using namespace cv;

// For storing output of getHeading()
struct PerfectMemoryResult
{
    double heading;
    uint snapshot;
    double minval;
};

class PerfectMemory
{
public:
    // Create PM model for images with specified width and height (after processing)
    PerfectMemory(unsigned int outputWidth, unsigned int outputHeight);
    
    // Add a new snapshot to memory
    void addSnapshot(Mat &snap);
    
    // Get the heading etc. by comparing current view to all stored snapshots
    void getHeading(Mat &snap, PerfectMemoryResult &res);

private:
    std::vector<Mat> snapshots; // vector to store snapshots
    
    // temporary values
    Mat m_diff;
    Mat m_tmp1;
    Mat m_tmp2;
    
#ifdef PM_LOG
    // number of times getHeading has been called
    int testCount = -1;
#endif
};

// Shift an image to the right by numRight pixels
void shiftColumns(Mat in, int numRight, Mat &out);