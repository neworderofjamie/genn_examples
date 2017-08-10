#include "perfect_memory.h"

// For file IO
#include <sys/stat.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// Create PM model for images with specified width and height (after processing)
PerfectMemory::PerfectMemory(unsigned int outputWidth, unsigned int outputHeight)
:   m_diff(outputWidth, outputHeight, CV_32FC1),
    m_tmp1(outputWidth, outputHeight, CV_32FC1),
    m_tmp2(outputWidth, outputHeight, CV_16UC1)
{
#ifdef PM_LOG
    // Check if log directory exists and exit if so
    struct stat sb;
    if (stat(PM_LOG_DIR, &sb) == 0 && S_ISDIR(sb.st_mode))
        return;

    // Otherwise create directory with usual permissions
    const int dir_err = mkdir(PM_LOG_DIR, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (dir_err < 0) {
        cout << "Error creating directory: " << dir_err << endl;
        exit(1);
    }
#endif
}

// Add a new snapshot to memory
void PerfectMemory::addSnapshot(Mat &current)
{
    cout << "\tAdding snapshot " << snapshots.size() << endl;

    // Clone the current view
    Mat snap = current.clone();
    
    // Add to vector
    snapshots.push_back(snap);

#ifdef PM_LOG
    imwrite(PM_LOG_DIR "snapshot" + to_string(snapshots.size()) + ".png", snap);
#endif
}

// Get the heading etc. by comparing current view to all stored snapshots
void PerfectMemory::getHeading(Mat &current, PerfectMemoryResult &res)
{
#ifdef PM_LOG
    testCount++;

    // Prefix for log files
    string pref = to_string(testCount) + "_";
    
    // Save current view
    imwrite(PM_LOG_DIR + pref + "current.png", current);

    // CSV file to store RIDF output
    ofstream ridfFile;
    ridfFile.open(PM_LOG_DIR + pref + "ridf.csv", ios::out | ios::trunc);
#endif

    // Stores the view as we rotate it azimuthally
    Mat shiftSnap(current.size(), current.type());
    
    // RIDF values
    int minrot; // best rotation (as index)
    uint minsnap; // best-matching snapshot
    double minval = numeric_limits<double>::infinity(); // lowest value across snapshots and rotations
    
    // Compare current view at all rotations against all snapshots
    for (int i = 0; i < current.cols; i++) {
        // Shift current to the right by i pixels
        shiftColumns(current, i, shiftSnap);
        
        // Iterate through all snapshots
        for (uint j = 0; j < snapshots.size(); j++) {
            // Get sum absolute difference between current (rotated) view and snapshot
            absdiff(snapshots.at(j), shiftSnap, m_diff);
            double sumabsdiff = sum(m_diff)[0];
            
            // If this value is lower than previous best, update
            if (sumabsdiff < minval) {
                minval = sumabsdiff;
                minrot = i;
                minsnap = j;
            }

#ifdef PM_LOG
            if (j > 0) {
                ridfFile << ", ";
            }
            ridfFile << sumabsdiff;
#endif
        }
#ifdef PM_LOG
        ridfFile << "\n";
#endif
    }
    
    // Best rotation
    double ratio = (double)minrot / (double)current.cols;

#ifdef PM_LOG
    ridfFile.close();

    // Store a text representation of RIDF output
    ofstream logfile;
    logfile.open(PM_LOG_DIR + pref + "log.txt", ios::out | ios::trunc);
    double rot = 360.0 * ratio; // degrees
    logfile << "test" << testCount << ": " << endl
            << "- rotation: " << rot << endl
            << "- snap: " << minsnap << " (n=" << snapshots.size() << ")" << endl
            << "- value: " << minval << endl << endl;
    logfile.close();
#endif
    
    // Fill PerfectMemoryResult struct
    res.heading = 2 * M_PI * ratio; // radians
    res.snapshot = minsnap;
    res.minval = minval;

    cout << "\tHeading: " << 360.0 * ratio << " deg" << endl
         << "\tBest-matching snapshot: " << minsnap << " (of " << snapshots.size() << ")" << endl
         << "\tMinimum value: " << minval << endl;
}

// Shift an image to the right by numRight pixels
void shiftColumns(Mat in, int numRight, Mat &out) {
    // Special case: no rotation
    if(numRight == 0) { 
        in.copyTo(out);
        return;
    }

    int ncols = in.cols;
    int nrows = in.rows;

    out = Mat::zeros(in.size(), in.type());

    // Adjust for when numRight < 0 || numRight >= ncols
    numRight = numRight % ncols;
    if(numRight < 0)
        numRight = ncols+numRight;

    // Shift columns right
    in(Rect(ncols-numRight,0, numRight,nrows)).copyTo(out(Rect(0,0,numRight,nrows)));
    in(Rect(0,0, ncols-numRight,nrows)).copyTo(out(Rect(numRight,0,ncols-numRight,nrows)));
}