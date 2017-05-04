// Standard C++ includes
#include <fstream>
#include <limits>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

// Standard C includes
#include <cassert>
#include <cstdlib>

// OpenCV includes
#include <opencv2/highgui/highgui.hpp>

// Common example code
#include "../common/analogue_csv_recorder.h"
#include "../common/opencv_dvs.h"

#include "opencv_CODE/definitions.h"

#include "parameters.h"

void centre_surround_kernel(unsigned int const k_w, unsigned int const k_h,
                            float const ctr_std_dev, float const srr_std_dev,
                            bool const is_on_centre, float *kernel);

int main(int argc, char *argv[])
{
    const unsigned int device = (argc > 1) ? std::atoi(argv[1]) : 0;
#ifndef CPU_ONLY
    OpenCVDVSGPU dvs(device, 32);
#else
    OpenCVDVSCPU dvs(device, 32);
#endif

    // Configure windows which will be used to show down-sampled images
    cv::namedWindow("Downsampled frame", CV_WINDOW_NORMAL);
    cv::namedWindow("Frame difference", CV_WINDOW_NORMAL);
    cv::namedWindow("P Membrane voltage", CV_WINDOW_NORMAL);
    cv::resizeWindow("Downsampled frame", 320, 320);
    cv::resizeWindow("Frame difference", 320, 320);
    cv::resizeWindow("P Membrane voltage", 320, 320);

    if(Parameters::useMagno) {
        cv::namedWindow("M Membrane voltage", CV_WINDOW_NORMAL);
        cv::resizeWindow("M Membrane voltage", 320, 320);
    }

    allocateMem();

    if(Parameters::useMagno) {
        for(unsigned int r = 1; r < 15; r++){
            for(unsigned int c = 1; c < 15; c++){
                unsigned int m2p_i = r*16 + c;
                unsigned int m_in_p_r = (r*2);
                unsigned int m_in_p_c = (c*2);
                for(unsigned int dr = -2; dr < 3; dr++){
                    for(unsigned int dc = -2; dc < 3; dc++){
                        unsigned int pr = m_in_p_r + dr;
                        unsigned int pc = m_in_p_c + dc;
                        unsigned int p2m_i = pr*32 + pc;
                        unsigned int w_idx = m2p_i*(32*32) + p2m_i;
                        float w = 2.0f/std::sqrt(dr*dr + dc*dc);
                        gMagnoToParvo[w_idx] = -4*w;
                    }
                }
            }
        }

        for(unsigned int r = 0; r < 15; r++){
            for(unsigned int c = 0; c < 15; c++){
                unsigned int m2p_i = r*16 + c;
                unsigned int m_in_p_r = (r*2 + 1);
                unsigned int m_in_p_c = (c*2 + 1);
                for(unsigned int dr = -1; dr < 2; dr++){
                    for(unsigned int dc = -1; dc < 2; dc++){
                        unsigned int pr = m_in_p_r + dr;
                        unsigned int pc = m_in_p_c + dc;
                        unsigned int p2m_i = pr*32 + pc;
                        unsigned int w_idx = p2m_i*(16*16) + m2p_i;
                        float w = 2.0f/std::sqrt(dr*dr + dc*dc);
                        gParvoToMagno[w_idx] = -w;
                    }
                }

            }
        }
    }

    initialize();

    initopencv();

    for(unsigned int i = 0;; i++)
    {
        // Read DVS state and put result into GeNN
        if(Parameters::useMagno) {
            tie(inputCurrentsP, inputCurrentsM, stepP, stepM) = dvs.updateMagno(i);
        }
        else {
            tie(inputCurrentsP, stepP) = dvs.update(i);
        }

        // Show raw frame and difference with previous
        dvs.showDownsampledFrame("Downsampled frame", i);
        dvs.showFrameDifference("Frame difference");

        // Simulate
#ifndef CPU_ONLY
        stepTimeGPU();
        pullPStateFromDevice();
#else
        stepTimeCPU();
#endif

        cv::Mat wrappedVoltageP(32, 32, CV_32FC1, VP);
        cv::imshow("P Membrane voltage", wrappedVoltageP);

        if(Parameters::useMagno) {
            cv::Mat wrappedVoltageM(16, 16, CV_32FC1, VM);
            cv::imshow("M Membrane voltage", wrappedVoltageM);
        }

        // **YUCK** required for OpenCV GUI to do anything
        cv::waitKey(1);
    }


    return 0;
}
