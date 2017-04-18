// Standard C++ includes
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <vector>

// Standard C includes
#include <cassert>
#include <cstdlib>

// OpenCV includes
#include <opencv2/highgui/highgui.hpp>

// Common example code
#include "../common/opencv_dvs.h"
#include "../common/timer.h"

// LGMD includes
#include "parameters.h"

#include "lgmd_opencv_CODE/definitions.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
typedef void (*allocateFn)(unsigned int);
typedef TimerAccumulate<std::chrono::microseconds>  ProfilingTimer;

unsigned int get_neuron_index(unsigned int resolution, unsigned int x, unsigned int y)
{
    return x + (y * resolution);
}

void print_sparse_matrix(unsigned int pre_resolution, const SparseProjection &projection)
{
    const unsigned int pre_size = pre_resolution * pre_resolution;
    for(unsigned int i = 0; i < pre_size; i++)
    {
        std::cout << i << ":";

        for(unsigned int j = projection.indInG[i]; j < projection.indInG[i + 1]; j++)
        {
            std::cout << projection.ind[j] << ",";
        }

        std::cout << std::endl;
    }
}

void build_one_to_one_connection(unsigned int resolution,
                                 SparseProjection &projection, allocateFn allocate)
{
    // Allocate one connection per neuron
    const unsigned int pop_size = resolution * resolution;
    allocate(pop_size);

    for(unsigned int i = 0; i < pop_size; i++)
    {
        projection.indInG[i] = i;
        projection.ind[i] = i;
    }

    projection.indInG[pop_size] = pop_size;
}

void build_centre_to_one_connection(unsigned int pre_resolution, unsigned int centre_size,
                                    SparseProjection &projection, allocateFn allocate)
{
    // Allocate centre_size * centre_size connections
    allocate(centre_size * centre_size);

    // Calculate start and end of border on each row
    const unsigned int border_size = (pre_resolution - centre_size) / 2;
    const unsigned int far_border = pre_resolution - border_size;

    // Loop through rows of pixels in centre
    unsigned int s = 0;
    unsigned int i = 0;
    for(unsigned int yi = 0; yi < pre_resolution; yi++)
    {
        for(unsigned int xi = 0; xi < pre_resolution; xi++)
        {
            projection.indInG[i++] = s;

            // If we're in the centre
            if(xi >= border_size && xi < far_border && yi >= border_size && yi < far_border) {
                projection.ind[s++] = 0;
            }
        }
    }

    // Add ending entry to data structure
    projection.indInG[i] = s;

    // Check
    assert(s == (centre_size * centre_size));
    assert(i == (pre_resolution * pre_resolution));
}

void build_i_s_connections(unsigned int resolution, unsigned int centre_size,
                           SparseProjection &projection1, allocateFn allocate1,
                           SparseProjection &projection2, allocateFn allocate2,
                           SparseProjection &projection4, allocateFn allocate4)
{
    const unsigned int pop_size = resolution * resolution;
    
    // Allocate sparse projections
    allocate1(centre_size * centre_size * 4);
    allocate2(centre_size * centre_size * 4);
    allocate4(centre_size * centre_size * 4);

    // Calculate start and end of border on each row
    const unsigned int border_size = (resolution - centre_size) / 2;
    const unsigned int far_border = resolution - border_size;

    // Create temporary vectors to hold rows
    std::vector<std::vector<unsigned int>> projection1Map(pop_size);
    std::vector<std::vector<unsigned int>> projection2Map(pop_size);
    std::vector<std::vector<unsigned int>> projection4Map(pop_size);

    // Loop through POSTsynaptic neurons
    for(unsigned int xj =  border_size; xj < far_border; xj++)
    {
        for(unsigned int yj = border_size; yj < far_border; yj++)
        {
            const unsigned int j = get_neuron_index(resolution, xj, yj);

            // Add adjacent neighbours to projection 1
            projection1Map[get_neuron_index(resolution, xj - 1, yj)].push_back(j);
            projection1Map[get_neuron_index(resolution, xj, yj - 1)].push_back(j);
            projection1Map[get_neuron_index(resolution, xj + 1, yj)].push_back(j);
            projection1Map[get_neuron_index(resolution, xj, yj + 1)].push_back(j);

            // Add diagonal neighbours to projection 2
            projection2Map[get_neuron_index(resolution, xj - 1, yj - 1)].push_back(j);
            projection2Map[get_neuron_index(resolution, xj + 1, yj - 1)].push_back(j);
            projection2Map[get_neuron_index(resolution, xj + 1, yj + 1)].push_back(j);
            projection2Map[get_neuron_index(resolution, xj - 1, yj + 1)].push_back(j);

            // Add one away neighbours to projection 4
            projection4Map[get_neuron_index(resolution, xj - 2, yj)].push_back(j);
            projection4Map[get_neuron_index(resolution, xj, yj - 2)].push_back(j);
            projection4Map[get_neuron_index(resolution, xj + 2, yj)].push_back(j);
            projection4Map[get_neuron_index(resolution, xj, yj + 2)].push_back(j);
        }
    }

    // Convert vector of vectors to GeNN format
    unsigned int s1 = 0;
    unsigned int s2 = 0;
    unsigned int s4 = 0;
    for(unsigned int i = 0; i < pop_size; i++)
    {
        projection1.indInG[i] = s1;
        projection2.indInG[i] = s2;
        projection4.indInG[i] = s4;

        std::copy(projection1Map[i].cbegin(), projection1Map[i].cend(), &projection1.ind[s1]);
        std::copy(projection2Map[i].cbegin(), projection2Map[i].cend(), &projection2.ind[s2]);
        std::copy(projection4Map[i].cbegin(), projection4Map[i].cend(), &projection4.ind[s4]);

        s1 += projection1Map[i].size();
        s2 += projection2Map[i].size();
        s4 += projection4Map[i].size();
    }

    // Add ending entries to data structure
    projection1.indInG[pop_size] = s1;
    projection2.indInG[pop_size] = s2;
    projection4.indInG[pop_size] = s4;

    // Check
    assert(s1 == (centre_size * centre_size * 4));
    assert(s2 == (centre_size * centre_size * 4));
    assert(s4 == (centre_size * centre_size * 4));
}
}   // Anonymous namespace

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
    cv::namedWindow("S Membrane voltage", CV_WINDOW_NORMAL);
    cv::resizeWindow("Downsampled frame", 320, 320);
    cv::resizeWindow("Frame difference", 320, 320);
    cv::resizeWindow("P Membrane voltage", 320, 320);
    cv::resizeWindow("S Membrane voltage", 320, 320);
    
    allocateMem();
    initialize();

    build_centre_to_one_connection(Parameters::input_size, Parameters::centre_size,
                                   CP_F_LGMD, &allocateP_F_LGMD);
    build_centre_to_one_connection(Parameters::input_size, Parameters::centre_size,
                                   CS_LGMD, &allocateS_LGMD);
    build_one_to_one_connection(Parameters::input_size,
                                CP_E_S, &allocateP_E_S);
    build_i_s_connections(Parameters::input_size, Parameters::centre_size,
                          CP_I_S_1, &allocateP_I_S_1,
                          CP_I_S_2, &allocateP_I_S_2,
                          CP_I_S_4, &allocateP_I_S_4);

    initlgmd_opencv();

    // Loop through timesteps until there is no more import
    double dvsUpdate = 0.0;
    double dvsRender = 0.0;
    double simulationStep = 0.0;
    double download = 0.0;
    double outputRender = 0.0;
    double eventProcessing = 0.0;
    for(unsigned int i = 0;; i++)
    {
        // Read DVS state and put result into GeNN
        {
            ProfilingTimer t(dvsUpdate);
            tie(inputCurrentsP, stepP) = dvs.update(i);
        }

        // Show raw frame and difference with previous
        {
            ProfilingTimer t(dvsRender);
            dvs.showDownsampledFrame("Downsampled frame", i);
            dvs.showFrameDifference("Frame difference");
        }
        
        // Simulate
#ifndef CPU_ONLY
        {
            ProfilingTimer t(simulationStep);
            stepTimeGPU();
        }

        //pullLGMDStateFromDevice();
        {
            ProfilingTimer t(download);
            
            pullPStateFromDevice();
            pullSStateFromDevice();
            pullLGMDCurrentSpikesFromDevice();
        }
#else
        {
            ProfilingTimer t(simulationStep);
            stepTimeCPU();
        }
#endif
        
        {
            ProfilingTimer t(outputRender);
            
            cv::Mat wrappedPVoltage(32, 32, CV_32FC1, VP);
            cv::imshow("P Membrane voltage", wrappedPVoltage);
            
            cv::Mat wrappedSVoltage(32, 32, CV_32FC1, VS);
            cv::imshow("S Membrane voltage", wrappedSVoltage);
            
            if(spikeCount_LGMD > 0) {
                std::cout << "LGMD SPIKE" << std::endl;
            }
        }
        
        // **YUCK** required for OpenCV GUI to do anything
        {
            ProfilingTimer t(eventProcessing);
            
            if(cv::waitKey(1) == 27) {
                std::cout << "DVS update:" << dvsUpdate / (double)i << std::endl;
                std::cout << "DVS render:" << dvsRender / (double)i  << std::endl;
                std::cout << "Simulation step:" << simulationStep / (double)i  << std::endl;
                std::cout << "Download:" << download / (double)i  << std::endl;
                std::cout << "Output render:" << outputRender / (double)i  << std::endl;
                std::cout << "Event processing:" << eventProcessing / (double)i  << std::endl;
                break;
            }
        }
    }
    
    

    return 0;
}