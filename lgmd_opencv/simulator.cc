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
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "lgmd_opencv_CODE/definitions.h"

// LGMD includes
#include "parameters.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
typedef void (*allocateFn)(unsigned int);

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

int main()
{
    // Create video capture object to read from USB camera (1)
    cv::VideoCapture camera(0);
    if(!camera.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return 1;
    }
    
    // Get frame dimensions
    const unsigned int width = camera.get(CV_CAP_PROP_FRAME_WIDTH);
    const unsigned int height = camera.get(CV_CAP_PROP_FRAME_HEIGHT);
    std::cout << "Width:" << width << ", height:" << height << std::endl;
    
    const unsigned int margin = (width - height) / 2;
    const cv::Rect cameraSquare(cv::Point(margin, 0), cv::Point(width - margin, height));
    
    // Create two downsampled frame
    cv::Mat downSampledFrames[2];
    downSampledFrames[0].create(32, 32, CV_32FC1);
    downSampledFrames[1].create(32, 32, CV_32FC1);
    downSampledFrames[0].setTo(0);
    downSampledFrames[1].setTo(0);
    
    // Create 3rd matrix to hold difference
    cv::Mat frameDifference;
    frameDifference.create(32, 32, CV_32FC1);
    frameDifference.setTo(0);

#ifndef CPU_ONLY
    cv::gpu::GpuMat frameDifferenceGPU;
#endif
    
    // Read first frame from camera
    cv::Mat rawFrame;
    if(!camera.read(rawFrame)) {
        std::cerr << "Cannot read first frame" << std::endl;
        return 1;
    }
    
    // Create square Region of Interest within this
    cv::Mat squareFrame = rawFrame(cameraSquare);
    cv::Mat greyscaleFrame;
    
    // Configure windows which will be used to show down-sampled images
    cv::namedWindow("Downsampled frame", CV_WINDOW_NORMAL);
    cv::namedWindow("Frame difference", CV_WINDOW_NORMAL);
    cv::namedWindow("P Membrane voltage", CV_WINDOW_NORMAL);
    cv::namedWindow("S Membrane voltage", CV_WINDOW_NORMAL);
    cv::resizeWindow("Downsampled frame", height, height);
    cv::resizeWindow("Frame difference", height, height);
    cv::resizeWindow("P Membrane voltage", height, height);
    cv::resizeWindow("S Membrane voltage", height, height);
    
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

    //SpikeCSVRecorder lgmdSpikeRecorder("lgmd_spikes.csv", glbSpkCntLGMD, glbSpkLGMD);
    //AnalogueCSVRecorder<scalar> sVoltageRecorder("s_voltages.csv", VS, Parameters::input_size * Parameters::input_size, "Voltage [mV]");
    //AnalogueCSVRecorder<scalar> lgmdVoltageRecorder("lgmd_voltages.csv", VLGMD, 1, "Voltage [mV]");

    // Loop through timesteps until there is no more import
    for(unsigned int i = 0;; i++)
    {
         // Get references to current and previous down-sampled frame
        cv::Mat &curDownSampledFrame = downSampledFrames[i % 2];
        cv::Mat &prevDownSampledFrame = downSampledFrames[(i + 1) % 2];
 
        // Convert square frame to greyscale
        cv::cvtColor(squareFrame, greyscaleFrame, CV_BGR2GRAY);
        greyscaleFrame.convertTo(greyscaleFrame, CV_32FC1, 1.0 / 255.0);
        
        // Resample greyscale camera output into current down-sampled frame
        cv::resize(greyscaleFrame, curDownSampledFrame, cv::Size(32, 32));
        
         // Show raw frame
        //cv::imshow("Raw frame", squareFrame);
        //cv::imshow("Greyscale frame", greyscaleFrame);
        cv::imshow("Downsampled frame", curDownSampledFrame);
        
        // If this isn't first frame
        if(i > 0) {
            // Calculate difference with previous frame
            frameDifference = curDownSampledFrame - prevDownSampledFrame;
            assert(frameDifference.type() == CV_32FC1);
            cv::imshow("Frame difference", frameDifference);
        }
    
#ifndef CPU_ONLY
        // Upload frame difference to GPU
        frameDifferenceGPU.upload(frameDifference);
        
        auto frameDifferencePtrStep = (cv::gpu::PtrStep<float>)frameDifferenceGPU;
        inputCurrentsP = frameDifferencePtrStep.data;
        stepP = frameDifferencePtrStep.step / sizeof(float);
#else
        assert(frameDifference.isContinuous());
        inputCurrentsP = (float*)frameDifference.data;
#endif
        // Read next frame
        if(!camera.read(rawFrame)) {
            std::cerr << "Cannot read frame" << std::endl;
            return 1;
        }

        // Simulate
#ifndef CPU_ONLY
        stepTimeGPU();

        //pullLGMDStateFromDevice();
        pullPStateFromDevice();
        pullSStateFromDevice();
        pullLGMDCurrentSpikesFromDevice();
#else
        stepTimeCPU();
#endif
        
        cv::Mat wrappedPVoltage(32, 32, CV_32FC1, VP);
        cv::imshow("P Membrane voltage", wrappedPVoltage);
        
        cv::Mat wrappedSVoltage(32, 32, CV_32FC1, VS);
        cv::imshow("S Membrane voltage", wrappedSVoltage);
        
        if(spikeCount_LGMD > 0) {
            std::cout << "LGMD SPIKE" << std::endl;
        }
        
        // **YUCK** required for OpenCV GUI to do anything
        cv::waitKey(1);
    }


    return 0;
}