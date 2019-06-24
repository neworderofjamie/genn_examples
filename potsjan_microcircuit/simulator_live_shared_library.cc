// Standard C++ includes
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>

// BoB robotics includes
#include "common/timer.h"
#include "genn_utils/shared_library_model.h"
#include "genn_utils/spike_csv_recorder.h"

// Model parameters
#include "parameters.h"

using namespace BoBRobotics;

void buildRowLengths(unsigned int numPre, unsigned int numPost, size_t numConnections, unsigned int *rowLengths, std::mt19937 &rng)
{
    // Calculate row lengths
    // **NOTE** we are FINISHING at second from last row because all remaining connections must go in last row
    size_t remainingConnections = numConnections;
    size_t matrixSize = (size_t)numPre * (size_t)numPost;
    std::generate_n(&rowLengths[0], numPre - 1,
                    [&remainingConnections, &matrixSize, numPost, &rng]()
                    {
                        const double probability = (double)numPost / (double)matrixSize;

                        // Create distribution to sample row length
                        std::binomial_distribution<size_t> rowLengthDist(remainingConnections, probability);

                        // Sample row length;
                        const size_t rowLength = rowLengthDist(rng);

                        // Update counters
                        remainingConnections -= rowLength;
                        matrixSize -= numPost;

                        return (unsigned int)rowLength;
                    });

    // Insert remaining connections into last row
    rowLengths[numPre - 1] = (unsigned int)remainingConnections;
}

//----------------------------------------------------------------------------
// LiveVisualiser
//----------------------------------------------------------------------------
class LiveVisualiser
{
public:
    LiveVisualiser(GeNNUtils::SharedLibraryModelFloat &model, const cv::Size outputRes, double scale)
    :   m_Model(model), m_OutputImage(outputRes, CV_8UC3), m_RotatedOutput(outputRes.width, outputRes.height, CV_8UC3)/*, t
        m_VideoWriter("test.avi", cv::VideoWriter::fourcc('H', '2', '6', '4'), 33.0, outputRes, true)*/
    {
        const int leftBorder = 50;
        const int bottomBorder = 20;
        const int verticalSpacing = 5;
        const int verticalSpacingLayer = 5;
        const int neuronWidth = (int)std::round((double)(outputRes.width - leftBorder) / scale);

        // Reserve array for populations
        m_Populations.reserve(Parameters::LayerMax * Parameters::PopulationMax);

        cv::Vec3b colours[Parameters::LayerMax * Parameters::PopulationMax] =
        {
            {27,158,119},
            {217,95,2},
            {117,112,179},
            {231,41,138},
            {102,166,30},
            {230,171,2},
            {166,118,29},
            {102,102,102}
        };
        // Loop through populations
        int populationY = 0;
        for(unsigned int layer = 0; layer < Parameters::LayerMax; layer++) {
            for(unsigned int pop = 0; pop < Parameters::PopulationMax; pop++) {
                // Get population name and number of neurons
                const std::string name = Parameters::getPopulationName(layer, pop);
                const int numNeurons = Parameters::getScaledNumNeurons(layer, pop);
                const int neuronHeight = (unsigned int)std::ceil((double)numNeurons / (double)neuronWidth);

                // Get spike count and spikes variables
                unsigned int *spikeCount = m_Model.getArray<unsigned int>("glbSpkCnt" + name);
                unsigned int *spikes = m_Model.getArray<unsigned int>("glbSpk" + name);

                // Create Rectangle of Interest where population activity will be rendered
                cv::Rect roi(leftBorder, populationY, (int)std::round((double)neuronWidth * scale), 
                             (int)std::round((double)neuronHeight * scale));

                const auto textSize = cv::getTextSize(name.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, 1, nullptr);
                cv::putText(m_OutputImage, name.c_str(), cv::Point(0, populationY + (roi.height / 2) + (textSize.height / 2)),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255));

                // Add suitable sized subimage, spike count and spikes to populations
                m_Populations.emplace_back(cv::Mat(neuronHeight, neuronWidth, CV_8UC3),
                                           roi, colours[m_Populations.size()], spikeCount, spikes);

                // Update y position for next population
                populationY += roi.height + verticalSpacing;
            }

            // Add addition inter-layer spacing
            populationY += verticalSpacingLayer;
        }
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void applySpikes()
    {
        // Loop through populations
        for(auto &pop : m_Populations) {
            // Loop through spikes
            cv::Vec3b *spikeImageRaw = reinterpret_cast<cv::Vec3b*>(std::get<0>(pop).data);
            for(unsigned int i = 0; i < std::get<3>(pop)[0]; i++) {
                auto &pixel = spikeImageRaw[std::get<4>(pop)[i]];
                pixel = std::get<2>(pop);
            }
        }
    }

    void render(const char *windowName, bool rotate=false)
    {
        unsigned long long simTimestep = m_Model.getTimestep();
        auto realTime = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> realMs = realTime - m_LastRealTime;
        const double simMs = (double)(simTimestep - m_LastSimTimestep) * 0.1;

        m_LastRealTime = realTime;
        m_LastSimTimestep = simTimestep;

        // Clear background behind text
        cv::rectangle(m_OutputImage, cv::Point(0, m_OutputImage.rows - 20),
                      cv::Point(m_OutputImage.cols, m_OutputImage.rows),
                      CV_RGB(0, 0, 0), CV_FILLED);
        
        // Render status text
        char status[255];
#ifdef JETSON_POWER
        // Read power from device
        std::ifstream powerStream("/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power0_input");
        unsigned int power;
        powerStream >> power;
        sprintf(status, "Power:%.1fW, Speed:%.2fx realtime", (float)power / 1000.0f, simMs / realMs.count());
#else
        sprintf(status, "Speed:%.2fx realtime", simMs / realMs.count());
#endif  // JETSON_POWER
        
        cv::putText(m_OutputImage, status, cv::Point(0, m_OutputImage.rows - 5),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255));

        // Decay each populations spike image
        for(auto &pop : m_Populations) {
            cv::Mat &spikeImage = std::get<0>(pop);
            cv::Vec3b *spikeImageRaw = reinterpret_cast<cv::Vec3b*>(std::get<0>(pop).data);

            // Use fixed point maths to decay each pixel
            std::transform(&spikeImageRaw[0], &spikeImageRaw[spikeImage.rows * spikeImage.cols], &spikeImageRaw[0],
                           [](const cv::Vec3b &pixel)
                           {
                               const uint16_t r = ((uint16_t)pixel[0] * 252) >> 8;
                               const uint16_t g = ((uint16_t)pixel[1] * 252) >> 8;
                               const uint16_t b = ((uint16_t)pixel[2] * 252) >> 8;

                               return cv::Vec3b(r, g, b);
                           });

            // Scale up spike image into output image ROI
            auto roi = cv::Mat(m_OutputImage, std::get<1>(pop));
            cv::resize(spikeImage, roi, roi.size(), 0.0, 0.0, cv::INTER_NEAREST);

        }
        
        
        if(rotate) {
            cv::transpose(m_OutputImage, m_RotatedOutput);
            cv::flip(m_RotatedOutput, m_RotatedOutput, 0);
            cv::imshow(windowName, m_RotatedOutput);
        }
        // Otherwise, Render output image directly to window`
        else {
            cv::imshow(windowName, m_OutputImage);
        }
        
        // Write frame
        //m_VideoWriter.write(m_OutputImage);

    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    GeNNUtils::SharedLibraryModelFloat &m_Model;
    cv::Mat m_OutputImage;
    cv::Mat m_RotatedOutput;
    //cv::VideoWriter m_VideoWriter;
    
    // Times used for tracking real vs simulated time
    std::chrono::time_point<std::chrono::high_resolution_clock> m_LastRealTime;
    unsigned long long m_LastSimTimestep;

    // Array of population spike variables to render
    std::vector<std::tuple<cv::Mat, cv::Rect, cv::Vec3b, unsigned int*, unsigned int*>> m_Populations;
};

void displayThreadHandler(LiveVisualiser &visualiser, std::mutex &mutex, std::atomic<bool> &run)
{
    cv::namedWindow("Output", CV_WINDOW_NORMAL);
    cv::resizeWindow("Output", 480, 800);

    bool rotated = false;
    while(true) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            visualiser.render("Output", rotated);
        }

        const auto key = cv::waitKey(33);
        if(key == 'f') {
            const auto currentFullscreen = cv::getWindowProperty("Output", cv::WND_PROP_FULLSCREEN);
            cv::setWindowProperty("Output", cv::WND_PROP_FULLSCREEN, 
                                  (currentFullscreen == cv::WINDOW_NORMAL) ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
        }
        else if(key == 'r') {
            rotated = !rotated;
            if(rotated) {
                cv::resizeWindow("Output", 800, 480);
            }
            else {
                cv::resizeWindow("Output", 480, 800);
            }
        }
        else if(key == 27) {
            break;
        }
    }
    
    // Clear run flag
    run = false;
}

int main()
{
    GeNNUtils::SharedLibraryModelFloat model("", "potjans_microcircuit");
    {
        Timer<> timer("Allocation:");

        model.allocateMem();
    }
    {
        Timer<> timer("Initialization:");
        model.initialize();
    }

    {
        Timer<> timer("Building connectivity:");
        std::mt19937 rng;
        for(unsigned int trgLayer = 0; trgLayer < Parameters::LayerMax; trgLayer++) {
            for(unsigned int trgPop = 0; trgPop < Parameters::PopulationMax; trgPop++) {
                const std::string trgName = Parameters::getPopulationName(trgLayer, trgPop);
                const unsigned int numTrg = Parameters::getScaledNumNeurons(trgLayer, trgPop);

                // Loop through source populations and layers
                for(unsigned int srcLayer = 0; srcLayer < Parameters::LayerMax; srcLayer++) {
                    for(unsigned int srcPop = 0; srcPop < Parameters::PopulationMax; srcPop++) {
                        const std::string srcName = Parameters::getPopulationName(srcLayer, srcPop);
                        const unsigned int numSrc = Parameters::getScaledNumNeurons(srcLayer, srcPop);

                        const unsigned int numConnections = Parameters::getScaledNumConnections(srcLayer, srcPop, trgLayer, trgPop);
                        if(numConnections > 0) {
                            const std::string synName = srcName + "_" + trgName;
                            model.allocateExtraGlobalParam(synName, "preCalcRowLength", numSrc);

                            unsigned int *preCalcRowLength = model.getArray<unsigned int>("preCalcRowLength" + synName);
                            buildRowLengths(numSrc, numTrg,  numConnections,
                                            preCalcRowLength, rng);                                                                                          \
                            model.pushExtraGlobalParam(synName, "preCalcRowLength", numSrc);
                        }

                    }
                }
            }

        }
    }

    // Final setup
    {
        Timer<> timer("Sparse init:");
        model.initializeSparse();
    }

    std::atomic<bool> run{true};
    std::mutex mutex;
    LiveVisualiser visualiser(model, cv::Size(480, 800), 2.75);
    std::thread displayThread(displayThreadHandler, std::ref(visualiser), std::ref(mutex), std::ref(run));

    
    double applyMs = 0.0;
    {
        Timer<> timer("Simulation:");
        // Loop through timesteps
        while(run)
        {
            // Simulate
            model.stepTime();

            // Pull current spikes from all populations
            for(unsigned int layer = 0; layer < Parameters::LayerMax; layer++) {
                for(unsigned int pop = 0; pop < Parameters::PopulationMax; pop++) {
                    model.pullCurrentSpikesFromDevice(Parameters::getPopulationName(layer, pop));
                }
            }

            {
                TimerAccumulate<> timer(applyMs);

                {
                    std::lock_guard<std::mutex> lock(mutex);
                    visualiser.applySpikes();
                }
            }
        }
    }

    std::cout << "Apply:" << applyMs << "ms" << std::endl;

    return 0;
}
