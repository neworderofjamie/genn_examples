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

// GeNN userproject includes
#include "timer.h"
#include "sharedLibraryModel.h"
#include "spikeRecorder.h"

// Model parameters
#include "parameters.h"
#include "puzzles.h"

//----------------------------------------------------------------------------
// LiveVisualiser
//----------------------------------------------------------------------------
template<size_t S>
class LiveVisualiser
{
public:
    LiveVisualiser(SharedLibraryModel<float> &model, const Puzzle<S> &puzzle, int squareSize)
    :   m_Model(model), m_Puzzle(puzzle), m_OutputImage(S * squareSize, S * squareSize, CV_8UC3), m_SquareSize(squareSize)
    {
        // Loop through populations
        for(size_t x = 0; x < S; x++) {
            for(size_t y = 0; y < S; y++) {
                auto &popSpikes = m_PopulationSpikes[x][y];

                // Zero spike counts
                std::fill(std::get<0>(popSpikes).begin(), std::get<0>(popSpikes).end(), 0);
                
                // Get function pointers for get current spike count function
                std::get<1>(popSpikes) = (GetCurrentSpikeCountFunc)m_Model.getSymbol("get" + Parameters::getPopName(x, y) + "CurrentSpikeCount");
                std::get<2>(popSpikes) = (GetCurrentSpikesFunc)m_Model.getSymbol("get" + Parameters::getPopName(x, y) + "CurrentSpikes");                
            }
        }
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void applySpikes()
    {
        // Loop through populations
        for(size_t x = 0; x < S; x++) {
            for(size_t y = 0; y < S; y++) {
                auto &popSpikes = m_PopulationSpikes[x][y];

                // Get total spike count and spikes from all domains
                const unsigned int numSpikes = std::get<1>(popSpikes)();
                const unsigned int *spikes = std::get<2>(popSpikes)();

                // Loop through spikes
                for(unsigned int i = 0; i < numSpikes; i++) {
                    // Calculate which domain spike is from
                    const unsigned int domain = spikes[i] / Parameters::coreSize;
                 
                    // Increment count
                    std::get<0>(popSpikes)[domain]++;
                }
            }
        }
    }

    void render(const char *windowName)
    {
        unsigned long long simTimestep = m_Model.getTimestep();
        auto realTime = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> realMs = realTime - m_LastRealTime;
        const double simMs = (double)(simTimestep - m_LastSimTimestep) * 0.1;

        m_LastRealTime = realTime;
        m_LastSimTimestep = simTimestep;

        // Clear background
        m_OutputImage.setTo(CV_RGB(0, 0, 0));
        
        // Render status text
        char status[255];
        sprintf(status, "Speed:%.2fx realtime", simMs / realMs.count());
        
        cv::putText(m_OutputImage, status, cv::Point(0, m_OutputImage.rows - 5),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255));

        // Loop through cells
        for(size_t x = 0; x < S; x++) {
            for(size_t y = 0; y < S; y++) {
                auto &popSpikes = m_PopulationSpikes[x][y];

                // Find domain population with most spikes
                auto maxSpikes = std::max_element(std::get<0>(popSpikes).begin(), std::get<0>(popSpikes).end());
                const size_t bestNumber = std::distance(std::get<0>(popSpikes).begin(), maxSpikes);

                // If there is a 
                
                //if(m_Puzzle.solution[y][x] != 0) {
                //}
                const std::string number = std::to_string(bestNumber + 1);
                cv::putText(m_OutputImage, number.c_str(), cv::Point(x * m_SquareSize, y * m_SquareSize),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255));

                // Zero spike counts
                std::fill(std::get<0>(popSpikes).begin(), std::get<0>(popSpikes).end(), 0);
            }
        }

        cv::imshow(windowName, m_OutputImage);
    }

private:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef unsigned int &(*GetCurrentSpikeCountFunc)(void);
    typedef unsigned int *(*GetCurrentSpikesFunc)(void);
    typedef std::tuple<std::array<unsigned int, 10>, GetCurrentSpikeCountFunc, GetCurrentSpikesFunc> PopulationSpikes;

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    SharedLibraryModel<float> &m_Model;
    const Puzzle<S> &m_Puzzle;

    const int m_SquareSize;
    
    cv::Mat m_OutputImage;
    
    // Times used for tracking real vs simulated time
    std::chrono::time_point<std::chrono::high_resolution_clock> m_LastRealTime;
    unsigned long long m_LastSimTimestep;

    // Accumulated spike counts and functions to get current for each population
    PopulationSpikes m_PopulationSpikes[S][S];

};

template<size_t S>
void displayThreadHandler(LiveVisualiser<S> &visualiser, std::mutex &mutex, std::atomic<bool> &run)
{
    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::resizeWindow("Output", 50 * S, 50 * S);

    while(true) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            visualiser.render("Output");
        }

        const auto key = cv::waitKey(33);
        if(key == 27) {
            break;
        }
    }
    
    // Clear run flag
    run = false;
}

int main()
{
    const auto &puzzle = Puzzles::easy;

    SharedLibraryModel<float> model("./", "sudoku");

    model.allocateMem();
    model.initialize();
    model.initializeSparse();


    std::atomic<bool> run{true};
    std::mutex mutex;
    LiveVisualiser<9> visualiser(model, puzzle, 50);
    std::thread displayThread(displayThreadHandler<9>, std::ref(visualiser), std::ref(mutex), std::ref(run));

    
    double applyS = 0.0;
    {
        Timer timer("Simulation:");
        
        // Loop through timesteps
        while(run)
        {
            // Simulate
            model.stepTime();

            // Pull current spikes from all populations
            // **TODO** copyCurrentSpikesFromDevice should be exposed in SLM 
            for(size_t y = 0; y < 9; y++) {
                for(size_t x = 0; x < 9; x++) {
                    model.pullCurrentSpikesFromDevice(Parameters::getPopName(x, y));
                }
            }

            {
                TimerAccumulate timer(applyS);

                {
                    std::lock_guard<std::mutex> lock(mutex);
                    visualiser.applySpikes();
                }
            }
        }
    }

    std::cout << "Apply:" << applyS << "s" << std::endl;

    return 0;
}
