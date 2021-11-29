// Standard C++ includes
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <sstream>
#include <thread>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "superspike_demo_CODE/definitions.h"

namespace
{
inline int _clz(unsigned int value)
{
#ifdef _WIN32
    unsigned long leadingZero = 0;
    if(_BitScanReverse(&leadingZero, value)) {
        return 31 - leadingZero;
    }
    else {
        return 32;
    }
#else
    return __builtin_clz(value);
#endif
}

void loadTargetSpikes(const std::string &filename)
{
    // Open ras file
    std::ifstream rasFile(filename);
    if(!rasFile.good()) {
        throw std::runtime_error("Cannot open ras file: " + filename);
    }

    // Read lines into strings
    std::vector<std::pair<unsigned int, double>> data;
    std::string lineString;
    while(std::getline(rasFile, lineString)) {
        // Wrap line in stream for easier parsing
        std::istringstream lineStream(lineString);

        // Add new pair to vector and read line into it
        data.emplace_back();
        lineStream >> data.back().second;
        lineStream >> data.back().first;

        // Make neuron indices zero-based and convert time to ms
        data.back().first--;
        data.back().second *= 1000.0;
        
        // Check neuron IDs and times are valid
        assert(data.back().first < Parameters::numOutput);
        assert(data.back().second < Parameters::trialMs);
    }

    // Sort data
    // **NOTE** std::pair < operator means this will sort by neuron then time
    std::sort(data.begin(), data.end());

    // Allocate memory for spike times
    allocatespikeTimesOutput(data.size());

    // Copy just the sorted spike times into this memory and push to device
    std::transform(data.cbegin(), data.cend(), &spikeTimesOutput[0],
                   [](const std::pair<unsigned int, double> &s){ return s.second; });
    pushspikeTimesOutputToDevice(data.size());
    
    // Loop through output neurons
    unsigned int spike = 0;
    for(unsigned int i = 0; i < Parameters::numOutput; i++) {
        // Fast-forward until there's a spike from this neuron
        while(spike < data.size() && data[spike].first < i) {
            spike++;
        }

        // Record neurons starting spike index
        startSpikeOutput[i] = spike;
        
        // Fast-forward through all this neuron's spikes
        while(spike < data.size() && data[spike].first == i) {
            spike++;
        }

        // Record neurons ending spike index
        endSpikeOutput[i] = spike;
    }

}

void generateFrozenPoissonInput(std::mt19937 &gen)
{
    std::exponential_distribution<float> dist(1.0);

    // Calcualte inter-spike-interval
    const float isiMs = 1000.0f / Parameters::inputFreqHz;

    // Loop through input neurons
    std::vector<float> spikeTimes;
    for(unsigned int i = 0; i < Parameters::numInput; i++) {
        // Record neurons starting spike index
        startSpikeInput[i] = spikeTimes.size();

        // Generate spike train using exponential distribution
        for(float t = isiMs * dist(gen); t < Parameters::trialMs; t += isiMs * dist(gen)) {
            spikeTimes.push_back(t);
        }

        // Record neurons ending spike index
        endSpikeInput[i] = spikeTimes.size();
    }

    // Allocate memory for spike times
    allocatespikeTimesInput(spikeTimes.size());
    std::copy(spikeTimes.cbegin(), spikeTimes.cend(), &spikeTimesInput[0]);
    pushspikeTimesInputToDevice(spikeTimes.size());
    
    // Push start and end to device
    pushstartSpikeInputToDevice();
    pushendSpikeInputToDevice();
}

float calculateError(unsigned int timestep)
{
    constexpr double a = Parameters::tauDecay / 1000.0;
    constexpr double b = Parameters::tauRise / 1000.0;
    constexpr double c = Parameters::tauAvgErr / 1000.0;
    const double scaleTrErrFlt = 1.0 / (std::pow((a*b)/(a-b),2)*(a/2+b/2-2*(a*b)/(a+b))) / c;

    const double timeS = timestep * Parameters::timestepMs / 1000.0;

    // Calculate mean error
    const float meanError = std::accumulate(&avgSqrErrOutput[0], &avgSqrErrOutput[Parameters::numOutput], 0.0f) / (float)Parameters::numOutput;
    return scaleTrErrFlt * meanError / (1.0 - std::exp(-timeS / c) + 1.0E-9);
}

void writeSpikeImage(cv::Mat &image, const uint32_t *spikes, unsigned int numNeurons,
                     unsigned int neuronScale, unsigned int timestepsPerPixel, 
                     const cv::Vec3b &spikeColour) 
{
    // Reset image to white
    image = cv::Vec3b(255, 255, 255);
    
    // Calculate number of words per-timestep
    const int timestepWords = (numNeurons  + 31) / 32;
    
    // Loop through timesteps
    for(unsigned int t = 0; t < Parameters::trialTimesteps; t++) {    
        // Loop through words representing timestep
        for(int w = 0; w < timestepWords; w++) {
            // Get word
            uint32_t spikeWord = spikes[(t * timestepWords) + w];
            
            // Calculate neuron id of highest bit of this word
            int neuronID = (w * 32) + 31;
            
            // While bits remain
            while(spikeWord != 0) {
                // Calculate leading zeros
                const int numLZ = _clz(spikeWord);
                
                // If all bits have now been processed, zero spike word
                // Otherwise shift past the spike we have found
                spikeWord = (numLZ == 31) ? 0 : (spikeWord << (numLZ + 1));
                
                // Subtract number of leading zeros from neuron ID
                neuronID -= numLZ;
                
                // Set pixel in image
                // **NOTE** we flip so output is right way up
                for(unsigned int i = 0; i < neuronScale; i++) {
                    image.at<cv::Vec3b>(((numNeurons - neuronID - 1) * neuronScale) + i, t / timestepsPerPixel) = spikeColour;
                }
                
                // New neuron id of the highest bit of this word
                neuronID--;
            }
        }
    }
}

void displayThreadHandler(const cv::Mat &outputImage, std::mutex &mutex, std::atomic<bool> &run, double &trialTime)
{
    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::resizeWindow("Output", 1920, 1080);
    
    while(true) {
        {
            std::lock_guard<std::mutex> l(mutex);
            
            // Clear background behind text
            cv::rectangle(outputImage, cv::Point(1304, 880), cv::Point(1920, 990),
                          CV_RGB(255, 255, 255), cv::FILLED);
            
            // Write current power usage to top-right corner
            char status[255];
            sprintf(status, "Trial time:%.1fs (%.1fx real-time)", trialTime, (Parameters::trialMs / 1000.0) / trialTime);
            cv::putText(outputImage, status, cv::Point(1304, 940),
                        cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0x97, 0xA7));
#if TEGRA_CHIP_ID > 0
            // Read power from device
#if TEGRA_CHIP_ID == 24
            std::ifstream powerStream("/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power0_input");
#elif TEGRA_CHIP_ID == 25
            std::ifstream powerStream("/sys/bus/i2c/drivers/ina3221x/7-0040/iio:device0/in_power0_input");
#else
            #error Unsupported Tegra device
#endif
            
            unsigned int power;
            powerStream >> power;
            
            sprintf(status, "Power:%.1fW", (float)power / 1000.0f);
            cv::putText(outputImage, status, cv::Point(1304, 975),
                        cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0x97, 0xA7));
#endif
            // Display image
            cv::imshow("Output", outputImage);
            
        }
        
        const auto key = cv::waitKey(33);
        if(key == 'f') {
            const auto currentFullscreen = cv::getWindowProperty("Output", cv::WND_PROP_FULLSCREEN);
            cv::setWindowProperty("Output", cv::WND_PROP_FULLSCREEN, 
                                  (currentFullscreen == cv::WINDOW_NORMAL) ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
        }
        else if(key == 27) {
            break;
        }
    }
    
    run = false;
}
}   // Anonymous namespace

int main()
{
    try
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        const cv::Vec3b spikeColour(0, 0, 0);
        
        const unsigned int neuronScale = 2;
        const unsigned int timestepsPerPixel = 40;
        
        // Create output image
        //cv::Mat outputImage(1080, 1920, CV_8UC3);
        
        cv::Mat outputImage = cv::imread("background.png");
        assert(outputImage.size().width == 1920 && outputImage.size().height == 1080);
        
        // Create ROIs within background for spikes
        cv::Mat inputSpikeROI(outputImage, cv::Rect(120, 425, Parameters::trialTimesteps / timestepsPerPixel, Parameters::numInput * neuronScale));
        cv::Mat hiddenSpikeROI(outputImage, cv::Rect(712, 425, Parameters::trialTimesteps / timestepsPerPixel, Parameters::numHidden * neuronScale));
        cv::Mat outputSpikeROI(outputImage, cv::Rect(1304, 425, Parameters::trialTimesteps / timestepsPerPixel, Parameters::numOutput * neuronScale));
        
        allocateMem();
        allocateRecordingBuffers(Parameters::trialTimesteps);
        initialize();
        
        // Load target spikes
        loadTargetSpikes("bob_logo-target.ras");
        
        initializeSparse();
        
        // Mutex for exchanging data at end of each trial
        std::mutex outputMutex;
        
        // Atomic flag for controlling main loop from GUI
        std::atomic<bool> run{true};
        
        // Trial time
        // **HACK** this is drawn in the same bit of the screen as the power - which is handled by the display thread
        double trialTime = 0.0;
        
        // Launch display thread
        std::thread displayThread(displayThreadHandler, std::ref(outputImage), std::ref(outputMutex), 
                                  std::ref(run), std::ref(trialTime));
        
        // Repeat demo
        while(run) {
            // Reinitialise weights etc
            initialize();
            
            // Generate frozen Poisson input
            generateFrozenPoissonInput(gen);

            // Calculate initial transpose
            updateCalculateTranspose();

            // Loop through trials
            unsigned int timestep = 0;
            float error = 0.0f;
            r0HiddenOutputWeightOptimiser = Parameters::r0;
            r0InputHiddenWeightOptimiser = Parameters::r0;
            for(unsigned int trial = 0; trial < Parameters::numTrials && run; trial++) {
                // Reduce learning rate every 400 trials
                if(trial != 0 && (trial % 400) == 0) {
                    r0HiddenOutputWeightOptimiser *= 0.1;
                    r0InputHiddenWeightOptimiser *= 0.1;
                }

                // Calculate error
                pullavgSqrErrOutputFromDevice();
                error = calculateError(timestep);
                
                // Get start time of trial
                auto startTime = std::chrono::high_resolution_clock::now();
                
                // Reset model timestep
                // **NOTE** this a bit gross but means we can simplify a lot of logic
                t = 0.0f;
                iT = 0;

                // Loop through timesteps within trial
                for(unsigned int i = 0; i < Parameters::trialTimesteps && run; i++) {
                    stepTime();

                    // If it's time to update weights
                    if(timestep != 0 && (timestep % Parameters::updateTimesteps) == 0) {
                        updateGradientLearn();
                    }

                    timestep++;

                }
                
                // Get end time fo trial
                auto endTime = std::chrono::high_resolution_clock::now();
                
                
                // Reset spike sources by re-uploading starting spike indices
                // **TODO** build repeating spike source array
                pushstartSpikeInputToDevice();
                pushstartSpikeOutputToDevice();

                // Pull recording data and error from device
                pullRecordingBuffersFromDevice();
                
                {
                    // Lock mutex
                    std::lock_guard<std::mutex> l(outputMutex);
                    
                    // Calculate trial time
                    trialTime = std::chrono::duration<double>(endTime - startTime).count();
                    
                    // Write spike images to output image ROI
                    writeSpikeImage(inputSpikeROI, recordSpkInput, Parameters::numInput, neuronScale, timestepsPerPixel, spikeColour);
                    writeSpikeImage(hiddenSpikeROI, recordSpkHidden, Parameters::numHidden, neuronScale, timestepsPerPixel, spikeColour);
                    writeSpikeImage(outputSpikeROI, recordSpkOutput, Parameters::numOutput, neuronScale, timestepsPerPixel, spikeColour);    
                    
                    // Clear background behind text
                    cv::rectangle(outputImage, cv::Point(20, 880), cv::Point(690, 990),
                                  CV_RGB(255, 255, 255), cv::FILLED);
                
                    // Display trial and error in top-right
                    char status[255];
                    sprintf(status, "Trial %u/%u", trial, Parameters::numTrials);
                    cv::putText(outputImage, status, cv::Point(20, 940),
                                cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0x97, 0xA7));
                    sprintf(status, "Error = %.0f", error);
                    cv::putText(outputImage, status, cv::Point(20, 975),
                                cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0x97, 0xA7));
                }

            }
            
            // Free spike times so correct number can be re-allocated next repeat
            freespikeTimesInput();
        }
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
