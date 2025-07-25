// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>

// Standard C includes
#include <cassert>
#include <csignal>
#include <cstdlib>

// OpenCV includes
#include <opencv2/opencv.hpp>

// GeNN includes
#include "modelSpec.h"

// Model includes
#include "parameters.h"

class DVSModel : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(DVSModel);
    SET_THRESHOLD_CONDITION_CODE("spikeVector[id / 32] & (1 << (id % 32))");
    SET_EXTRA_GLOBAL_PARAMS( {{"spikeVector", "uint32_t*"}} );
};
IMPLEMENT_SNIPPET(DVSModel);

class CentreToMacroSnippet : public InitSparseConnectivitySnippet::Base
{
    DECLARE_SNIPPET(CentreToMacroSnippet);

    SET_PARAMS({{"inputSize", "unsigned int"}, {"centreSize", "unsigned int"}, 
                {"kernelSize", "unsigned int"}, {"macroPixelSize", "unsigned int"}});
    
    SET_ROW_BUILD_CODE(
        "const unsigned int nearBorder = (inputSize - centreSize) / 2;\n"
        "const unsigned int farBorder = nearBorder + centreSize;\n"
        "// If we're in the centre\n"
        "const unsigned int xi = id_pre % inputSize;\n"
        "const unsigned int yi = id_pre / inputSize;\n"
        "if(xi >= nearBorder && xi < farBorder && yi >= nearBorder && yi < farBorder) {\n"
        "    const unsigned int yj = (yi - nearBorder) / kernelSize;\n"
        "    const unsigned int xj = (xi - nearBorder) / kernelSize;\n"
        "    addSynapse(xj + (yj * macroPixelSize));\n"
        "}\n");
    SET_MAX_ROW_LENGTH(1);
};
IMPLEMENT_SNIPPET(CentreToMacroSnippet);

class MacroToExcitatory : public InitSparseConnectivitySnippet::Base
{
    DECLARE_SNIPPET(MacroToExcitatory);

    SET_PARAMS({{"macroPixelSize", "unsigned int"}, {"detectorSize", "unsigned int"}});
    SET_DERIVED_PARAMS({{"detectorLeft", [](const ParamValues&, double){ return Parameters::DetectorLeft; }},
                        {"detectorRight", [](const ParamValues&, double){ return Parameters::DetectorRight; }},
                        {"detectorUp", [](const ParamValues&, double){ return Parameters::DetectorUp; }},
                        {"detectorDown", [](const ParamValues&, double){ return Parameters::DetectorDown; }},
                        {"detectorMax", [](const ParamValues&, double){ return Parameters::DetectorMax; }}});

    SET_ROW_BUILD_CODE(
        "const unsigned int xi = id_pre % macroPixelSize;\n"
        "const unsigned int yi = id_pre / macroPixelSize;\n"
        "if(xi >= 1 && xi < (macroPixelSize - 1)\n"
        "    && yi >= 1 && yi < (macroPixelSize - 1))\n"
        "{\n"
        "    const unsigned int xj = (xi - 1) * detectorMax;\n"
        "    const unsigned int yj = yi - 1;\n"
        "    // Add excitatory synapses to all detectors\n"
        "    addSynapse(xj + detectorLeft + (yj * detectorSize * detectorMax));\n"
        "    addSynapse(xj + detectorRight + (yj * detectorSize * detectorMax));\n"
        "    addSynapse(xj + detectorUp + (yj * detectorSize * detectorMax));\n"
        "    addSynapse(xj + detectorDown + (yj * detectorSize * detectorMax));\n"
        "}\n");
    SET_MAX_ROW_LENGTH(Parameters::DetectorMax);
};
IMPLEMENT_SNIPPET(MacroToExcitatory);

class MacroToInhibitory : public InitSparseConnectivitySnippet::Base
{
    DECLARE_SNIPPET(MacroToInhibitory);

    SET_PARAMS({{"macroPixelSize", "unsigned int"}, {"detectorSize", "unsigned int"}});
    SET_DERIVED_PARAMS({{"detectorLeft", [](const ParamValues&, double){ return Parameters::DetectorLeft; }},
                        {"detectorRight", [](const ParamValues&, double){ return Parameters::DetectorRight; }},
                        {"detectorUp", [](const ParamValues&, double){ return Parameters::DetectorUp; }},
                        {"detectorDown", [](const ParamValues&, double){ return Parameters::DetectorDown; }},
                        {"detectorMax", [](const ParamValues&, double){ return Parameters::DetectorMax; }}});

    SET_ROW_BUILD_CODE(
        "const unsigned int xi = id_pre % macroPixelSize;\n"
        "const unsigned int yi = id_pre / macroPixelSize;\n"
        "// Create inhibitory connection to 'left' detector associated with macropixel one to right\n"
        "if(xi < (macroPixelSize - 2)\n"
        "    && yi >= 1 && yi < (macroPixelSize - 1))\n"
        "{\n"
        "    const unsigned int xj = (xi - 1 + 1) * detectorMax;\n"
        "    const unsigned int yj = yi - 1;\n"
        "    addSynapse(xj + detectorLeft + (yj * detectorSize * detectorMax));\n"
        "}\n"
        "// Create inhibitory connection to 'right' detector associated with macropixel one to right\n"
        "if(xi >= 2\n"
        "    && yi >= 1 && yi < (macroPixelSize - 1))\n"
        "{\n"
        "    const unsigned int xj = (xi - 1 - 1) * detectorMax;\n"
        "    const unsigned int yj = yi - 1;\n"
        "    addSynapse(xj + detectorRight + (yj * detectorSize * detectorMax));\n"
        "}\n"
        "// Create inhibitory connection to 'up' detector associated with macropixel one below\n"
        "if(xi >= 1 && xi < (macroPixelSize - 1)\n"
        "    && yi < (macroPixelSize - 2))\n"
        "{\n"
        "    const unsigned int xj = (xi - 1) * detectorMax;\n"
        "    const unsigned int yj = yi - 1 + 1;\n"
        "    addSynapse(xj + detectorUp + (yj * detectorSize * detectorMax));\n"
        "}\n"
        "// Create inhibitory connection to 'down' detector associated with macropixel one above\n"
        "if(xi >= 1 && xi < (macroPixelSize - 1)\n"
        "    && yi >= 2)\n"
        "{\n"
        "    const unsigned int xj = (xi - 1) * detectorMax;\n"
        "    const unsigned int yj = yi - 1 - 1;\n"
        "    addSynapse(xj + detectorDown + (yj * detectorSize * detectorMax));\n"
        "}\n");
    SET_MAX_ROW_LENGTH(Parameters::DetectorMax);
};
IMPLEMENT_SNIPPET(MacroToInhibitory);

// Anonymous namespace
namespace
{
volatile std::sig_atomic_t g_SignalStatus;

void signalHandler(int status)
{
    g_SignalStatus = status;
}

void displayThreadHandler(std::mutex &inputMutex, const cv::Mat &inputImage,
                          std::mutex &outputMutex, const float (&output)[Parameters::detectorSize][Parameters::detectorSize][2])
{
    cv::namedWindow("Input", cv::WINDOW_NORMAL);
    cv::resizeWindow("Input", Parameters::inputSize * Parameters::inputScale,
                     Parameters::inputSize * Parameters::inputScale);

    // Create output image
    const unsigned int outputImageSize = Parameters::detectorSize * Parameters::outputScale;
    cv::Mat outputImage(outputImageSize, outputImageSize, CV_8UC3);

#ifdef JETSON_POWER
    std::ifstream powerStream("/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device/in_power0_input");
    std::ifstream gpuPowerStream("/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device/in_power1_input");
    std::ifstream cpuPowerStream("/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device/in_power2_input");
#endif  // JETSON_POWER

    while(g_SignalStatus == 0)
    {
        // Clear background
        outputImage.setTo(cv::Scalar::all(0));

        {
            std::lock_guard<std::mutex> lock(outputMutex);

            // Loop through output coordinates
            for(unsigned int x = 0; x < Parameters::detectorSize; x++)
            {
                for(unsigned int y = 0; y < Parameters::detectorSize; y++)
                {
                    const cv::Point start(x * Parameters::outputScale, y * Parameters::outputScale);
                    const cv::Point end = start + cv::Point(Parameters::outputVectorScale * output[x][y][0],
                                                            Parameters::outputVectorScale * output[x][y][1]);

                    cv::line(outputImage, start, end,
                             CV_RGB(0xFF, 0xFF, 0xFF));
                }
            }
        }


#ifdef JETSON_POWER
        // Read all power measurements
        unsigned int power, cpuPower, gpuPower;
        powerStream >> power;
        cpuPowerStream >> cpuPower;
        gpuPowerStream >> gpuPower;

        // Clear all stream flags (EOF gets set)
        powerStream.clear();
        cpuPowerStream.clear();
        gpuPowerStream.clear();

        char power[255];
        sprintf(power, "Power:%umW, GPU power:%umW", power, gpuPower);
        cv::putText(outputImage, power, cv::Point(0, outputImageSize - 20),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 0xFF));
        sprintf(power, "CPU power:%umW", cpuPower);
        cv::putText(outputImage, power, cv::Point(0, outputImageSize - 5),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 0xFF));
#endif

        cv::imshow("Output", outputImage);

        {
            std::lock_guard<std::mutex> lock(inputMutex);
            cv::imshow("Input", inputImage);
        }


        cv::waitKey(33);
    }
}

void applyOutputSpikes(const std::vector<unsigned int> &spikeIDs,
                       float (&output)[Parameters::detectorSize][Parameters::detectorSize][2])
{
    // Loop through output spikes
    for(auto spike : spikeIDs) {
        // Convert spike ID to x, y, detector
        const auto spikeCoord = std::div((int)spike, (int)Parameters::detectorSize * Parameters::DetectorMax);
        const int spikeY = spikeCoord.quot;
        const auto xCoord = std::div(spikeCoord.rem, (int)Parameters::DetectorMax);
        const int spikeX =  xCoord.quot;

        // Apply spike to correct axis of output pixel based on detector it was emitted by
        switch(xCoord.rem)
        {
            case Parameters::DetectorLeft:
                output[spikeX][spikeY][0] -= 1.0f;
                break;

            case Parameters::DetectorRight:
                output[spikeX][spikeY][0] += 1.0f;
                break;

            case Parameters::DetectorUp:
                output[spikeX][spikeY][1] -= 1.0f;
                break;

            case Parameters::DetectorDown:
                output[spikeX][spikeY][1] += 1.0f;
                break;

        }
    }

    // Decay output
    for(unsigned int x = 0; x < Parameters::detectorSize; x++)
    {
        for(unsigned int y = 0; y < Parameters::detectorSize; y++)
        {
            output[x][y][0] *= Parameters::flowPersistence;
            output[x][y][1] *= Parameters::flowPersistence;
        }
    }
}
}

void modelDefinition(ModelSpec &model)
{
    model.setDT(Parameters::timestep);
    model.setName("optical_flow");

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters for P population
    ParamValues lifParams{
        {"C", 1.0},
        {"TauM", 20.0},
        {"Vrest", -60.0},
        {"Vreset", -60.0},
        {"Vthresh", -50.0},
        {"Ioffset", 0.0},
        {"TauRefrac", 1.0}};

    // LIF initial conditions
    VarValues lifInit{
        {"V", -60.0},
        {"RefracTime", 0.0}};

    ParamValues dvsMacroPixelWeightUpdateInit{
        {"g", 0.8}};

    ParamValues macroPixelOutputExcitatoryWeightUpdateInit{
        {"g", 1.0}};

    ParamValues macroPixelOutputInhibitoryWeightUpdateInit{
        {"g", -0.5}};

    // Exponential current parameters
    ParamValues macroPixelPostSynParams{
        {"tau", 5.0}};

    ParamValues outputExcitatoryPostSynParams{
        {"tau", 25.0}};

    ParamValues outputInhibitoryPostSynParams{
        {"tau", 50.0}};
        
    ParamValues dvsMacroPixelConnectInit{
        {"inputSize", Parameters::inputSize}, 
        {"centreSize", Parameters::centreSize}, 
        {"kernelSize", Parameters::kernelSize}, 
        {"macroPixelSize", Parameters::macroPixelSize}};
    
    ParamValues macroConnectInit{
        {"macroPixelSize", Parameters::macroPixelSize}, 
        {"detectorSize", Parameters::detectorSize}};

    //------------------------------------------------------------------------
    // Neuron populations
    //------------------------------------------------------------------------
    // Create IF_curr neuron
    auto *dvs = model.addNeuronPopulation<DVSModel>("DVS", Parameters::inputSize * Parameters::inputSize);
    auto *macroPixel = model.addNeuronPopulation<NeuronModels::LIF>("MacroPixel", Parameters::macroPixelSize * Parameters::macroPixelSize,
                                                                    lifParams, lifInit);

    auto *output = model.addNeuronPopulation<NeuronModels::LIF>("Output", Parameters::detectorSize * Parameters::detectorSize * Parameters::DetectorMax,
                                                                lifParams, lifInit);
    output->setSpikeRecordingEnabled(true);

    //------------------------------------------------------------------------
    // Synapse populations
    //------------------------------------------------------------------------
    model.addSynapsePopulation(
        "DVS_MacroPixel", SynapseMatrixType::SPARSE,
        dvs, macroPixel,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(dvsMacroPixelWeightUpdateInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(macroPixelPostSynParams),
        initConnectivity<CentreToMacroSnippet>(dvsMacroPixelConnectInit));

    model.addSynapsePopulation(
        "MacroPixel_Output_Excitatory", SynapseMatrixType::SPARSE,
        macroPixel, output,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(macroPixelOutputExcitatoryWeightUpdateInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(outputExcitatoryPostSynParams),
        initConnectivity<MacroToExcitatory>(macroConnectInit));

    model.addSynapsePopulation(
        "MacroPixel_Output_Inhibitory", SynapseMatrixType::SPARSE,
        macroPixel, output,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(macroPixelOutputInhibitoryWeightUpdateInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(outputInhibitoryPostSynParams),
        initConnectivity<MacroToInhibitory>(macroConnectInit));

    // Use zero-copy for input and output spikes as we want to access them every timestep
    //dvs->setSpikeZeroCopyEnabled(true);
    //output->setSpikeZeroCopyEnabled(true);
}

void simulate(const ModelSpec &model, Runtime::Runtime &runtime)
{
    constexpr unsigned int timestepWords = ((Parameters::inputSize * Parameters::inputSize) + 31) / 32;

    // Lookup populations
    auto *dvs = model.findNeuronGroup("DVS");
    auto *output = model.findNeuronGroup("Output");

    runtime.allocate(1);
    runtime.allocateArray(*dvs, "spikeVector", timestepWords);
    runtime.initialize();
    runtime.initializeSparse();

    // Lookup arrays
    auto *spikeVectorDVS = runtime.getArray(*dvs, "spikeVector");
    auto *spikeVectorDVSPtr = spikeVectorDVS->getHostPointer<uint32_t>();

    // Create DAVIS device
    auto dvsDevice = DVS::create<libcaer::devices::davis>();
    dvsDevice.start();

    const DVS::CropRect dvsCropRect{43, 0, 303, 260};
    
    double dvsGet = 0.0;
    double step = 0.0;
    double render = 0.0;

    std::mutex inputMutex;
    cv::Mat inputImage(Parameters::inputSize, Parameters::inputSize, CV_32F);

    std::mutex outputMutex;
    float outputData[Parameters::detectorSize][Parameters::detectorSize][2] = {0};
    std::thread displayThread(displayThreadHandler,
                              std::ref(inputMutex), std::ref(inputImage),
                              std::ref(outputMutex), std::ref(outputData));

    // Convert timestep to a duration
    const auto dtDuration = std::chrono::duration<double, std::milli>{Parameters::timestep};

    // Duration counters
    std::chrono::duration<double> sleepTime{0};
    std::chrono::duration<double> overrunTime{0};
    unsigned int i = 0;

    // Catch interrupt (ctrl-c) signals
    std::signal(SIGINT, signalHandler);

    for(i = 0; g_SignalStatus == 0; i++)
    {
        auto tickStart = std::chrono::high_resolution_clock::now();

        {
            //TimerAccumulate timer(dvsGet);
            spikeVectorDVS->memsetHostPointer(0);
            dvsDevice.readEvents(spikeVectorDVS, DVS::Polarity::ON_ONLY,
                                 1.0f, &dvsCropRect);
            
            // Copy to GPU
            spikeVectorDVS->pushToDevice();
        }

        {
            //TimerAccumulate timer(render);
            std::lock_guard<std::mutex> lock(inputMutex);

            {
                for(unsigned int w = 0; w < timestepWords; w++) {
                    // Get word
                    uint32_t spikeWord = spikeVectorDVSPtr[w];

                    // Calculate neuron id of highest bit of this word
                    unsigned int neuronID = (w * 32) + 31;

                    // While bits remain
                    while(spikeWord != 0) {
                        // Calculate leading zeros
                        const int numLZ = __builtin_clz(spikeWord);

                        // If all bits have now been processed, zero spike word
                        // Otherwise shift past the spike we have found
                        spikeWord = (numLZ == 31) ? 0 : (spikeWord << (numLZ + 1));

                        // Subtract number of leading zeros from neuron ID
                        neuronID -= numLZ;

                        // Write out CSV line
                        const auto spikeCoord = std::div((int)neuronID, (int)Parameters::inputSize);
                        inputImage.at<float>(spikeCoord.quot, spikeCoord.rem) += 1.0f;

                        // New neuron id of the highest bit of this word
                        neuronID--;
                    }
                }

                // Decay image
                inputImage *= Parameters::spikePersistence;
            }
        }

        {
            //TimerAccumulate timer(step);

            // Simulate
            runtime.stepTime();
            runtime.pullRecordingBuffersFromDevice();
        }

        {
            //TimerAccumulate timer(render);
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                auto outputSpikes = runtime.getRecordedSpikes(*output);
                applyOutputSpikes(outputSpikes[0].second, outputData);
            }
        }

        // Get time of tick start
        auto tickEnd = std::chrono::high_resolution_clock::now();

        // If there we're ahead of real-time pause
        auto tickDuration = tickEnd - tickStart;
        if(tickDuration < dtDuration) {
            auto tickSleep = dtDuration - tickDuration;
            sleepTime += tickSleep;
            std::this_thread::sleep_for(tickSleep);
        }
        else {
            overrunTime += (tickDuration - dtDuration);
        }
    }

    // Wait for display thread to die
    displayThread.join();

    std::cout << "Ran for " << i << " " << Parameters::timestep << "ms timesteps, overan for " << overrunTime.count() << "s, slept for " << sleepTime.count() << "s" << std::endl;
    //std::cout << "Average DVS:" << (dvsGet * 1000.0) / i<< "ms, Step:" << (step * 1000.0) / i << "s, Render:" << (render * 1000.0) / i<< std::endl;
}
