// Standard C++ includes
#include <fstream>
#include <list>
#include <mutex>
#include <thread>

// Standard C includes
#include <csignal>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Common includes
#include "../common/dvs.h"

//----------------------------------------------------------------------------
// DVSInput
//----------------------------------------------------------------------------
class DVSInput : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(DVSInput);
    SET_THRESHOLD_CONDITION_CODE("spikeVector[id / 32] & (1 << (id % 32))");
    SET_EXTRA_GLOBAL_PARAMS( {{"spikeVector", "uint32_t*"}} );
};
IMPLEMENT_SNIPPET(DVSInput);

//----------------------------------------------------------------------------
// OutputClassification
//----------------------------------------------------------------------------
class OutputClassification : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(OutputClassification);

    SET_PARAMS({"TauOut"});    // Membrane time constant [ms]

    SET_VARS({{"Y", "scalar"}, {"B", "scalar", VarAccess::READ_ONLY}});

    SET_DERIVED_PARAMS({
        {"Kappa", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("TauOut").cast<double>()); }}});

    SET_SIM_CODE(
        "Y = (Kappa * Y) + Isyn + B;\n");
};
IMPLEMENT_SNIPPET(OutputClassification);

//----------------------------------------------------------------------------
// RecurrentALIF
//----------------------------------------------------------------------------
class RecurrentALIF : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(RecurrentALIF);

    SET_PARAMS({
        "TauM",         // Membrane time constant [ms]
        "TauAdap",      // Adaption time constant [ms]
        "Vthresh",      // Spiking threshold [mV]
        "TauRefrac",    // Refractory time constant [ms]
        "Beta"});       // Scale of adaption [mV]

    SET_VARS({{"V", "scalar"}, {"A", "scalar"}, {"RefracTime", "scalar"}});

    SET_DERIVED_PARAMS({
        {"Alpha", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("TauM").cast<double>()); }},
        {"Rho", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("TauAdap").cast<double>()); }}});

    SET_ADDITIONAL_INPUT_VARS({{"IsynFeedback", "scalar", 0.0}});

    SET_SIM_CODE(
        "V = (Alpha * V) + Isyn;\n"
        "A *= Rho;\n"
        "if (RefracTime > 0.0) {\n"
        "  RefracTime -= dt;\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("RefracTime <= 0.0 && V >= (Vthresh + (Beta * A))");

    SET_RESET_CODE(
        "RefracTime = TauRefrac;\n"
        "V -= Vthresh;\n"
        "A += 1.0;\n");
};
IMPLEMENT_SNIPPET(RecurrentALIF);

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
volatile std::sig_atomic_t g_SignalStatus;

void signalHandler(int status)
{
    g_SignalStatus = status;
}

void loadArray(const std::string &filename, Runtime::ArrayBase *array)
{
    // Open binary file
    std::ifstream input(filename, std::ios::binary );

    // Check length matches array
    input.seekg (0, std::ios::end);
    const auto length = input.tellg();
    input.seekg(0, std::ios::beg);
    assert(size_t{length} == array->getSizeBytes());

    // Read data from file into host pointer
    input.read(reinterpret_cast<char*>(array->getHostPointer()), array->getSizeBytes());
}

void renderSpikeImage(const NeuronGroup &ng, std::list<cv::Mat> &spikeImages, const Runtime::Runtime &runtime,
                      double startTime)
{
     // Get hidden spikes
    const auto spikes = runtime.getRecordedSpikes(ng)[0];

    // If there's not yet 10 images in list, add one
    if(spikeImages.size() < 10) {
        spikeImages.emplace_back(256, 32, CV_8UC3);
    }
    // Otherwise splice oldest image back to end
    else {
        spikeImages.splice(spikeImages.end(), spikeImages, spikeImages.begin());
    }

    // Clear newest spike image
    spikeImages.back().setTo(cv::Scalar::all(255));

    // Loop through spikes and set pixels in spike image
    for(size_t i = 0; i < spikes.first.size(); i++) {
        spikeImages.back().at<cv::Vec3b>(spikes.second[i], (int)(spikes.first[i] - startTime)) = cv::Vec3b(0, 0, 0);
    }
}

void displayThreadHandler(std::mutex &inputMutex, const cv::Mat &inputImage,
                          std::mutex &outputMutex, const float (&output)[11],
                          std::mutex &hiddenSpikeMutex, const std::list<cv::Mat> &hidden1SpikeImages, const std::list<cv::Mat> &hidden2SpikeImages)
{
    cv::namedWindow("Input", cv::WINDOW_NORMAL);
    cv::resizeWindow("Input", 32 * 10,
                     32 * 10);

    // Load background
    cv::Mat outputImage = cv::imread("background.png");

    // Create 8bpp image to copy input images into
    cv::Mat inputImage8(32, 32, CV_8UC3);

#ifdef JETSON_POWER
    std::ifstream powerStream("/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device/in_power0_input");
    std::ifstream gpuPowerStream("/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device/in_power1_input");
    std::ifstream cpuPowerStream("/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device/in_power2_input");
#endif  // JETSON_POWER

    while(g_SignalStatus == 0)
    {
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

        // Draw output bars
        {
            std::lock_guard<std::mutex> lock(outputMutex);
            const size_t maxOutput = std::distance(std::begin(output), std::max_element(std::begin(output), std::end(output)));
            const int barTop = 400;
            const int barHeight = 256;
            const int barWidth = 30;
            for(size_t i = 0; i < 11; i++) {
                // White out whole are of rectangle
                const int x = 1250 + (i * 40);
                cv::rectangle(outputImage, cv::Point(x, barTop), cv::Point(x + barWidth, barTop + barHeight),
                              cv::Scalar::all(255), cv::FILLED);

                // Draw bar
                const int scaledOutput = int(6.4f * std::clamp(output[i], 0.0f, 40.0f));
                const auto colour = (i == maxOutput) ? CV_RGB(0, 255, 0) : CV_RGB(255, 0, 0);
                cv::rectangle(outputImage, cv::Point(x, barTop + barHeight - scaledOutput), cv::Point(x + barWidth, barTop + barHeight),
                            colour, cv::FILLED);
            }
        }

        // Draw hidden spikes
        {
            std::lock_guard<std::mutex> lock(hiddenSpikeMutex);

            {
                size_t i = 0;
                for(const auto &img : hidden1SpikeImages) {
                    cv::Mat roi(outputImage, cv::Rect(454 + (i++ * 32), 400, 32, 256));
                    img.copyTo(roi);
                }
            }

            {
                size_t i = 0;
                for(const auto &img : hidden2SpikeImages) {
                    cv::Mat roi(outputImage, cv::Rect(852 + (i++ * 32), 400, 32, 256));
                    img.copyTo(roi);
                }
            }

        }

        // Convert spike image to 8bpp
        {
            std::lock_guard<std::mutex> lock(inputMutex);
            inputImage.convertTo(inputImage8, CV_8U, 255.0);
        }
        // Resize into ROI
        cv::Mat roiMat(outputImage, cv::Rect(120, 400, 256, 256));
        cv::resize(inputImage8, roiMat, roiMat.size(), 0.0, 0.0, cv::INTER_NEAREST);

        // Render
        cv::imshow("Output", outputImage);
        cv::waitKey(33);
    }
}
}

void modelDefinition(ModelSpec &model)
{
    model.setDT(1.0);
    model.setName("dvs_classifier");
    //model.setTiming(Parameters::measureTiming);
    model.setMergePostsynapticModels(true);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters for P population
    ParamValues hiddenParam{
        {"TauM", 20.0},
        {"TauAdap", 2000.0},
        {"Vthresh", 0.6},
        {"TauRefrac", 5.0},
        {"Beta", 0.0174}};

    // LIF initial conditions
    VarValues hiddenInit{
        {"V", 0.0},
        {"A", 0.0},
        {"RefracTime", 0.0}};

    ParamValues outputParam{
        {"TauOut", 100.0}};

    VarValues outputInit{
        {"Y", 0.0},
        {"B", uninitialisedVar()}};

    VarValues weightInit{
        {"g", uninitialisedVar()}};

    //------------------------------------------------------------------------
    // Neuron populations
    //------------------------------------------------------------------------
    auto *dvs = model.addNeuronPopulation<DVSInput>("DVS", 32 * 32 * 2);
    auto *hidden1 = model.addNeuronPopulation<RecurrentALIF>("Hidden1", 256, hiddenParam, hiddenInit);
    auto *hidden2 = model.addNeuronPopulation<RecurrentALIF>("Hidden2", 256, hiddenParam, hiddenInit);
    auto *output = model.addNeuronPopulation<OutputClassification>("Output", 11, outputParam, outputInit);
    hidden1->setSpikeRecordingEnabled(true);
    hidden2->setSpikeRecordingEnabled(true);

    //------------------------------------------------------------------------
    // Synapse populations
    //------------------------------------------------------------------------
    model.addSynapsePopulation("DVS_Hidden1", SynapseMatrixType::DENSE,
                               dvs, hidden1,
                               initWeightUpdate<WeightUpdateModels::StaticPulse>({}, weightInit),
                               initPostsynaptic<PostsynapticModels::DeltaCurr>());
    model.addSynapsePopulation("Hidden1_Hidden2", SynapseMatrixType::DENSE,
                               hidden1, hidden2,
                               initWeightUpdate<WeightUpdateModels::StaticPulse>({}, weightInit),
                               initPostsynaptic<PostsynapticModels::DeltaCurr>());
    model.addSynapsePopulation("Hidden1_Output", SynapseMatrixType::DENSE,
                               hidden1, output,
                               initWeightUpdate<WeightUpdateModels::StaticPulse>({}, weightInit),
                               initPostsynaptic<PostsynapticModels::DeltaCurr>());
    model.addSynapsePopulation("Hidden2_Hidden2", SynapseMatrixType::DENSE,
                               hidden2, hidden2,
                               initWeightUpdate<WeightUpdateModels::StaticPulse>({}, weightInit),
                               initPostsynaptic<PostsynapticModels::DeltaCurr>());
    model.addSynapsePopulation("Hidden2_Output", SynapseMatrixType::DENSE,
                               hidden2, output,
                               initWeightUpdate<WeightUpdateModels::StaticPulse>({}, weightInit),
                               initPostsynaptic<PostsynapticModels::DeltaCurr>());
}

void simulate(const ModelSpec &model, Runtime::Runtime &runtime)
{
    // Lookup neuron groups
    auto *dvs = model.findNeuronGroup("DVS");
    auto *hidden1 = model.findNeuronGroup("Hidden1");
    auto *hidden2 = model.findNeuronGroup("Hidden2");
    auto *output = model.findNeuronGroup("Output");

    // Lookup synapse groups
    auto *dvsHidden1 = model.findSynapseGroup("DVS_Hidden1");
    auto *hidden1Hidden2 = model.findSynapseGroup("Hidden1_Hidden2");
    auto *hidden1Output = model.findSynapseGroup("Hidden1_Output");
    auto *hidden2Hidden2 = model.findSynapseGroup("Hidden2_Hidden2");
    auto *hidden2Output = model.findSynapseGroup("Hidden2_Output");

    runtime.allocate(32);
    runtime.allocateArray(*dvs, "spikeVector", 64);
    runtime.initialize();

    // Load weights
    // Pop0 = input
    // Pop2 = hidden1
    // Pop3 = hidden2
    // Pop1 = output
    loadArray("checkpoints_0_512_100_dvs_gesture_1234_256_256_False_True_alif_alif_1.0_1.0_1.0_1.0/99-Conn_Pop0_Pop2-g.bin",
              runtime.getArray(*dvsHidden1, "g"));
    loadArray("checkpoints_0_512_100_dvs_gesture_1234_256_256_False_True_alif_alif_1.0_1.0_1.0_1.0/99-Conn_Pop2_Pop3-g.bin",
              runtime.getArray(*hidden1Hidden2, "g"));
    loadArray("checkpoints_0_512_100_dvs_gesture_1234_256_256_False_True_alif_alif_1.0_1.0_1.0_1.0/99-Conn_Pop2_Pop1-g.bin",
              runtime.getArray(*hidden1Output, "g"));
    loadArray("checkpoints_0_512_100_dvs_gesture_1234_256_256_False_True_alif_alif_1.0_1.0_1.0_1.0/99-Conn_Pop3_Pop3-g.bin",
              runtime.getArray(*hidden2Hidden2, "g"));
    loadArray("checkpoints_0_512_100_dvs_gesture_1234_256_256_False_True_alif_alif_1.0_1.0_1.0_1.0/99-Conn_Pop3_Pop1-g.bin",
              runtime.getArray(*hidden2Output, "g"));

    // Load bias
    loadArray("checkpoints_0_512_100_dvs_gesture_1234_256_256_False_True_alif_alif_1.0_1.0_1.0_1.0/99-Pop1-Bias.bin",
              runtime.getArray(*output, "B"));
    runtime.initializeSparse();

    // Lookup arrays
    auto *spikeVectorDVS = runtime.getArray(*dvs, "spikeVector");
    auto *spikeVectorDVSPtr = spikeVectorDVS->getHostPointer<uint32_t>();
    auto *outputY = runtime.getArray(*output, "Y");
    auto *outputYPtr = outputY->getHostPointer<float>();

    // Create DAVIS device, only looking at central 256*256 pixels
    using Filter = DVS::ROIFilter<45, 301, 2, 258>;

    // Convert timestep to a duration
    const auto dtDuration = std::chrono::duration<double, std::milli>{1.0};

    // Subtract ROI offset and shift right to put each coordinate in [0,32)
    using TransformX = DVS::CombineTransform<DVS::Subtract<45>, DVS::ShiftRight<3>>;
    using TransformY = DVS::CombineTransform<DVS::Subtract<2>, DVS::ShiftRight<3>>;
    DVS::Davis dvsDevice;
    dvsDevice.start();

    std::mutex inputMutex;
    cv::Mat inputImage(32, 32, CV_32FC3);

    // Create circular buffer of 10 spike images
    std::mutex hiddenSpikeMutex;
    std::list<cv::Mat> hidden1SpikeImages;
    std::list<cv::Mat> hidden2SpikeImages;


    std::mutex outputMutex;
    float outputData[11];
    std::thread displayThread(displayThreadHandler,
                              std::ref(inputMutex), std::cref(inputImage),
                              std::ref(outputMutex), std::cref(outputData),
                              std::ref(hiddenSpikeMutex), std::cref(hidden1SpikeImages), std::cref(hidden2SpikeImages));

    // Catch interrupt (ctrl-c) signals
    std::signal(SIGINT, signalHandler);

     // Duration counters
    std::chrono::duration<double> sleepTime{0};
    std::chrono::duration<double> overrunTime{0};
    unsigned int i = 0;
    for(i = 0; g_SignalStatus == 0; i++) {
        auto tickStart = std::chrono::high_resolution_clock::now();

        {
            //TimerAccumulate timer(dvsGet);
            spikeVectorDVS->memsetHostPointer(0);
            dvsDevice.readEventsHist<32, 2, Filter, TransformX, TransformY, true>(spikeVectorDVSPtr);

            // Copy to GPU
            spikeVectorDVS->pushToDevice();
        }

        {
            //TimerAccumulate timer(render);
            std::lock_guard<std::mutex> lock(inputMutex);

            {
                for(unsigned int w = 0; w < 64; w++) {
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

                        // Convert to x, y, p
                        const unsigned int neuronX = (neuronID / 2) % 32;
                        const unsigned int neuronY = (neuronID / 2) / 32;
                        const unsigned int neuronPolarity = neuronID % 2;

                        // Add to pixel
                        inputImage.at<cv::Vec3f>(neuronY, neuronX) += cv::Vec3f(0.0f, 1.0f - neuronPolarity, (float)neuronPolarity);

                        // New neuron id of the highest bit of this word
                        neuronID--;
                    }
                }

                // Decay image
                inputImage *= 0.97f;
            }
        }

        {
            //TimerAccumulate timer(step);

            // Simulate
            runtime.stepTime();

            outputY->pullFromDevice();

            // Every 32 timesteps (roughly one frame)
            if(i != 0 && (i % 32) == 0) {
                // Pull recording buffers from device
                runtime.pullRecordingBuffersFromDevice();

                // Lock mutex
                std::lock_guard<std::mutex> lock(hiddenSpikeMutex);

                // Render spike images
                const double startTime = i - 32.0;
                renderSpikeImage(*hidden1, hidden1SpikeImages, runtime, startTime);
                renderSpikeImage(*hidden2, hidden2SpikeImages, runtime, startTime);
            }
        }

        {
            //TimerAccumulate timer(render);
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                std::copy_n(outputYPtr, 11, outputData);
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

    std::cout << "Ran for " << i << " 1ms timesteps, overan for " << overrunTime.count() << "s, slept for " << sleepTime.count() << "s" << std::endl;

}
