// Standard C++ includes
#include <algorithm>
#include <bitset>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// Standard C includes
#include <cmath>

// OpenCV includes
#include <opencv2/opencv.hpp>

// OpenGL includes
#include <GL/glew.h>
#include <GL/glu.h>

// GLFW
#include <GLFW/glfw3.h>

// CUDA includes
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// GeNN includes
#include "GeNNHelperKrnls.h"

// Common includes
#include "../common/connectors.h"
#include "../common/spike_csv_recorder.h"
#include "../common/timer.h"

// GeNN generated code includes
#include "ant_world_CODE/definitions.h"

// Antworld includes
#include "common.h"
#include "parameters.h"
#include "render_mesh.h"
#include "route.h"
#include "snapshot_processor.h"
#include "world.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
// What colour should the ground be?
constexpr GLfloat groundColour[] = {0.898f, 0.718f, 0.353f};

// What colour should the brightest tussocks be?
constexpr GLfloat worldColour[] = {0.0f, 1.0f, 0.0f};

// How fast does the ant move?
constexpr float antTurnSpeed = 4.0f;
constexpr float antMoveSpeed = 0.05f;

// Constant to multiply degrees by to get radians
constexpr int displayRenderWidth = 296;
constexpr int displayRenderHeight = 76;

constexpr int intermediateSnapshotWidth = 74;
constexpr int intermediateSnapshowHeight = 19;

constexpr unsigned int numNoiseSources = Parameters::numPN + Parameters::numKC + Parameters::numEN;

curandState *d_RNGState = nullptr;
scalar *d_Noise = nullptr;

enum class State
{
    Training,
    Testing,
    RandomWalk,
    SpinningTrain,
    SpinningTest,
    Idle,
};

// Enumeration of keys
enum Key
{
    KeyLeft,
    KeyRight,
    KeyUp,
    KeyDown,
    KeyTrainSnapshot,
    KeyTestSnapshot,
    KeySpin,
    KeyReset,
    KeyMax
};

// Bitset used for passing which keys have been pressed between key callback and render loop
typedef std::bitset<KeyMax> KeyBitset;

//----------------------------------------------------------------------------
void keyCallback(GLFWwindow *window, int key, int, int action, int)
{
    // If action isn't a press or a release, do nothing
    if(action != GLFW_PRESS && action != GLFW_RELEASE) {
        return;
    }

    // Determine what state key bit should be set to
    const bool newKeyState = (action == GLFW_PRESS);

    // Extract key bitset from window's user pointer
    KeyBitset *keybits = (KeyBitset*)glfwGetWindowUserPointer(window);

    // Apply new key state to bits of key bits
    switch(key) {
        case GLFW_KEY_LEFT:
            keybits->set(KeyLeft, newKeyState);
            break;

        case GLFW_KEY_RIGHT:
            keybits->set(KeyRight, newKeyState);
            break;

        case GLFW_KEY_UP:
            keybits->set(KeyUp, newKeyState);
            break;

        case GLFW_KEY_DOWN:
            keybits->set(KeyDown, newKeyState);
            break;

        case GLFW_KEY_SPACE:
            keybits->set(KeyTrainSnapshot, newKeyState);
            break;

        case GLFW_KEY_ENTER:
            keybits->set(KeyTestSnapshot, newKeyState);
            break;

        case GLFW_KEY_R:
            keybits->set(KeyReset, newKeyState);
            break;

        case GLFW_KEY_S:
            keybits->set(KeySpin, newKeyState);
            break;
    }
}
//----------------------------------------------------------------------------
void generateCubeFaceLookAtMatrices(GLfloat (&matrices)[6][16])
{
    // Set matrix model (which matrix stack you trash is somewhat arbitrary)
    glMatrixMode(GL_MODELVIEW);

    // Loop through cube faces
    for(unsigned int f = 0; f < 6; f++) {
        // Load identity matrix
        glLoadIdentity();

        // Load lookup matrix
        switch (f + GL_TEXTURE_CUBE_MAP_POSITIVE_X)
        {
            case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
                gluLookAt(0.0,  0.0,    0.0,
                          1.0,  0.0,    0.0,
                          0.0,  0.0,    1.0);
                break;

            case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
                gluLookAt(0.0,  0.0,    0.0,
                          -1.0, 0.0,    0.0,
                          0.0,  0.0,    1.0);
                break;

            case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
                gluLookAt(0.0,  0.0,    0.0,
                          0.0,  0.0,    -1.0,
                          0.0,  1.0,    0.0);
                break;

            case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
                gluLookAt(0.0,  0.0,    0.0,
                          0.0,  0.0,    1.0,
                          0.0,  -1.0,    0.0);
                break;

            case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
                gluLookAt(0.0,  0.0,    0.0,
                          0.0,  1.0,    0.0,
                          0.0,  0.0,    1.0);
                break;

            case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
                gluLookAt(0.0,  0.0,    0.0,
                          0.0,  -1.0,   0.0,
                          0.0,  0.0,    1.0);
                break;

            default:
                break;
        };

        // Save matrix
        glGetFloatv(GL_MODELVIEW_MATRIX, matrices[f]);
    }
}
//----------------------------------------------------------------------------
void renderAntView(float antX, float antY, float antHeading,
                   const World &world, const RenderMesh &renderMesh,
                   GLuint cubemapFBO, GLuint cubemapTexture, const GLfloat (&cubeFaceLookAtMatrices)[6][16])
{
    // Configure viewport to cubemap-sized square
    glViewport(0, 0, 256, 256);

    // Bind world
    world.bind();

    // Bind the cubemap FBO for offscreen rendering
    glBindFramebuffer(GL_FRAMEBUFFER, cubemapFBO);

    // Configure perspective projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.0,
                   1.0,
                   0.001, 14.0);

    glMatrixMode(GL_MODELVIEW);

    // Save ant transform to matrix
    float antMatrix[16];
    glLoadIdentity();
    glRotatef(antHeading, 0.0f, 0.0f, 1.0f);
    glTranslatef(-antX, -antY, -0.01f);
    glGetFloatv(GL_MODELVIEW_MATRIX, antMatrix);

    // Loop through each heading we need to render
    for(GLenum f = 0; f < 6; f++) {
        // Attach correct frame buffer face to frame buffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, f + GL_TEXTURE_CUBE_MAP_POSITIVE_X, cubemapTexture, 0);

        // Load look at matrix for this cube face
        glLoadMatrixf(cubeFaceLookAtMatrices[f]);

        // Multiply this by ant transform
        glMultMatrixf(antMatrix);

        // Clear colour and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw world
        // **NOTE** buffers were manually bound previously
        world.render(false);
    }

    // Unbind the FBO for onscreen rendering
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Set viewport to strip at stop of window
    glViewport(0, displayRenderWidth + 10,
               displayRenderWidth, displayRenderHeight);

    // Bind cubemap texture
    glEnable(GL_TEXTURE_CUBE_MAP);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0,
               0.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Render render mesh
    renderMesh.render();

    // Disable texture coordinate array, cube map texture and cube map texturing!
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    glDisable(GL_TEXTURE_CUBE_MAP);

}
//----------------------------------------------------------------------------
void renderTopDownView(float antX, float antY, float antHeading,
                       const World &world, const Route &route)
{
    // Set viewport to square at bottom of screen
    glViewport(0, 0, displayRenderWidth, displayRenderWidth);

    // Configure top-down orthographic projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 10.0,
               0.0, 10.0);

    // Build modelview matrix to centre world
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Render world
    world.render();

    // Render route
    route.render(antX, antY, antHeading);

}
//----------------------------------------------------------------------------
unsigned int convertMsToTimesteps(double ms)
{
    return (unsigned int)std::round(ms / Parameters::timestepMs);
}
//----------------------------------------------------------------------------
void initGeNN(std::mt19937 &gen)
{
    {
        Timer<> timer("Allocation:");
        allocateMem();
    }

    {
        Timer<> timer("Initialization:");
        initialize();

        // Null unused external input pointers
        IextKC = nullptr;
        IextEN = nullptr;
    }

    {
        Timer<> timer("Configuring on-device RNG:");

        // Allocate device array to hold RNG state
        CHECK_CUDA_ERRORS(cudaMalloc(&d_RNGState, numNoiseSources * sizeof(curandState)));

        // Initialize RNG state
        xorwow_setup(d_RNGState, numNoiseSources, 123);

        // Allocate device array to hold input noise
        CHECK_CUDA_ERRORS(cudaMalloc(&d_Noise, numNoiseSources * sizeof(scalar)));

        // Point extra neuron variables at correct parts of noise array
        InoisePN = &d_Noise[0];
        InoiseKC = &d_Noise[Parameters::numPN];
        InoiseEN = &d_Noise[Parameters::numPN + Parameters::numKC];
    }

    {
        Timer<> timer("Building connectivity:");

        buildFixedNumberPreConnector(Parameters::numPN, Parameters::numKC,
                                     Parameters::numPNSynapsesPerKC, CpnToKC, &allocatepnToKC, gen);
    }

    // Final setup
    {
        Timer<> timer("Sparse init:");
        initant_world();
    }
}
//----------------------------------------------------------------------------
std::tuple<unsigned int, unsigned int, unsigned int> presentToMB(float *inputData, unsigned int inputDataStep, bool reward)
{
    Timer<> timer("\tSimulation:");

    // Convert simulation regime parameters to timesteps
    const unsigned long long rewardTimestep = iT + convertMsToTimesteps(Parameters::rewardTimeMs);
    const unsigned int presentDuration = convertMsToTimesteps(Parameters::presentDurationMs);
    const unsigned int postStimuliDuration = convertMsToTimesteps(Parameters::postStimuliDurationMs);

    const unsigned int duration = presentDuration + postStimuliDuration;
    const unsigned long long endPresentTimestep = iT + presentDuration;
    const unsigned long long endTimestep = iT + duration;

    // Open CSV output files
#ifdef RECORD_SPIKES
    const float startTimeMs = t;
    SpikeCSVRecorder pnSpikes("pn_spikes.csv", glbSpkCntPN, glbSpkPN);
    SpikeCSVRecorder kcSpikes("kc_spikes.csv", glbSpkCntKC, glbSpkKC);
    SpikeCSVRecorder enSpikes("en_spikes.csv", glbSpkCntEN, glbSpkEN);

    std::bitset<Parameters::numPN> pnSpikeBitset;
    std::bitset<Parameters::numKC> kcSpikeBitset;
#endif  // RECORD_SPIKES

    // Update input data step
    IextStepPN = inputDataStep;

    // Configure threads and grids
    // **YUCK** I have no idea why this isn't in GeNNHelperKrnls
    int sampleBlkNo = ceilf(float(numNoiseSources / float(BlkSz)));
    dim3 sThreads(BlkSz, 1);
    dim3 sGrid(sampleBlkNo, 1);

    // Loop through timesteps
    unsigned int numPNSpikes = 0;
    unsigned int numKCSpikes = 0;
    unsigned int numENSpikes = 0;
    while(iT < endTimestep)
    {
        // Generate normally distributed noise on GPU
        generate_random_gpuInput_xorwow<scalar>(d_RNGState, d_Noise, numNoiseSources,
                                                1.0f, 0.0f,
                                                sGrid, sThreads);
        // If we should be presenting an image
        if(iT < endPresentTimestep) {
            IextPN = inputData;
        }
        // Otherwise update offset to point to block of zeros
        else {
            IextPN = nullptr;
        }

        // If we should reward in this timestep, inject dopamine
        if(reward && iT == rewardTimestep) {
            injectDopaminekcToEN = true;
        }

#ifndef CPU_ONLY
        // Simulate on GPU
        stepTimeGPU();

        // Download spikes
#ifdef RECORD_SPIKES
        pullPNCurrentSpikesFromDevice();
        pullKCCurrentSpikesFromDevice();
        pullENCurrentSpikesFromDevice();
#else
        CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPN, d_glbSpkCntPN, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntKC, d_glbSpkCntKC, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEN, d_glbSpkCntEN, sizeof(unsigned int), cudaMemcpyDeviceToHost));
#endif
#else
        // Simulate on CPU
        stepTimeCPU();
#endif
        // If a dopamine spike has been injected this timestep
        if(injectDopaminekcToEN) {
            // Decay global dopamine traces
            dkcToEN = dkcToEN * std::exp(-(t - tDkcToEN) / Parameters::tauD);

            // Add effect of dopamine spike
            dkcToEN += Parameters::dopamineStrength;

            // Update last reward time
            tDkcToEN = t;

            // Clear dopamine injection flags
            injectDopaminekcToEN = false;
        }

        numPNSpikes += spikeCount_PN;
        numKCSpikes += spikeCount_KC;
        numENSpikes += spikeCount_EN;
#ifdef RECORD_SPIKES
        for(unsigned int i = 0; i < spikeCount_PN; i++) {
            pnSpikeBitset.set(spike_PN[i]);
        }

        for(unsigned int i = 0; i < spikeCount_KC; i++) {
            kcSpikeBitset.set(spike_KC[i]);
        }
        // Record spikes
        pnSpikes.record(t - startTimeMs);
        kcSpikes.record(t - startTimeMs);
        enSpikes.record(t - startTimeMs);
#endif  // RECORD_SPIKES
    }

#ifdef RECORD_TERMINAL_SYNAPSE_STATE
    // Download synaptic state
    pullkcToENStateFromDevice();

    std::ofstream terminalStream("terminal_synaptic_state.csv");
    terminalStream << "Weight, Eligibility" << std::endl;
    for(unsigned int s = 0; s < Parameters::numKC * Parameters::numEN; s++) {
        terminalStream << gkcToEN[s] << "," << ckcToEN[s] * std::exp(-(t - tCkcToEN[s]) / 40.0) << std::endl;
    }
    std::cout << "Final dopamine level:" << dkcToEN * std::exp(-(t - tDkcToEN) / Parameters::tauD) << std::endl;
#endif  // RECORD_TERMINAL_SYNAPSE_STATE

#ifdef RECORD_SPIKES
    std::ofstream activeNeuronStream("active_neurons.csv", std::ios_base::app);
    activeNeuronStream << pnSpikeBitset.count() << "," << kcSpikeBitset.count() << "," << numENSpikes << std::endl;
#endif  // RECORD_SPIKES
    if(reward) {
        constexpr unsigned int numWeights = Parameters::numKC * Parameters::numEN;

        CHECK_CUDA_ERRORS(cudaMemcpy(gkcToEN, d_gkcToEN, numWeights * sizeof(scalar), cudaMemcpyDeviceToHost));

        unsigned int numUsedWeights = std::count(&gkcToEN[0], &gkcToEN[numWeights], 0.0f);
        std::cout << "\t" << numWeights - numUsedWeights << " unused weights" << std::endl;
    }

    return std::make_tuple(numPNSpikes, numKCSpikes, numENSpikes);
}
//----------------------------------------------------------------------------
void handleGLFWError(int errorNumber, const char *message)
{
    std::cerr << "GLFW error number:" << errorNumber << ", message:" << message << std::endl;
}
}   // anonymous namespace
//----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    std::mt19937 gen;

    // Set GLFW error callback
    glfwSetErrorCallback(handleGLFWError);

    // Initialize the library
    if(!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    // Prevent window being resized
    glfwWindowHint(GLFW_RESIZABLE, false);

    // Create a windowed mode window and its OpenGL context
    GLFWwindow *window = glfwCreateWindow(displayRenderWidth, displayRenderHeight + displayRenderWidth + 10,
                                          "Ant World", nullptr, nullptr);
    if(!window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create window");
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if(glewInit() != GLEW_OK) {
        throw std::runtime_error("Failed to initialize GLEW");
    }

    // Enable VSync
    glfwSwapInterval(2);

    // Set clear colour to match matlab and enable depth test
    glClearColor(0.0f, 1.0f, 1.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glLineWidth(4.0);
    glPointSize(4.0);

    // Create key bitset and setc it as window user pointer
    KeyBitset keybits;
    glfwSetWindowUserPointer(window, &keybits);

    // Set key callback
    glfwSetKeyCallback(window, keyCallback);

    // Create route object and load route file specified by command line
    Route route(0.2f, 800);
    if(argc > 1) {
        route.load(argv[1]);
    }

    // Load world into OpenGL
    World world("world5000_gray.bin", worldColour, groundColour);

    // Build mesh to render cubemap to screen
    // **NOTE** this matches the matlab:
    // hfov = hfov/180/2*pi;
    // axis([0 14 -hfov hfov -pi/12 pi/3]);
    RenderMesh renderMesh(296.0f, 75.0f, 15.0f,
                          40, 10);

    // Create FBO for rendering to cubemap and bind
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Create cubemap and bind
    GLuint cubemap;
    glGenTextures(1, &cubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);

    // Create textures for all faces of cubemap
    // **NOTE** even though we don't need top and bottom faces we still need to create them or rendering fails
    for(unsigned int t = 0; t < 6; t++) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + t, 0, GL_RGB,
                     256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    }
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    // Create depth render buffer
    GLuint depthbuff;
    glGenRenderbuffers(1, &depthbuff);
    glBindRenderbuffer(GL_RENDERBUFFER, depthbuff);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 256, 256);

    // Attach depth buffer to frame buffer
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthbuff);

    // Check frame buffer is created correctly
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Frame buffer not complete");
    }

    // Unbind cube map and frame buffer
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Pre-generate lookat matrices to point at cubemap faces
    GLfloat cubeFaceLookAtMatrices[6][16];
    generateCubeFaceLookAtMatrices(cubeFaceLookAtMatrices);

    // Initialize GeNN
    initGeNN(gen);

    // Host OpenCV array to hold pixels read from screen
    cv::Mat snapshot(displayRenderHeight, displayRenderWidth, CV_8UC3);

    // Create snapshot processor to perform image processing on snapshot
    SnapshotProcessor snapshotProcessor(intermediateSnapshotWidth, intermediateSnapshowHeight,
                                        Parameters::inputWidth, Parameters::inputHeight);

    // Initialize ant position
    float antX = 5.0f;
    float antY = 5.0f;
    float antHeading = 270.0f;
    if(route.size() > 0) {
        std::tie(antX, antY, antHeading) = route[0];
    }

    // If a route is loaded, start in training mode, otherwise idle
    State state = (route.size() > 0) ? State::Training : State::Idle;
    //State state = State::RandomWalk;

    unsigned int trainPoint = 0;

    unsigned int testingScan = 0;

    unsigned int numErrors = 0;

    float bestHeading = 0.0f;
    unsigned int bestTestENSpikes = std::numeric_limits<unsigned int>::max();

    // Calculate scan parameters
    constexpr double halfScanAngle = Parameters::scanAngle / 2.0;
    constexpr unsigned int numScanSteps = (unsigned int)round(Parameters::scanAngle / Parameters::scanStep);
    constexpr unsigned int numSpinSteps = (unsigned int)round(Parameters::scanAngle / Parameters::spinStep);

    // When random walking, distribution of angles to turn by
    std::uniform_real_distribution<float> randomAngleOffset(-halfScanAngle, halfScanAngle);

    std::ofstream spin;

    std::future<std::tuple<unsigned int, unsigned int, unsigned int>> gennResult;
    while (!glfwWindowShouldClose(window)) {
        // If there is no valid result (GeNN process has never run), we are ready to take a snapshot
        bool readyForNextSnapshot = false;
        bool resultsAvailable = false;
        unsigned int numPNSpikes;
        unsigned int numKCSpikes;
        unsigned int numENSpikes;
        if(!gennResult.valid()) {
            readyForNextSnapshot = true;
        }
        // Otherwise if GeNN has run and the result is ready for us, s
        else if(gennResult.wait_for(std::chrono::seconds(0)) == future_status::ready) {
            std::tie(numPNSpikes, numKCSpikes, numENSpikes) = gennResult.get();
            std::cout << "\t" << numPNSpikes << " PN spikes, " << numKCSpikes << " KC spikes, " << numENSpikes << " EN spikes" << std::endl;

            readyForNextSnapshot = true;
            resultsAvailable = true;
        }

        // Update heading and ant position based on keys
        bool trainSnapshot = false;
        bool testSnapshot = false;
        if(keybits.test(KeyLeft)) {
            antHeading -= antTurnSpeed;
        }
        if(keybits.test(KeyRight)) {
            antHeading += antTurnSpeed;
        }
        if(keybits.test(KeyUp)) {
            antX += antMoveSpeed * sin(antHeading * degreesToRadians);
            antY += antMoveSpeed * cos(antHeading * degreesToRadians);
        }
        if(keybits.test(KeyDown)) {
            antX -= antMoveSpeed * sin(antHeading * degreesToRadians);
            antY -= antMoveSpeed * cos(antHeading * degreesToRadians);
        }
        if(keybits.test(KeyReset)) {
            if(route.size() > 0) {
                std::tie(antX, antY, antHeading) = route[0];
            }
            else {
                antX = 5.0f;
                antY = 5.0f;
                antHeading = 270.0f;
            }
        }
        if(keybits.test(KeySpin) && state == State::Idle) {
            trainSnapshot = true;
            state = State::SpinningTrain;
        }

        // If GeNN is ready to handle next snapshot, trigger snapshots if keys are pressed
        if(readyForNextSnapshot && keybits.test(KeyTrainSnapshot)) {
            trainSnapshot = true;
        }
        if(readyForNextSnapshot && keybits.test(KeyTestSnapshot)) {
            testSnapshot = true;
        }

        // If we're training
        if(state == State::Training) {
            // If results from previous training snapshot are available, mark them on route
            if(resultsAvailable) {
                route.setWaypointFamiliarity(trainPoint - 1,
                                             (double)numENSpikes / 20.0);
            }

            // If GeNN is free to process next snapshot
            if(readyForNextSnapshot) {
                // If GeNN isn't training and we have more route points to train
                if(trainPoint < route.size()) {
                    // Snap ant to next snapshot point
                    std::tie(antX, antY, antHeading) = route[trainPoint];

                    // Update window title
                    std::string windowTitle = "Ant World - Training snaphot " + std::to_string(trainPoint) + "/" + std::to_string(route.size());
                    glfwSetWindowTitle(window, windowTitle.c_str());

                    // Set flag to train this snapshot
                    trainSnapshot = true;

                    // Go onto next training point
                    trainPoint++;
                }
                // Otherwise, if we've reached end of route
                else {
                    std::cout << "Training complete (" << route.size() << " snapshots)" << std::endl;

                    // Go to testing state
                    state = State::Testing;

                    // Snap ant back to start of route, facing in starting scan direction
                    std::tie(antX, antY, antHeading) = route[0];
                    antHeading -= halfScanAngle;

                    // Add initial replay point to route
                    route.addPoint(antX, antY, false);

                    // Reset scan
                    testingScan = 0;
                    bestTestENSpikes = std::numeric_limits<unsigned int>::max();

                    // Take snapshot
                    testSnapshot = true;
                }
            }
        }
        // Otherwise, if we're testing
        else if(state == State::Testing) {
            if(resultsAvailable) {
                // If this is an improvement on previous best spike count
                if(numENSpikes < bestTestENSpikes) {
                    bestHeading = antHeading;
                    bestTestENSpikes = numENSpikes;

                    std::cout << "\tUpdated result: " << bestHeading << " is most familiar heading with " << bestTestENSpikes << " spikes" << std::endl;
                }

                // Update window title
                std::string windowTitle = "Ant World - Testing with " + std::to_string(numErrors) + " errors";
                glfwSetWindowTitle(window, windowTitle.c_str());

                // Go onto next scan
                testingScan++;

                // If scan isn't complete
                if(testingScan < numScanSteps) {
                    // Scan right
                    antHeading += Parameters::scanStep;

                    // Take test snapshot
                    testSnapshot = true;
                }
                else {
                    std::cout << "Scan complete: " << bestHeading << " is most familiar heading with " << bestTestENSpikes << " spikes" << std::endl;

                    // Snap ant to it's best heading
                    antHeading = bestHeading;

                    // Move ant forward by snapshot distance
                    antX += Parameters::snapshotDistance * sin(antHeading * degreesToRadians);
                    antY += Parameters::snapshotDistance * cos(antHeading * degreesToRadians);

                    // If we've reached destination
                    if(route.atDestination(antX, antY, Parameters::errorDistance)) {
                        std::cout << "Destination reached with " << numErrors << " errors" << std::endl;

                        // Reset state to idle
                        state = State::Idle;

                        // Add final point to route
                        route.addPoint(antX, antY, false);
                    }
                    // Otherwise
                    else {
                        // Calculate distance to route
                        float distanceToRoute;
                        size_t nearestRouteWaypoint;
                        std::tie(distanceToRoute, nearestRouteWaypoint) = route.getDistanceToRoute(antX, antY);
                        std::cout << "\tDistance to route: " << distanceToRoute * 100.0f << "cm" << std::endl;

                        // If we are further away than error threshold
                        if(distanceToRoute > Parameters::errorDistance) {
                            // Snap ant to next snapshot position
                            // **HACK** this is dubious but looks very much like what the original model was doing in figure 1i
                            std::tie(antX, antY, antHeading) = route[nearestRouteWaypoint + 1];

                            // Add error point to route
                            route.addPoint(antX, antY, true);

                            // Increment error counter
                            numErrors++;
                        }
                        // Otherwise add 'correct' point to route
                        else {
                            route.addPoint(antX, antY, false);
                        }

                        // Reset scan
                        antHeading -= halfScanAngle;
                        testingScan = 0;
                        bestTestENSpikes = std::numeric_limits<unsigned int>::max();

                        // Take snapshot
                        testSnapshot = true;
                    }
                }
            }
        }
        else if(state == State::RandomWalk) {
            // Pick random heading
            antHeading += randomAngleOffset(gen);

            // Move ant forward by snapshot distance
            antX += Parameters::snapshotDistance * sin(antHeading * degreesToRadians);
            antY += Parameters::snapshotDistance * cos(antHeading * degreesToRadians);

            // If we've reached destination
            if(route.atDestination(antX, antY, Parameters::errorDistance)) {
                std::cout << "Destination reached with " << numErrors << " errors" << std::endl;

                // Reset state to idle
                state = State::Idle;

                // Add final point to route
                route.addPoint(antX, antY, false);
            }
            // Otherwise
            else {
                // Calculate distance to route
                float distanceToRoute;
                size_t nearestRouteWaypoint;
                std::tie(distanceToRoute, nearestRouteWaypoint) = route.getDistanceToRoute(antX, antY);

                // If we are further away than error threshold
                if(distanceToRoute > Parameters::errorDistance) {
                    // Snap ant to next snapshot position
                    // **HACK** this is dubious but looks very much like what the original model was doing in figure 1i
                    std::tie(antX, antY, antHeading) = route[nearestRouteWaypoint + 1];

                    // Add error point to route
                    route.addPoint(antX, antY, true);

                    // Increment error counter
                    numErrors++;
                }
                // Otherwise add 'correct' point to route
                else {
                    route.addPoint(antX, antY, false);
                }
            }
        }
        if(state == State::SpinningTrain) {
            if(resultsAvailable) {
                spin.open("spin.csv");

                // Start testing scan
                state = State::SpinningTest;
                antHeading -= halfScanAngle;
                testingScan = 0;
                testSnapshot = true;
            }
        }
        else if(state == State::SpinningTest) {
            if(resultsAvailable) {
                   // Write heading and number of spikes to file
                spin << antHeading << "," << numENSpikes << std::endl;

                // Go onto next scan
                testingScan++;

                // If scan isn't complete
                if(testingScan < numSpinSteps) {
                    // Scan right
                    antHeading += Parameters::spinStep;

                    // Take test snapshot
                    testSnapshot = true;
                }
                else {
                    spin.close();

                    state = State::Idle;
                }
            }
        }

        // Clear colour and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Render ant's eye view at top of the screen
        renderAntView(antX, antY, antHeading,
                      world, renderMesh,
                      fbo, cubemap, cubeFaceLookAtMatrices);

        // Render top-down view at bottom of the screen
        renderTopDownView(antX, antY, antHeading,
                          world, route);


        // Swap front and back buffers
        glfwSwapBuffers(window);

        // If we should take a snapshot
        if(trainSnapshot || testSnapshot) {
            Timer<> timer("\tSnapshot generation:");

            std::cout << "Snapshot at (" << antX << "," << antY << "," << antHeading << ")" << std::endl;

            // Read pixels from framebuffer
            // **TODO** it should be theoretically possible to go directly from frame buffer to GpuMat
            glReadPixels(0, displayRenderWidth + 10, displayRenderWidth, displayRenderHeight,
                         GL_BGR, GL_UNSIGNED_BYTE, snapshot.data);

            // Process snapshot
            float *finalSnapshotData;
            unsigned int finalSnapshotStep;
            std::tie(finalSnapshotData, finalSnapshotStep) = snapshotProcessor.process(snapshot);

            // Start simulation, applying reward if we are training
            gennResult = std::async(std::launch::async, presentToMB,
                                    finalSnapshotData, finalSnapshotStep, trainSnapshot);
        }

        // Poll for and process events
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}