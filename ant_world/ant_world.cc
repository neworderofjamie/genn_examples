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
constexpr int displayRenderWidth = 640;
constexpr int displayRenderHeight = 178;

constexpr int intermediateSnapshotWidth = 74;
constexpr int intermediateSnapshowHeight = 19;

enum class State
{
    Training,
    Testing,
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
void initGeNN()
{
    std::mt19937 gen;

    {
        Timer<> timer("Allocation:");
        allocateMem();
    }

    {
        Timer<> timer("Initialization:");
        initialize();
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
unsigned int presentToMB(uint8_t *inputData, unsigned int inputDataStep, bool reward)
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
    SpikeCSVRecorder pnSpikes("pn_spikes.csv", glbSpkCntPN, glbSpkPN);
    SpikeCSVRecorder kcSpikes("kc_spikes.csv", glbSpkCntKC, glbSpkKC);
    SpikeCSVRecorder enSpikes("en_spikes.csv", glbSpkCntEN, glbSpkEN);
#endif  // RECORD_SPIKES

    // Update input data step
    IextStepPN = inputDataStep;

    // Loop through timesteps
    unsigned int numENSpikes = 0;
    while(iT < endTimestep)
    {
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

        numENSpikes += spikeCount_EN;
#ifdef RECORD_SPIKES
        // Record spikes
        pnSpikes.record(iT - startTimestep);
        kcSpikes.record(iT - startTimestep);
        enSpikes.record(iT - startTimestep);
#endif  // RECORD_SPIKES
    }

    return numENSpikes;
}
}   // anonymous namespace
//----------------------------------------------------------------------------
int main()
{
    // Initialize the library
    if(!glfwInit()) {
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    GLFWwindow *window = glfwCreateWindow(displayRenderWidth, displayRenderHeight + displayRenderWidth + 10,
                                          "Ant World", nullptr, nullptr);
    if(!window)
    {
        glfwTerminate();
        return -1;
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
    glLineWidth(4.0f);

    // Create key bitset and setc it as window user pointer
    KeyBitset keybits;
    glfwSetWindowUserPointer(window, &keybits);

    // Set key callback
    glfwSetKeyCallback(window, keyCallback);

    // Load route
    Route route(0.2f, "ant1_route1.bin");

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
    initGeNN();

    // Host OpenCV array to hold pixels read from screen
    cv::Mat snapshot(displayRenderHeight, displayRenderWidth, CV_8UC3);

    // Create snapshot processor to perform image processing on snapshot
    SnapshotProcessor snapshotProcessor(intermediateSnapshotWidth, intermediateSnapshowHeight,
                                        Parameters::inputWidth, Parameters::inputHeight);

    // Loop until the user closes the window
    float antX = route[0][0];
    float antY = route[0][1];
    float antHeading = route[0][2];

    State state = State::Training;

    unsigned int numSnapshots = 0;
    float distanceSinceLastPoint = 0.0f;
    unsigned int trainPoint = 0;

    unsigned int testingScan = 0;

    float bestHeading = 0.0f;
    unsigned int bestTestENSpikes = std::numeric_limits<unsigned int>::max();

    constexpr double halfScanAngle = Parameters::scanAngle;
    constexpr unsigned int numScanSteps = (unsigned int)round(Parameters::scanAngle / Parameters::scanStep);

    std::ofstream replay("test.csv");

    std::future<unsigned int> gennResult;
    while (!glfwWindowShouldClose(window)) {
        bool trainSnapshot = false;
        bool testSnapshot = false;
        const bool gennIdle = !gennResult.valid() || (gennResult.wait_for(std::chrono::seconds(0)) == future_status::ready);

        // Update heading and ant position based on keys
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
            antHeading = route[0][2];
            antX = route[0][0];
            antY = route[0][1];
        }

        // If GeNN is idle, trigger snapshots if keys are pressed
        if(gennIdle && keybits.test(KeyTrainSnapshot)) {
            trainSnapshot = true;
        }
        if(gennIdle && keybits.test(KeyTestSnapshot)) {
            testSnapshot = true;
        }

        // If we're training
        if(state == State::Training) {
            if(gennIdle) {
                // If we have just trained a snapshot, show spike count (for tuning purposes)
                if(gennResult.valid()) {
                    std::cout << "\t" << gennResult.get() << " EN spikes" << std::endl;
                }

                // If GeNN isn't training and we have more route points to train
                if(trainPoint < route.size()) {
                    // Snap ant to next route point
                    antX = route[trainPoint][0];
                    antY = route[trainPoint][1];
                    antHeading = route[trainPoint][2];

                    // If this isn't the first point
                    if(trainPoint > 0) {
                        // Calculate distance from last point
                        const float deltaX = antX - route[trainPoint - 1][0];
                        const float deltaY = antY - route[trainPoint - 1][1];
                        const float distance = std::sqrt((deltaX * deltaX) + (deltaY * deltaY));

                        // Add to total
                        distanceSinceLastPoint += distance;
                    }

                    // If this is the first point or we've gone further than snapshot distance
                    if(trainPoint == 0 || distanceSinceLastPoint > Parameters::snapshotDistance) {
                        // Set flag to train this snapshot
                        trainSnapshot = true;

                        // Count snapshots
                        numSnapshots++;

                        // Reset counter
                        distanceSinceLastPoint = 0.0f;
                    }

                    // Go onto next training point
                    trainPoint++;
                }
                // Otherwise, if we've reached end of route
                else {
                    std::cout << "Training complete (" << numSnapshots << " snapshots)" << std::endl;

                    // Go to testing state
                    state = State::Testing;

                    // Snap ant back to start of route, facing in starting scan direction
                    antX = route[0][0];
                    antY = route[0][1];
                    antHeading = route[0][2] - halfScanAngle;

                    // Reset scan
                    testingScan = 0;
                    bestTestENSpikes = std::numeric_limits<unsigned int>::max();

                    // Take snapshot
                    testSnapshot = true;
                }
            }
        }
        else if(state == State::Testing) {
            if(gennIdle) {
                // If the last snapshot was more familiar than the current best update
                const unsigned int numSpikes = gennResult.get();
                std::cout << "\t" << numSpikes << " EN spikes" << std::endl;

                // If this is an improvement on previous best spike count
                if(numSpikes < bestTestENSpikes) {
                    bestHeading = antHeading;
                    bestTestENSpikes = numSpikes;

                    std::cout << "\tUpdated result: " << bestHeading << " is most familiar heading with " << bestTestENSpikes << " spikes" << std::endl;
                }

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

                    replay << antX << "," << antY << std::endl;

                    // Reset scan
                    antHeading -= halfScanAngle;
                    testingScan = 0;
                    bestTestENSpikes = std::numeric_limits<unsigned int>::max();

                    // Take snapshot
                    testSnapshot = true;
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
            uint8_t *finalSnapshotData;
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