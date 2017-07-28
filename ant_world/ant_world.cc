// Standard C++ includes
#include <algorithm>
#include <bitset>
#include <fstream>
#include <future>
#include <iostream>
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

// Model includes
#include "parameters.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define BUFFER_OFFSET(i) ((void*)(i))

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
constexpr float antTurnSpeed = 1.0f;
constexpr float antMoveSpeed = 0.05f;

// Constant to multiply degrees by to get radians
constexpr float degreesToRadians = 0.017453293f;

constexpr int displayRenderWidth = 640;
constexpr int displayRenderHeight = 178;

constexpr int intermediateSnapshotWidth = 74;
constexpr int intermediateSnapshowHeight = 19;


// Enumeration of keys
enum Key
{
    KeyLeft,
    KeyRight,
    KeyUp,
    KeyDown,
    KeySnapshot,
    KeyMax
};

// Bitset used for passing which keys have been pressed between key callback and render loop
typedef std::bitset<KeyMax> KeyBitset;

// Loads world file from matlab format into position and colour vertex buffer objects
std::tuple<GLuint, GLuint, unsigned int> loadWorld(const std::string &filename, bool falseColour=true)
{
    // Generate two vertex buffer objects, one for positions and one for colours
    GLuint vbo[2];
    glGenBuffers(2, vbo);

    // Open file for binary IO
    std::ifstream input(filename, std::ios::binary);
    if(!input.good()) {
        throw std::runtime_error("Cannot open world file:" + filename);
    }

    // Seek to end of file, get size and rewind
    input.seekg(0, std::ios_base::end);
    const std::streampos numTriangles = input.tellg() / (sizeof(double) * 12);
    input.seekg(0);
    std::cout << "World has " << numTriangles << " triangles" << std::endl;
    {
        // Bind positions buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

        // Reserve 3 XYZ positions for each triangle and 6 for the ground
        std::vector<GLfloat> positions((6 + (numTriangles * 3)) * 3);

        // Add first ground triangle vertex positions
        positions[0] = 0.0f;    positions[1] = 0.0f;    positions[2] = 0.0f;
        positions[3] = 10.5f;   positions[4] = 10.5f;   positions[5] = 0.0f;
        positions[6] = 0.0f;    positions[7] = 10.5f;   positions[8] = 0.0f;

        // Add second ground triangle vertex positions
        positions[9] = 0.0f;    positions[10] = 0.0f;   positions[11] = 0.0f;
        positions[12] = 10.5f;  positions[13] = 0.0f;   positions[14] = 0.0f;
        positions[15] = 10.5f;  positions[16] = 10.5f;  positions[17] = 0.0f;

        // Loop through components(X, Y and Z)
        for(unsigned int c = 0; c < 3; c++) {
            // Loop through vertices in each triangle
            for(unsigned int v = 0; v < 3; v++) {
                // Loop through triangles
                for(unsigned int t = 0; t < numTriangles; t++) {
                    // Read triangle position component
                    double trianglePosition;
                    input.read(reinterpret_cast<char*>(&trianglePosition), sizeof(double));

                    // Copy three coordinates from triangle into correct place in vertex array
                    positions[18 + (t * 9) + (v * 3) + c] = (GLfloat)trianglePosition;
                }
            }
        }

        // Upload positions
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(GLfloat), positions.data(), GL_STATIC_DRAW);
    }

    {
        // Bind colours buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

        // Reserve 3 RGB colours for each triangle and for the ground
        std::vector<GLfloat> colours((6 + (numTriangles * 3)) * 3);

        // Ground triangle colours
        for(unsigned int c = 0; c < (6 * 3); c += 3) {
            colours[c] = groundColour[0];
            colours[c + 1] = groundColour[1];
            colours[c + 2] = groundColour[2];
        }

        // If we should 'false colour' world
        if(falseColour) {
            // Loop through triangles
            for(unsigned int t = 0; t < numTriangles; t++) {
                // Read triangle colour component
                // **NOTE** we only bother reading the R channel because colours are greyscale anyway
                double triangleColour;
                input.read(reinterpret_cast<char*>(&triangleColour), sizeof(double));

                // Loop through vertices that make up triangle and
                // set to world colour multiplied by triangle colour
                for(unsigned int v = 0; v < 3; v++) {
                    colours[18 + (t * 9) + (v * 3)] = worldColour[0] * triangleColour;
                    colours[18 + (t * 9) + (v * 3) + 1] = worldColour[1] * triangleColour;
                    colours[18 + (t * 9) + (v * 3) + 2] = worldColour[2] * triangleColour;
                }
            }
        }
        // Otherwise
        else {
            // Loop through components (R, G and B)
            for(unsigned int c = 0; c < 3; c++) {
                // Loop through triangles
                for(unsigned int t = 0; t < numTriangles; t++) {
                    // Read triangle colour component
                    double triangleColour;
                    input.read(reinterpret_cast<char*>(&triangleColour), sizeof(double));

                    // Copy it into correct position for each vertex in triangle
                    colours[18 + (t * 9) + c] = (GLfloat)triangleColour;
                    colours[18 + (t * 9) + c + 3] = (GLfloat)triangleColour;
                    colours[18 + (t * 9) + c + 6] = (GLfloat)triangleColour;
                }
            }
        }

        // Upload colours
        glBufferData(GL_ARRAY_BUFFER, colours.size() * sizeof(GLfloat), colours.data(), GL_STATIC_DRAW);
    }

    // Return VBO handles and index count
    return std::make_tuple(vbo[0], vbo[1], numTriangles * 3);
}
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
            keybits->set(KeySnapshot, newKeyState);
            break;
    }
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
        Timer<> t("Allocation:");
        allocateMem();
    }

    {
        Timer<> t("Initialization:");
        initialize();
    }

    {
        Timer<> t("Building connectivity:");

        buildFixedNumberPreConnector(Parameters::numPN, Parameters::numKC,
                                     Parameters::numPNSynapsesPerKC, CpnToKC, &allocatepnToKC, gen);
    }

    // Final setup
    {
        Timer<> t("Sparse init:");
        initant_world();
    }
}
//----------------------------------------------------------------------------
void presentToMB(uint8_t *inputData, unsigned int inputDataStep)
{
    // Convert simulation regime parameters to timesteps
    const unsigned int rewardTimestep = convertMsToTimesteps(Parameters::rewardTimeMs);
    const unsigned int presentDuration = convertMsToTimesteps(Parameters::presentDurationMs);
    const unsigned int postStimuliDuration = convertMsToTimesteps(Parameters::postStimuliDurationMs);

    const unsigned int duration = presentDuration + postStimuliDuration;

    std::cout << "Simulating for " << duration << " timesteps" << std::endl;

    // Open CSV output files
    SpikeCSVRecorder pnSpikes("pn_spikes.csv", glbSpkCntPN, glbSpkPN);
    SpikeCSVRecorder kcSpikes("kc_spikes.csv", glbSpkCntKC, glbSpkKC);
    SpikeCSVRecorder enSpikes("en_spikes.csv", glbSpkCntEN, glbSpkEN);

    // Update input data step
    IextStepPN = inputDataStep;

    // Loop through timesteps
    for(unsigned int t = 0; t < duration; t++)
    {
        // If we should be presenting an image
        if(t < presentDuration) {
            IextPN = NULL;
        }
        // Otherwise update offset to point to block of zeros
        else {
            IextPN = inputData;
        }

        // If we should reward in this timestep, inject dopamine
        if(t == rewardTimestep) {
            std::cout << "\tApplying reward at timestep " << t << std::endl;
            injectDopaminekcToEN = true;
        }

#ifndef CPU_ONLY
        // Simulate on GPU
        stepTimeGPU();

        // Download spikes
        pullPNCurrentSpikesFromDevice();
        pullKCCurrentSpikesFromDevice();
        pullENCurrentSpikesFromDevice();
#else
        // Simulate on CPU
        stepTimeCPU();
#endif
        // If a dopamine spike has been injected this timestep
        if(t == rewardTimestep) {
            const scalar tMs =  (scalar)t * DT;

            // Decay global dopamine traces
            dkcToEN = dkcToEN * std::exp(-(tMs - tDkcToEN) / Parameters::tauD);

            // Add effect of dopamine spike
            dkcToEN += Parameters::dopamineStrength;

            // Update last reward time
            tDkcToEN = tMs;

            // Clear dopamine injection flags
            injectDopaminekcToEN = false;
        }

        // Record spikes
        pnSpikes.record(t);
        kcSpikes.record(t);
        enSpikes.record(t);
    }
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
    GLFWwindow *window = glfwCreateWindow(displayRenderWidth, displayRenderHeight, "Ant World", nullptr, nullptr);
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
    glfwSwapInterval(1);

    // Set clear colour to match matlab and enable depth test
    glClearColor(0.0f, 1.0f, 1.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    
    // Create key bitset and setc it as window user pointer
    KeyBitset keybits;
    glfwSetWindowUserPointer(window, &keybits);

    // Set key callback
    glfwSetKeyCallback(window, keyCallback);

    // Load world into OpenGL
    GLuint worldPositionVBO;
    GLuint worldColourVBO;
    unsigned int numVertices;
    std::tie(worldPositionVBO, worldColourVBO, numVertices) = loadWorld("world5000_gray.bin");

    // Bind world position VBO
    glBindBuffer(GL_ARRAY_BUFFER, worldPositionVBO);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(0));

    // Bind world colour VBO
    glBindBuffer(GL_ARRAY_BUFFER, worldColourVBO);
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(0));

    bool ortho = false;
    glMatrixMode(GL_PROJECTION);

    // Setup camera to look down on whole world
    if(ortho) {
        glOrtho(-5.0, 5.0,
                -5.0, 5.0,
                10, -1.0);
    }
    else {
        gluPerspective(76.0,
                       36.0 / 10.0,
                       0.0001, 10.0);
    }
    glMatrixMode(GL_MODELVIEW);

    // Centre the world
    glLoadIdentity();

    if(ortho) {
        glTranslatef(-5.0f, -5.0f, 0.0f);
    }
    else {
        glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
        glTranslatef(-5.0f, -5.0f, -0.2f);
    }

    // Initialize GeNN
    initGeNN();

    // Host OpenCV array to hold pixels read from screen
    cv::Mat snapshot(displayRenderHeight, displayRenderWidth, CV_8UC3);

    // Host OpenCV array to hold intermediate resolution colour snapshot
    cv::Mat intermediateSnapshot(intermediateSnapshowHeight, intermediateSnapshotWidth, CV_8UC3);

    // Host OpenCV array to hold intermediate resolution greyscale snapshot
    cv::Mat intermediateSnapshotGreyscale(intermediateSnapshowHeight, intermediateSnapshotWidth, CV_8UC1);

    // Host OpenCV array to hold final resolution greyscale snapshot
    cv::Mat finalSnapshot(Parameters::inputHeight, Parameters::inputWidth, CV_8UC1);

    // GPU OpenCV array to hold
    cv::cuda::GpuMat finalSnapshotGPU(Parameters::inputHeight, Parameters::inputWidth, CV_8UC1);

    // Create CLAHE algorithm for histogram normalization
    // **NOTE** parameters to match Matlab defaults
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(0.01 * 255.0, cv::Size(8, 8));

    // Loop until the user closes the window
    float antHeading = 0.0f;
    float antX = 5.0f;
    float antY = 5.0f;
    std::future<void> gennResult;
    while (!glfwWindowShouldClose(window)) {
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

        // Build new modelview transform
        glLoadIdentity();
        glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
        glRotatef(antHeading, 0.0f, 0.0f, 1.0f);
        glTranslatef(-antX, -antY, -0.2f);

        // Draw to window
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, numVertices);

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // If snapshot key is pressed
        if(keybits.test(KeySnapshot)) {
            // Read pixels from framebuffer
            // **TODO** it should be theoretically possible to go directly from frame buffer to GpuMat
            glReadPixels(0, 0, displayRenderWidth, displayRenderHeight,
                         GL_BGR, GL_UNSIGNED_BYTE, snapshot.data);

            // Downsample to intermediate size
            cv::resize(snapshot, intermediateSnapshot,
                       cv::Size(intermediateSnapshotWidth, intermediateSnapshowHeight));

            // Convert to greyscale
            cv::cvtColor(intermediateSnapshot, intermediateSnapshotGreyscale, CV_BGR2GRAY);

            // Invert image
            cv::subtract(255, intermediateSnapshotGreyscale, intermediateSnapshotGreyscale);

            // Apply histogram normalization
            clahe->apply(intermediateSnapshotGreyscale, intermediateSnapshotGreyscale);

            // Finally resample down to final size
            cv::resize(intermediateSnapshotGreyscale, finalSnapshot,
                       cv::Size(Parameters::inputWidth, Parameters::inputHeight),
                       0.0, 0.0, CV_INTER_CUBIC);

            cv::imwrite("snapshot.png", finalSnapshot);

            // Upload final snapshot to GPU
            finalSnapshotGPU.upload(finalSnapshot);

            // Extract device pointers and step
            auto finalSnapshotPtrStep = (cv::cuda::PtrStep<uint8_t>)finalSnapshotGPU;
            gennResult = std::async(std::launch::async, presentToMB,
                                    finalSnapshotPtrStep.data, finalSnapshotPtrStep.step);
        }

        // Poll for and process events
        glfwPollEvents();
    }

    // Delete vertex buffer objects
    glDeleteBuffers(1, &worldPositionVBO);
    glDeleteBuffers(1, &worldColourVBO);

    glfwTerminate();
    return 0;
}