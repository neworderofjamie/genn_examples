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

// Model includes
#include "parameters.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define BUFFER_OFFSET(i) ((void*)(i))
//#define RECORD_SPIKES

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
constexpr float radiansToDegrees = 57.295779513f;
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

// Loads world file from matlab format into position and colour vertex buffer objects
std::tuple<GLuint, GLuint, GLuint, unsigned int> loadWorld(const std::string &filename)
{
    // Create a vertex array object to bind everything together
    GLuint vao;
    glGenVertexArrays(1, &vao);

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

    // Bind vertex array
    glBindVertexArray(vao);

    {
        // Bind positions buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

        // ReserouteVBOrve 3 XYZ positions for each triangle and 6 for the ground
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
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(GLfloat), positions.data(), GL_STATIC_READ);

        // Set vertex pointer and enable client state in VAO
        glVertexPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(0));
        glEnableClientState(GL_VERTEX_ARRAY);
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

        // Upload colours
        glBufferData(GL_ARRAY_BUFFER, colours.size() * sizeof(GLfloat), colours.data(), GL_STATIC_READ);

        // Set colour pointer and enable client state in VAO
        glColorPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(0));
        glEnableClientState(GL_COLOR_ARRAY);
    }

    // Return VBO handles and index count
    return std::make_tuple(vao, vbo[0], vbo[1], numTriangles * 3);
}
//----------------------------------------------------------------------------
std::tuple<std::vector<std::array<float, 3>>, GLuint, GLuint, unsigned int> loadRoute(const std::string &filename)
{
    // Open file for binary IO
    std::ifstream input(filename, std::ios::binary);
    if(!input.good()) {
        throw std::runtime_error("Cannot open route file:" + filename);
    }

    // Seek to end of file, get size and rewind
    input.seekg(0, std::ios_base::end);
    const std::streampos numPoints = input.tellg() / (sizeof(double) * 3);
    input.seekg(0);
    std::cout << "Route has " << numPoints << " points" << std::endl;

    // Resize route
    std::vector<std::array<float, 3>> route(numPoints);

    // Loop through components(X, Y and heading)
    for(unsigned int c = 0; c < 3; c++) {
        // Heading is correctly scaled by X and Y need converting into metres
        const float scale = (c == 2) ? 1.0f : (1.0f / 100.0f);

        // Loop through points on path
        for(unsigned int i = 0; i < numPoints; i++) {
            // Read point component
            double pointPosition;
            input.read(reinterpret_cast<char*>(&pointPosition), sizeof(double));

            // Convert to float, scale and insert into route
            route[i][c] = (float)pointPosition * scale;
        }
    }

    // Create a vertex array object to bind everything together
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Generate vertex buffer objects for positions
    GLuint vbo;
    glGenBuffers(1, &vbo);

    // Bind vertex array
    glBindVertexArray(vao);

    // Bind and upload positions buffer
    // **NOTE** we're not actually going to be rendering the 3rd component as it's an angle not a z-coordinate
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, route.size() * sizeof(GLfloat) * 3, route.data(), GL_STATIC_READ);

    // Set vertex pointer to stride over angles and enable client state in VAO
    glVertexPointer(2, GL_FLOAT, 3 * sizeof(float), BUFFER_OFFSET(0));
    glEnableClientState(GL_VERTEX_ARRAY);

    // Return route data, vao, vbo and route size
    return std::make_tuple(route, vao, vbo, route.size());
}
//----------------------------------------------------------------------------
std::tuple<GLuint, GLuint, GLuint, unsigned int> buildRenderMesh(float horizontalFOV, float verticalFOV,
                                                                 unsigned int numHorizontalSegments, unsigned int numVerticalSegments)
{
    // We need a vertical for each segment and one extra
    const unsigned int numVerticals = numHorizontalSegments + 1;

    // Reserve 2 XY positions and 2 SRT texture coordinates for each vertical
    std::vector<GLfloat> positions;
    std::vector<GLfloat> textureCoords;
    positions.reserve(numVerticals * 2 * 2);
    textureCoords.reserve(numVerticals * 3 * 2);

    // Loop through vertices
    const float segmentWidth = 1.0f / (float)numHorizontalSegments;
    const float startLatitude = -horizontalFOV / 2.0f;
    const float latitudeStep = horizontalFOV / (float)numHorizontalSegments;

    const float bottomLongitude = -verticalFOV / 2.0f;
    const float sinBottomLongitude = sin(bottomLongitude * degreesToRadians);
    const float cosBottomLongitude = cos(bottomLongitude * degreesToRadians);
    const float topLongitude = verticalFOV / 2.0f;
    const float sinTopLongitude = sin(topLongitude * degreesToRadians);
    const float cosTopLongitude = cos(topLongitude * degreesToRadians);

    for(unsigned int i = 0; i < numVerticals; i++) {
        // Calculate screenspace segment position
        const float x = segmentWidth * (float)i;

        // Calculate angle of vertical and hence S and T components of texture coordinate
        const float latitude = startLatitude + ((float)i * latitudeStep);
        const float sinLatitude = sin(latitude * degreesToRadians);
        const float cosLatitude = cos(latitude * degreesToRadians);

        // Add bottom vertex position
        positions.push_back(x);
        positions.push_back(1.0f);

        // Add bottom texture coordinate
        textureCoords.push_back(sinLatitude * cosBottomLongitude);
        textureCoords.push_back(sinBottomLongitude);
        textureCoords.push_back(cosLatitude * cosBottomLongitude);

        // Add top vertex position
        positions.push_back(x);
        positions.push_back(0.0f);

        // Add top texture coordinate
        textureCoords.push_back(sinLatitude * cosTopLongitude);
        textureCoords.push_back(sinTopLongitude);
        textureCoords.push_back(cosLatitude * cosTopLongitude);
    }

    // Create a vertex array object to bind everything together
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Generate two vertex buffer objects, one for positions and one for texture coordinates
    GLuint vbo[2];
    glGenBuffers(2, vbo);

    // Bind vertex array
    glBindVertexArray(vao);

    // Bind and upload positions buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(GLfloat), positions.data(), GL_STATIC_READ);

    // Set vertex pointer and enable client state in VAO
    glVertexPointer(2, GL_FLOAT, 0, BUFFER_OFFSET(0));
    glEnableClientState(GL_VERTEX_ARRAY);

    // Bind and upload texture coordinates buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, textureCoords.size() * sizeof(GLfloat), textureCoords.data(), GL_STATIC_READ);

    // Set texture coordinate pointer and enable client state in VAO
    glTexCoordPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(0));
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    // Return VBO handles and vertex count
    return std::make_tuple(vao, vbo[0], vbo[1], numVerticals * 2);
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
                   GLuint worldVAO, unsigned int numWorldVertices,
                   GLuint renderMeshVAO, unsigned int numRenderMeshVertices,
                   GLuint cubemapFBO, GLuint cubemapTexture, const GLfloat (&cubeFaceLookAtMatrices)[6][16])
{
    // Configure viewport to cubemap-sized square
    glViewport(0, 0, 256, 256);

    // Bind world VAO
    glBindVertexArray(worldVAO);

    // Bind the cubemap FBO for offscreen rendering
    glBindFramebuffer(GL_FRAMEBUFFER, cubemapFBO);

    // Configure perspective projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.0,
                   1.0,
                   0.001, 100.0);

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
        glDrawArrays(GL_TRIANGLES, 0, numWorldVertices);
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

    // Bind render mesh VAO
    glBindVertexArray(renderMeshVAO);

    // Draw render mesh quad strip
    glDrawArrays(GL_QUAD_STRIP, 0, numRenderMeshVertices);

    // Disable texture coordinate array, cube map texture and cube map texturing!
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    glDisable(GL_TEXTURE_CUBE_MAP);

}
//----------------------------------------------------------------------------
void renderTopDownView(GLuint worldVAO, unsigned int numWorldVertices,
                       GLuint routeVAO, unsigned int numRouteVertices)
{
    // Set viewport to square at bottom of screen
    glViewport(0, 0, displayRenderWidth, displayRenderWidth);

    // Bind world VAO
    glBindVertexArray(worldVAO);

    // Configure top-down orthographic projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-5.0, 5.0,
            -5.0, 5.0,
            -10, 1.0);

    // Build modelview matrix to centre world
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(-5.0f, -5.0f, 0.0f);

    // Draw world
    glDrawArrays(GL_TRIANGLES, 0, numWorldVertices);

    // Bind route VAO
    glBindVertexArray(routeVAO);

    // Translate above the terrain and draw route
    glTranslatef(0.0f, 0.0f, 1.0f);
    glDrawArrays(GL_LINE_STRIP, 0, numRouteVertices);
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
    Timer<> timer("Simulation:");

    // Convert simulation regime parameters to timesteps
    const unsigned long long startTimestep = iT;
    const unsigned long long rewardTimestep = iT + convertMsToTimesteps(Parameters::rewardTimeMs);
    const unsigned int presentDuration = convertMsToTimesteps(Parameters::presentDurationMs);
    const unsigned int postStimuliDuration = convertMsToTimesteps(Parameters::postStimuliDurationMs);

    const unsigned int duration = presentDuration + postStimuliDuration;
    const unsigned long long endPresentTimestep = iT + presentDuration;
    const unsigned long long endTimestep = iT + duration;
    std::cout << "Simulating from " << startTimestep << " to " << endTimestep << std::endl;
    std::cout << "Presenting snapshot until " << endPresentTimestep << std::endl;
    if(reward) {
        std::cout << "Rewarding at " << rewardTimestep << std::endl;
    }

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

    std::cout << numENSpikes << " EN spikes" << std::endl;
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
    std::vector<std::array<float, 3>> route;
    GLuint routeVAO;
    GLuint routePositionVBO;
    unsigned int numRouteVertices;
    std::tie(route, routeVAO, routePositionVBO, numRouteVertices) = loadRoute("ant1_route1.bin");

    // Create VBO from route
    // **NOTE** we're not actually going to be rendering the 3rd component as it's an angle not a z-coordinate
    GLuint routeVBO;
    glGenBuffers(1, &routeVBO);
    glBindBuffer(GL_ARRAY_BUFFER, routeVBO);
    glBufferData(GL_ARRAY_BUFFER, route.size() * sizeof(GLfloat) * 3, route.data(), GL_STATIC_READ);

    // Load world into OpenGL
    GLuint worldVAO;
    GLuint worldPositionVBO;
    GLuint worldColourVBO;
    unsigned int numWorldVertices;
    std::tie(worldVAO, worldPositionVBO, worldColourVBO, numWorldVertices) = loadWorld("world5000_gray.bin");

    // Build mesh to render cubemap to screen
    GLuint renderMeshVAO;
    GLuint renderMeshPositionVBO;
    GLuint renderMeshTextureCoordsVBO;
    unsigned int numRenderMeshVertices;
    std::tie(renderMeshVAO, renderMeshPositionVBO, renderMeshTextureCoordsVBO, numRenderMeshVertices) = buildRenderMesh(296.0f, 76.0f, 40, 10);

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
    float antX = route[0][0];
    float antY = route[0][1];
    float antHeading = route[0][2] - 90.0f;

    State state = State::Idle;

    unsigned int numSnapshots = 0;
    float distanceSinceLastPoint = 0.0f;
    unsigned int trainPoint = 1;

    unsigned int testingScan = 0;

    unsigned int bestTest = 0;
    unsigned int bestTestENSpikes = std::numeric_limits<unsigned int>::max();

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
            // If GeNN isn't training and we have more route points to train
            if(gennIdle && trainPoint < route.size()) {
                // Snap ant to next route point
                antX = route[trainPoint][0];
                antY = route[trainPoint][1];
                antHeading = route[trainPoint][2] - 90.0f;

                // Calculate distance from last point
                const float deltaX = antX - route[trainPoint - 1][0];
                const float deltaY = antY - route[trainPoint - 1][1];
                const float distance = std::sqrt((deltaX * deltaX) + (deltaY * deltaY));

                // Add distance to total
                distanceSinceLastPoint += distance;

                // If we've gone further than 10cm
                if(distanceSinceLastPoint > (10.0 / 100.0f)) {
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
            else if(gennIdle && trainPoint == route.size()) {
                std::cout << "Training complete (" << numSnapshots << " snapshots)" << std::endl;
                state = State::Idle;
                // Go to testing state
                /*state = State::Testing;

                // Snap ant back to start of route, facing in starting scan direction
                antX = route[0][0];
                antY = route[0][1];
                antHeading = route[0][2] - 60.0f;

                // Reset scan
                testingScan = 0;
                bestTestENSpikes = std::numeric_limits<unsigned int>::max();

                // Take snapshot
                testSnapshot = true;*/
            }
        }
        else if(state == State::Testing) {
            if(gennIdle) {
                // If the last snapshot was more familiar than the current best update
                const unsigned int numSpikes = gennResult.get();
                if(numSpikes < bestTestENSpikes) {
                    bestTest = testingScan;
                    bestTestENSpikes = numSpikes;

                    std::cout << "Updated result: " << bestTest << " is most familiar direction with " << bestTestENSpikes << " spikes" << std::endl;
                }

                // Go onto next scan
                testingScan++;

                // If scan isn't complete
                if(testingScan < 12) {
                    // Scan right
                    antHeading += 10.0f;

                    // Take test snapshot
                    testSnapshot = true;
                }
                else {
                    std::cout << "Scan complete: " << bestTest << " is most familiar direction with " << bestTestENSpikes << " spikes" << std::endl;

                    // Return ant to it's best heading
                    antHeading -= 10.0f * (float)(12 - bestTest);

                    // Move ant forward by 10cm
                    antX += 0.01f * sin(antHeading * degreesToRadians);
                    antY += 0.01f * cos(antHeading * degreesToRadians);

                    replay << antX << "," << antY << std::endl;

                    // Reset scan
                    antHeading -= 60.0f;
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
                      worldVAO, numWorldVertices,
                      renderMeshVAO, numRenderMeshVertices,
                      fbo, cubemap, cubeFaceLookAtMatrices);

        // Render top-down view at bottom of the screen
        renderTopDownView(worldVAO, numWorldVertices,
                          routeVAO, numRouteVertices);


        // Swap front and back buffers
        glfwSwapBuffers(window);

        // If we should take a snapshot
        if(trainSnapshot || testSnapshot) {
            Timer<> timer("Snapshot generation:");

            // Read pixels from framebuffer
            // **TODO** it should be theoretically possible to go directly from frame buffer to GpuMat
            glReadPixels(0, displayRenderWidth + 10, displayRenderWidth, displayRenderHeight,
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

            // Start simulation, applying reward if we are training
            gennResult = std::async(std::launch::async, presentToMB,
                                    finalSnapshotPtrStep.data, finalSnapshotPtrStep.step, trainSnapshot);
        }

        // Poll for and process events
        glfwPollEvents();
    }

    // Delete world objects
    glDeleteBuffers(1, &worldPositionVBO);
    glDeleteBuffers(1, &worldColourVBO);
    glDeleteVertexArrays(1, &worldVAO);


    // Delete render mesh objects
    glDeleteBuffers(1, &renderMeshPositionVBO);
    glDeleteBuffers(1, &renderMeshTextureCoordsVBO);
    glDeleteVertexArrays(1, &renderMeshVAO);

    glfwTerminate();
    return 0;
}