// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

// GL
#include <GL/glew.h>
#include <GL/glu.h>

// GLFW
#include <GLFW/glfw3.h>

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define BUFFER_OFFSET(i) ((void*)(i))

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
constexpr GLfloat groundColour[] = {0.898f, 0.718f, 0.353f};
constexpr GLfloat worldColour[] = {0.0f, 1.0f, 0.0f};

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
        positions[0] = -10.0f;  positions[1] = -10.0f;  positions[2] = 0.0f;
        positions[3] = 10.5f;   positions[4] = 10.5f;   positions[5] = 0.0f;
        positions[6] = -10.0f;  positions[7] = 10.5f;   positions[8] = 0.0f;

        // Add second ground triangle vertex positions
        positions[9] = -10.0f;  positions[10] = -10.0f; positions[11] = 0.0f;
        positions[12] = 10.5f;  positions[13] = -10.0f; positions[14] = 0.0f;
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
}   // anonymous namespace

int main()
{
    // Initialize the library
    if(!glfwInit()) {
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    GLFWwindow *window = glfwCreateWindow(640, 178, "Hello World", nullptr, nullptr);
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
        return -1;
    }

    // Enable VSync
    glfwSwapInterval(1);

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

    std::cout << "Loaded world" << std::endl;

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Clear colour
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw world
        glDrawArrays(GL_TRIANGLES, 0, numVertices);

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Delete vertex buffer objects
    glDeleteBuffers(1, &worldPositionVBO);
    glDeleteBuffers(1, &worldColourVBO);

    glfwTerminate();
    return 0;
}