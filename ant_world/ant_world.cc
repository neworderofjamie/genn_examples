// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

// GL
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

#define BUFFER_OFFSET(i) ((void*)(i))

std::tuple<GLuint, GLuint, unsigned int> loadWorld(const std::string &filename)
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

        // Reserve 3 XYZ positions for each triangle
        std::vector<GLfloat> positions(numTriangles * 3 * 3);

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
                    positions[(t * 9) + (v * 3) + c] = (GLfloat)trianglePosition;
                }
            }
        }

        // Upload positions
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(GLfloat), positions.data(), GL_STATIC_DRAW);
    }

    {
        // Bind colours buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

        // Reserve 3 RGB colours for each triangle
        std::vector<GLfloat> colours(numTriangles * 3 * 3);

        // Loop through components (R, G and B)
        for(unsigned int c = 0; c < 3; c++) {
            // Loop through triangles
            for(unsigned int t = 0; t < numTriangles; t++) {
                // Read triangle colour component
                double triangleColour;
                input.read(reinterpret_cast<char*>(&triangleColour), sizeof(double));

                // Copy it into correct position for each vertex in triangle
                colours[(t * 9) + c] = (GLfloat)triangleColour;
                colours[(t * 9) + c + 3] = (GLfloat)triangleColour;
                colours[(t * 9) + c + 6] = (GLfloat)triangleColour;
            }
        }

        std::cout << colours[0] << "," << colours[1] << "," << colours[2] << std::endl;

        // Upload colours
        glBufferData(GL_ARRAY_BUFFER, colours.size() * sizeof(GLfloat), colours.data(), GL_STATIC_DRAW);
    }

    return std::make_tuple(vbo[0], vbo[1], numTriangles * 3);
}

int main()
{
    // Initialize the library
    if(!glfwInit()) {
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    GLFWwindow *window = glfwCreateWindow(640, 480, "Hello World", nullptr, nullptr);
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

    glMatrixMode(GL_PROJECTION);

    // Setup camera to look down on whole world
    glOrtho(-5.0, 5.0,
            -5.0, 5.0,
            10, -1.0);

    glMatrixMode(GL_MODELVIEW);

    // Centre the world
    glLoadIdentity();
    glTranslatef(-5.0f, -5.0f, 0.0f);

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

    glDeleteBuffers(1, &worldPositionVBO);
    glDeleteBuffers(1, &worldColourVBO);

    glfwTerminate();
    return 0;
}