#include "route.h"

// Standard C++ includes
#include <fstream>
#include <iostream>

// Antworld includes
#include "common.h"

//----------------------------------------------------------------------------
// Route
//----------------------------------------------------------------------------
Route::Route() : m_VAO(0), m_PositionVBO(0)
{
}
//----------------------------------------------------------------------------
Route::Route(const std::string &filename)
{
     if(!load(filename)) {
        throw std::runtime_error("Cannot load route");
    }
}
//----------------------------------------------------------------------------
Route::~Route()
{
     // Delete world objects
    glDeleteBuffers(1, &m_PositionVBO);
    glDeleteVertexArrays(1, &m_VAO);
}
//----------------------------------------------------------------------------
bool Route::load(const std::string &filename)
{
    // Open file for binary IO
    std::ifstream input(filename, std::ios::binary);
    if(!input.good()) {
        std::cerr << "Cannot open route file:" << filename << std::endl;
        return false;
    }

    // Seek to end of file, get size and rewind
    input.seekg(0, std::ios_base::end);
    const std::streampos numPoints = input.tellg() / (sizeof(double) * 3);
    input.seekg(0);
    std::cout << "Route has " << numPoints << " points" << std::endl;

    // Resize route
    m_Route.resize(numPoints);

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
            m_Route[i][c] = (float)pointPosition * scale;
        }
    }

    // Create a vertex array object to bind everything together
    glGenVertexArrays(1, &m_VAO);

    // Generate vertex buffer objects for positions
    glGenBuffers(1, &m_PositionVBO);

    // Bind vertex array
    glBindVertexArray(m_VAO);

    // Bind and upload positions buffer
    // **NOTE** we're not actually going to be rendering the 3rd component as it's an angle not a z-coordinate
    glBindBuffer(GL_ARRAY_BUFFER, m_PositionVBO);
    glBufferData(GL_ARRAY_BUFFER, m_Route.size() * sizeof(GLfloat) * 3, m_Route.data(), GL_STATIC_DRAW);

    // Set vertex pointer to stride over angles and enable client state in VAO
    glVertexPointer(2, GL_FLOAT, 3 * sizeof(float), BUFFER_OFFSET(0));
    glEnableClientState(GL_VERTEX_ARRAY);

    return true;
}
//----------------------------------------------------------------------------
void Route::render() const
{
     // Bind route VAO
    glBindVertexArray(m_VAO);

    // Draw route
    glDrawArrays(GL_LINE_STRIP, 0, m_Route.size());
}