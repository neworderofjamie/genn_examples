#include "world.h"

// Standard C++ includes
#include <fstream>
#include <iostream>
#include <vector>

// Antworld includes
#include "common.h"

//----------------------------------------------------------------------------
// World
//----------------------------------------------------------------------------
World::World() : m_VAO(0), m_PositionVBO(0), m_ColourVBO(0), m_NumVertices(0)
{
}
//----------------------------------------------------------------------------
World::World(const std::string &filename, const GLfloat (&worldColour)[3],
             const GLfloat (&groundColour)[3])
{
    if(!load(filename, worldColour, groundColour)) {
        throw std::runtime_error("Cannot load world");
    }
}
//----------------------------------------------------------------------------
World::~World()
{
    // Delete world objects
    glDeleteBuffers(1, &m_PositionVBO);
    glDeleteBuffers(1, &m_ColourVBO);
    glDeleteVertexArrays(1, &m_VAO);
}
//----------------------------------------------------------------------------
bool World::load(const std::string &filename, const GLfloat (&worldColour)[3],
                 const GLfloat (&groundColour)[3])
{
    // Create a vertex array object to bind everything together
    glGenVertexArrays(1, &m_VAO);

    // Generate two vertex buffer objects, one for positions and one for colours
    glGenBuffers(1, &m_PositionVBO);
    glGenBuffers(1, &m_ColourVBO);

    // Open file for binary IO
    std::ifstream input(filename, std::ios::binary);
    if(!input.good()) {
        std::cerr << "Cannot open world file:" << filename << std::endl;
        return false;
    }

    // Seek to end of file, get size and rewind
    input.seekg(0, std::ios_base::end);
    const std::streampos numTriangles = input.tellg() / (sizeof(double) * 12);
    m_NumVertices = numTriangles * 3;
    input.seekg(0);
    std::cout << "World has " << numTriangles << " triangles" << std::endl;

    // Bind vertex array
    glBindVertexArray(m_VAO);

    {
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

        // Bind positions buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_PositionVBO);

        // Upload positions
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(GLfloat), positions.data(), GL_STATIC_DRAW);

        // Set vertex pointer and enable client state in VAO
        glVertexPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(0));
        glEnableClientState(GL_VERTEX_ARRAY);
    }

    {
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

        // Bind colours buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_ColourVBO);

        // Upload colours
        glBufferData(GL_ARRAY_BUFFER, colours.size() * sizeof(GLfloat), colours.data(), GL_STATIC_DRAW);

        // Set colour pointer and enable client state in VAO
        glColorPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(0));
        glEnableClientState(GL_COLOR_ARRAY);
    }

    return true;
}
//----------------------------------------------------------------------------
void World::bind() const
{
    // Bind world VAO
    glBindVertexArray(m_VAO);
}
//----------------------------------------------------------------------------
void World::render(bool shouldBind) const
{
    // If we should bind as well, do so
    if(shouldBind) {
        bind();
    }

    // Draw world
    glDrawArrays(GL_TRIANGLES, 0, m_NumVertices);
}
//----------------------------------------------------------------------------
