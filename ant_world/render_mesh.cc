#include "render_mesh.h"

// Standard C++ includes
#include <vector>

// Standard C includes
#include <cmath>

// Antworld includes
#include "common.h"

//----------------------------------------------------------------------------
// RenderMesh
//----------------------------------------------------------------------------
RenderMesh::RenderMesh() : m_VAO(0), m_PositionVBO(0), m_TextureCoordsVBO(0), m_NumVertices(0)
{
}
//----------------------------------------------------------------------------
RenderMesh::RenderMesh(float horizontalFOV, float verticalFOV,
                       unsigned int numHorizontalSegments, unsigned int numVerticalSegments)
{
    // We need a vertical for each segment and one extra
    const unsigned int numVerticals = numHorizontalSegments + 1;
    m_NumVertices = numVerticals * 2;

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
    glGenVertexArrays(1, &m_VAO);

    // Generate two vertex buffer objects, one for positions and one for texture coordinates
    glGenBuffers(1, &m_PositionVBO);
    glGenBuffers(1, &m_TextureCoordsVBO);

    // Bind vertex array
    glBindVertexArray(m_VAO);

    // Bind and upload positions buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_PositionVBO);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(GLfloat), positions.data(), GL_STATIC_READ);

    // Set vertex pointer and enable client state in VAO
    glVertexPointer(2, GL_FLOAT, 0, BUFFER_OFFSET(0));
    glEnableClientState(GL_VERTEX_ARRAY);

    // Bind and upload texture coordinates buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_TextureCoordsVBO);
    glBufferData(GL_ARRAY_BUFFER, textureCoords.size() * sizeof(GLfloat), textureCoords.data(), GL_STATIC_READ);

    // Set texture coordinate pointer and enable client state in VAO
    glTexCoordPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(0));
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
}
//----------------------------------------------------------------------------
void RenderMesh::render() const
{
    // Bind render mesh VAO
    glBindVertexArray(m_VAO);

    // Draw render mesh quad strip
    glDrawArrays(GL_QUAD_STRIP, 0, m_NumVertices);
}
//----------------------------------------------------------------------------
RenderMesh::~RenderMesh()
{
    // Delete render mesh objects
    glDeleteBuffers(1, &m_PositionVBO);
    glDeleteBuffers(1, &m_TextureCoordsVBO);
    glDeleteVertexArrays(1, &m_VAO);
}