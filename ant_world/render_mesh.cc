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
RenderMesh::RenderMesh() : m_VAO(0), m_PositionVBO(0), m_TextureCoordsVBO(0), m_IBO(0), m_NumIndices(0)
{
}
//----------------------------------------------------------------------------
RenderMesh::RenderMesh(float horizontalFOV, float verticalFOV,
                       unsigned int numHorizontalSegments, unsigned int numVerticalSegments)
{
    // We need a vertical for each segment and one extra
    const unsigned int numHorizontalVerts = numHorizontalSegments + 1;
    const unsigned int numVerticalVerts = numVerticalSegments + 1;

    // Create a vertex array object to bind everything together
    glGenVertexArrays(1, &m_VAO);

    // Bind vertex array
    glBindVertexArray(m_VAO);

    {
        // Calculate number of vertices in mesh
        const unsigned int numVertices = numHorizontalVerts * numVerticalVerts;

        // Reserve 2 XY positions and 2 SRT texture coordinates for each vertical
        std::vector<GLfloat> positions;
        std::vector<GLfloat> textureCoords;
        positions.reserve(numVertices * 2);
        textureCoords.reserve(numVertices * 2);

        const float segmentWidth = 1.0f / (float)numHorizontalSegments;
        const float startLatitude = -horizontalFOV / 2.0f;
        const float latitudeStep = horizontalFOV / (float)numHorizontalSegments;

        const float segmentHeight = 1.0f / (float)numVerticalSegments;
        const float startLongitude = verticalFOV / 2.0f;
        const float longitudeStep = -verticalFOV / (float)numVerticalSegments;

        // Loop through vertices
        for(unsigned int j = 0; j < numVerticalVerts; j++) {
            // Calculate screenspace segment y position
            const float y = segmentHeight * (float)j;

            // Calculate angle of hoirzontal and calculate it's sin and cos
            const float longitude = startLongitude + ((float)j * longitudeStep);
            const float sinLongitude = sin(longitude * degreesToRadians);
            const float cosLongitude = cos(longitude * degreesToRadians);

            for(unsigned int i = 0; i < numHorizontalVerts; i++) {
                // Calculate screenspace segment x position
                const float x = segmentWidth * (float)i;

                // Calculate angle of vertical and calculate it's sin and cos
                const float latitude = startLatitude + ((float)i * latitudeStep);
                const float sinLatitude = sin(latitude * degreesToRadians);
                const float cosLatitude = cos(latitude * degreesToRadians);

                // Add vertex position
                positions.push_back(x);
                positions.push_back(y);

                // Add vertex texture coordinate
                textureCoords.push_back(sinLatitude * cosLongitude);
                textureCoords.push_back(sinLongitude);
                textureCoords.push_back(cosLatitude * cosLongitude);
            }
        }

        // Generate two vertex buffer objects, one for positions and one for texture coordinates
        glGenBuffers(1, &m_PositionVBO);
        glGenBuffers(1, &m_TextureCoordsVBO);

        // Bind and upload positions buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_PositionVBO);
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(GLfloat), positions.data(), GL_STATIC_DRAW);

        // Set vertex pointer and enable client state in VAO
        glVertexPointer(2, GL_FLOAT, 0, BUFFER_OFFSET(0));
        glEnableClientState(GL_VERTEX_ARRAY);

        // Bind and upload texture coordinates buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_TextureCoordsVBO);
        glBufferData(GL_ARRAY_BUFFER, textureCoords.size() * sizeof(GLfloat), textureCoords.data(), GL_STATIC_DRAW);

        // Set texture coordinate pointer and enable client state in VAO
        glTexCoordPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(0));
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    {
        // Calculate number of quads required to draw mesh
        m_NumIndices = numHorizontalSegments * numVerticalSegments * 4;

        // Reserce indices
        std::vector<GLuint> indices;
        indices.reserve(m_NumIndices);

        // Loop through quads
        for(unsigned int y = 0; y < numVerticalSegments; y++) {
            for(unsigned int x = 0; x < numHorizontalSegments; x++) {
                indices.push_back((y * numHorizontalVerts) + x);
                indices.push_back((y * numHorizontalVerts) + x + 1);
                indices.push_back(((y + 1) * numHorizontalVerts) + x + 1);
                indices.push_back(((y + 1) * numHorizontalVerts) + x);
            }
        }

        // Generate index buffer objects to hold primitive indices
        glGenBuffers(1, &m_IBO);

        // Bind and upload index buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
    }
}
//----------------------------------------------------------------------------
RenderMesh::~RenderMesh()
{
    // Delete render mesh objects
    glDeleteBuffers(1, &m_PositionVBO);
    glDeleteBuffers(1, &m_TextureCoordsVBO);
    glDeleteBuffers(1, &m_IBO);
    glDeleteVertexArrays(1, &m_VAO);
}
//----------------------------------------------------------------------------
void RenderMesh::render() const
{
    // Bind render mesh VAO
    glBindVertexArray(m_VAO);

    // Draw render mesh quads
    glDrawElements(GL_QUADS, m_NumIndices, GL_UNSIGNED_INT, BUFFER_OFFSET(0));
}
