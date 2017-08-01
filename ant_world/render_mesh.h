#pragma once

// OpenGL includes
#include <GL/glew.h>
#include <GL/glu.h>

//----------------------------------------------------------------------------
// RenderMesh
//----------------------------------------------------------------------------
class RenderMesh
{
public:
    RenderMesh();
    RenderMesh(float horizontalFOV, float verticalFOV, float startLongitude,
               unsigned int numHorizontalSegments, unsigned int numVerticalSegments);
    ~RenderMesh();

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void render() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    GLuint m_VAO;
    GLuint m_PositionVBO;
    GLuint m_TextureCoordsVBO;
    GLuint m_IBO;
    unsigned int m_NumIndices;

};