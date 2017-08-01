#pragma once

// Standard C++ includes
#include <string>

// OpenGL includes
#include <GL/glew.h>
#include <GL/glu.h>

//----------------------------------------------------------------------------
// World
//----------------------------------------------------------------------------
class World
{
public:
    World();
    World(const std::string &filename, const GLfloat (&worldColour)[3],
          const GLfloat (&groundColour)[3]);
    ~World();

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool load(const std::string &filename, const GLfloat (&worldColour)[3],
              const GLfloat (&groundColour)[3]);
    void bind() const;
    void render(bool shouldBind = true) const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    GLuint m_VAO;
    GLuint m_PositionVBO;
    GLuint m_ColourVBO;
    unsigned int m_NumVertices;
};