#pragma once

// Standard C++ includes
#include <array>
#include <string>
#include <vector>

// OpenGL includes
#include <GL/glew.h>
#include <GL/glu.h>

//----------------------------------------------------------------------------
// Route
//----------------------------------------------------------------------------
class Route
{
public:
    Route();
    Route(const std::string &filename);
    ~Route();

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool load(const std::string &filename);
    void render() const;

    size_t size() const{ return m_Route.size(); }

    const std::array<float, 3> &operator[](size_t pos) const{ return m_Route[pos]; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    GLuint m_VAO;
    GLuint m_PositionVBO;
    std::vector<std::array<float, 3>> m_Route;
};