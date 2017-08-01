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
    Route(float arrowLength);
    Route(float arrowLength, const std::string &filename);
    ~Route();

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool load(const std::string &filename);
    void render(float antX, float antY, float antHeading) const;
    size_t size() const{ return m_Route.size(); }

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    const std::array<float, 3> &operator[](size_t pos) const{ return m_Route[pos]; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    GLuint m_RouteVAO;
    GLuint m_RoutePositionVBO;
    std::vector<std::array<float, 3>> m_Route;

    GLuint m_OverlayVAO;
    GLuint m_OverlayPositionVBO;
    GLuint m_OverlayColoursVBO;
};