#pragma once

// Standard C++ includes
#include <array>
#include <set>
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
    Route(float arrowLength, const std::string &filename, double waypointDistance);
    ~Route();

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool load(const std::string &filename, double waypointDistance);
    void render(float antX, float antY, float antHeading) const;

    bool atDestination(float x, float y, float threshold) const;
    std::tuple<float, size_t> getDistanceToRoute(float x, float y) const;
    void setWaypointFamiliarity(size_t pos, double familiarity);

    size_t size() const{ return m_Route.size(); }

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    std::tuple<float, float, float> operator[](size_t pos) const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    GLuint m_RouteVAO;
    GLuint m_RoutePositionVBO;
    GLuint m_RouteColoursVBO;

    std::vector<std::array<float, 2>> m_Route;
    std::set<size_t> m_TrainedSnapshots;

    GLuint m_OverlayVAO;
    GLuint m_OverlayPositionVBO;
    GLuint m_OverlayColoursVBO;
};