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
    Route(float arrowLength, unsigned int maxRouteEntries);
    Route(float arrowLength, unsigned int maxRouteEntries, const std::string &filename);
    ~Route();

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool load(const std::string &filename);
    void render(float antX, float antY, float antHeading) const;

    bool atDestination(float x, float y, float threshold) const;
    std::tuple<float, size_t> getDistanceToRoute(float x, float y) const;
    void setWaypointFamiliarity(size_t pos, double familiarity);
    void addPoint(float x, float y, bool error);

    size_t size() const{ return m_Waypoints.size(); }

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    std::tuple<float, float, float> operator[](size_t waypoint) const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    GLuint m_WaypointsVAO;
    GLuint m_WaypointsPositionVBO;
    GLuint m_WaypointsColourVBO;

    GLuint m_RouteVAO;
    GLuint m_RoutePositionVBO;
    GLuint m_RouteColourVBO;
    unsigned int m_RouteNumPoints;
    const unsigned int m_RouteMaxPoints;

    std::vector<std::array<float, 2>> m_Waypoints;
    std::vector<float> m_HeadingDegrees;
    std::set<size_t> m_TrainedSnapshots;

    GLuint m_OverlayVAO;
    GLuint m_OverlayPositionVBO;
    GLuint m_OverlayColoursVBO;
};