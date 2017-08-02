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
    Route(float arrowLength, const std::string &filename);
    ~Route();

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool load(const std::string &filename);
    void render(float antX, float antY, float antHeading) const;

    bool atDestination(float x, float y, float threshold) const;
    std::tuple<float, size_t> getDistanceToRoute(float x, float y) const;
    std::tuple<float, float, float> getNextSnapshotPosition(size_t segment) const;

    void markTrainedSnapshot(size_t index){ m_TrainedSnapshots.insert(index); }
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
    std::set<size_t> m_TrainedSnapshots;

    GLuint m_OverlayVAO;
    GLuint m_OverlayPositionVBO;
    GLuint m_OverlayColoursVBO;
};