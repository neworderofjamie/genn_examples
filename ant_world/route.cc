#include "route.h"

// Standard C++ includes
#include <fstream>
#include <iostream>
#include <limits>
#include <tuple>

// Standard C++ includes
#include <cmath>

// Antworld includes
#include "common.h"

namespace
{
float sqr(float x)
{
    return (x * x);
}
//----------------------------------------------------------------------------
float distanceSquared(float x1, float y1, float x2, float y2)
{
    return sqr(x2 - x1) + sqr(y2 - y1);
}
}
//----------------------------------------------------------------------------
// Route
//----------------------------------------------------------------------------
Route::Route(float arrowLength) : m_RouteVAO(0), m_RoutePositionVBO(0),
    m_OverlayVAO(0), m_OverlayPositionVBO(0), m_OverlayColoursVBO(0)
{
    const GLfloat arrowPositions[] = {
        0.0f, 0.0f,
        0.0f, arrowLength,
    };

    const GLfloat arrowColours[] = {
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f
    };

    // Create a vertex array object to bind everything together
    glGenVertexArrays(1, &m_OverlayVAO);

    // Generate vertex buffer objects for positions and colours
    glGenBuffers(1, &m_OverlayPositionVBO);
    glGenBuffers(1, &m_OverlayColoursVBO);

    // Bind vertex array
    glBindVertexArray(m_OverlayVAO);

    // Bind and upload positions buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_OverlayPositionVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 2 * 2, arrowPositions, GL_STATIC_DRAW);

    // Set vertex pointer to stride over angles and enable client state in VAO
    glVertexPointer(2, GL_FLOAT, 0, BUFFER_OFFSET(0));
    glEnableClientState(GL_VERTEX_ARRAY);

     // Bind and upload colours buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_OverlayColoursVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 2 * 6, arrowColours, GL_STATIC_DRAW);

    // Set colour pointer and enable client state in VAO
    glColorPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(0));
    glEnableClientState(GL_COLOR_ARRAY);
}
//----------------------------------------------------------------------------
Route::Route(float arrowLength, const std::string &filename) : Route(arrowLength)
{
    if(!load(filename)) {
        throw std::runtime_error("Cannot load route");
    }
}
//----------------------------------------------------------------------------
Route::~Route()
{
    // Delete route objects
    glDeleteBuffers(1, &m_RoutePositionVBO);
    glDeleteVertexArrays(1, &m_RouteVAO);

    // Delete overlay objects
    glDeleteBuffers(1, &m_OverlayPositionVBO);
    glDeleteBuffers(1, &m_OverlayColoursVBO);
    glDeleteVertexArrays(1, &m_OverlayVAO);
}
//----------------------------------------------------------------------------
bool Route::load(const std::string &filename)
{
    // Open file for binary IO
    std::ifstream input(filename, std::ios::binary);
    if(!input.good()) {
        std::cerr << "Cannot open route file:" << filename << std::endl;
        return false;
    }

    // Seek to end of file, get size and rewind
    input.seekg(0, std::ios_base::end);
    const std::streampos numPoints = input.tellg() / (sizeof(double) * 3);
    input.seekg(0);
    std::cout << "Route has " << numPoints << " points" << std::endl;

    // Resize route
    m_Route.resize(numPoints);

    // Loop through components(X, Y and heading)
    for(unsigned int c = 0; c < 3; c++) {
        // Loop through points on path
        for(unsigned int i = 0; i < numPoints; i++) {
            // Read point component
            double pointPosition;
            input.read(reinterpret_cast<char*>(&pointPosition), sizeof(double));

            // Convert to float, scale and insert into route
            if(c == 2) {
                m_Route[i][c] = 90.0f - (float)pointPosition;
            }
            else {
                m_Route[i][c] = (float)pointPosition * (1.0f / 100.0f);
            }
        }
    }

    // Create a vertex array object to bind everything together
    glGenVertexArrays(1, &m_RouteVAO);

    // Generate vertex buffer objects for positions
    glGenBuffers(1, &m_RoutePositionVBO);

    // Bind vertex array
    glBindVertexArray(m_RouteVAO);

    // Bind and upload positions buffer
    // **NOTE** we're not actually going to be rendering the 3rd component as it's an angle not a z-coordinate
    glBindBuffer(GL_ARRAY_BUFFER, m_RoutePositionVBO);
    glBufferData(GL_ARRAY_BUFFER, m_Route.size() * sizeof(GLfloat) * 3, m_Route.data(), GL_STATIC_DRAW);

    // Set vertex pointer to stride over angles and enable client state in VAO
    glVertexPointer(2, GL_FLOAT, 3 * sizeof(float), BUFFER_OFFSET(0));
    glEnableClientState(GL_VERTEX_ARRAY);

    return true;
}
//----------------------------------------------------------------------------
void Route::render(float antX, float antY, float antHeading) const
{
    // Bind route VAO
    glBindVertexArray(m_RouteVAO);

    glPushMatrix();
    glTranslatef(0.0f, 0.0f, 0.1f);
    glDrawArrays(GL_LINE_STRIP, 0, m_Route.size());

    glBindVertexArray(m_OverlayVAO);

    glTranslatef(antX, antY, 0.1f);
    glRotatef(-antHeading, 0.0f, 0.0f, 1.0f);
    glDrawArrays(GL_LINES, 0, 2);
    glPopMatrix();
}
//----------------------------------------------------------------------------
std::tuple<float, float, float, float> Route::distanceToRoute(float x, float y) const
{
    // Loop through segments
    float minimumDistanceSquared = std::numeric_limits<float>::max();
    float snapX = 0.0f;
    float snapY = 0.0f;
    float snapHeading = 0.0f;
    for(unsigned int s = 0; s < (m_Route.size() - 1); s++)
    {
        // Get positions of start and end of segment
        const float startX = m_Route[s][0];
        const float startY = m_Route[s][1];
        const float endX = m_Route[s + 1][0];
        const float endY = m_Route[s + 1][1];

        const float segmentLengthSquared = distanceSquared(startX, startY, endX, endY);

        // If segment has no length
        if(segmentLengthSquared == 0) {
            // Calculate distance from point to segment start (arbitrary)
            const float distanceToStartSquared = distanceSquared(startX, startY, x, y);

            // If this is closer than current minimum, update minimum
            if(distanceToStartSquared < minimumDistanceSquared) {
                minimumDistanceSquared = distanceToStartSquared;
                snapX = startX;
                snapY = startY;
                snapHeading = m_Route[s][2];
            }
        }
        else {
            // Calculate dot product of vector from start of segment and vector along segment and clamp
            float t = ((x - startX) * (endX - startY) + (y - startY) * (endY - startY)) / segmentLengthSquared;
            t = std::max(0.0f, std::min(1.0f, t));

            // Use this to project point onto segment
            const float projX = startX + (t * (endX - startX));
            const float projY = startY + (t * (endY - startY));

            // Calculate distance from this point to point
            const float distanceToSegmentSquared = distanceSquared(x, y, projX, projY);

            // If this is closer than current minimum, update minimum
            if(distanceToSegmentSquared < minimumDistanceSquared) {
                minimumDistanceSquared = distanceToSegmentSquared;
                snapX = projX;
                snapY = projY;
                snapHeading = m_Route[s][2];
            }
        }
    }

    return std::make_tuple(sqrt(minimumDistanceSquared), snapX, snapY, snapHeading);
}