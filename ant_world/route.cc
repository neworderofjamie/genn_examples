#include "route.h"

// Standard C++ includes
#include <fstream>
#include <iostream>
#include <limits>
#include <tuple>

// Standard C++ includes
#include <cassert>
#include <cmath>

// Antworld includes
#include "common.h"

// //----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
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
}   // Anonymous namespace

//----------------------------------------------------------------------------
// Route
//----------------------------------------------------------------------------
Route::Route(float arrowLength, unsigned int maxRouteEntries)
    : m_WaypointsVAO(0), m_WaypointsPositionVBO(0), m_WaypointsColourVBO(0),
    m_RouteVAO(0), m_RoutePositionVBO(0), m_RouteColourVBO(0), m_RouteNumPoints(0), m_RouteMaxPoints(maxRouteEntries),
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


    // Create a vertex array object to bind everything together
    glGenVertexArrays(1, &m_RouteVAO);

    // Generate vertex buffer objects for positions and colours
    glGenBuffers(1, &m_RoutePositionVBO);
    glGenBuffers(1, &m_RouteColourVBO);

    // Bind vertex array
    glBindVertexArray(m_RouteVAO);

    // Bind and upload positions buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_RoutePositionVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 2 * maxRouteEntries, nullptr, GL_DYNAMIC_DRAW);

    // Set vertex pointer to stride over angles and enable client state in VAO
    glVertexPointer(2, GL_FLOAT, 0, BUFFER_OFFSET(0));
    glEnableClientState(GL_VERTEX_ARRAY);

     // Bind and upload colours buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_RouteColourVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uint8_t) * 2 * maxRouteEntries, nullptr, GL_DYNAMIC_DRAW);

    // Set colour pointer and enable client state in VAO
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, BUFFER_OFFSET(0));
    glEnableClientState(GL_COLOR_ARRAY);
}
//----------------------------------------------------------------------------
Route::Route(float arrowLength, unsigned int maxRouteEntries, const std::string &filename)
    : Route(arrowLength, maxRouteEntries)
{
    if(!load(filename)) {
        throw std::runtime_error("Cannot load route");
    }
}
//----------------------------------------------------------------------------
Route::~Route()
{
    // Delete waypoint objects
    glDeleteBuffers(1, &m_WaypointsPositionVBO);
    glDeleteVertexArrays(1, &m_WaypointsColourVBO);
    glDeleteVertexArrays(1, &m_WaypointsVAO);

    // Delete route objects
    glDeleteBuffers(1, &m_RoutePositionVBO);
    glDeleteVertexArrays(1, &m_RouteColourVBO);
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

    {
        // Loop through components(X and Y, ignoring heading)
        std::vector<std::array<float, 2>> fullRoute(numPoints);
        for(unsigned int c = 0; c < 2; c++) {
            // Loop through points on path
            for(unsigned int i = 0; i < numPoints; i++) {
                // Read point component
                double pointPosition;
                input.read(reinterpret_cast<char*>(&pointPosition), sizeof(double));

                // Convert to float, scale to metres and insert into route
                fullRoute[i][c] = (float)pointPosition * (1.0f / 100.0f);
            }
        }

        // Reserve correctly sized vector for waypoints
        m_Waypoints.reserve((numPoints / 10) + 1);

        // Loop through every 10 path points and add waypoint
        for(unsigned int i = 0; i < numPoints; i += 10)
        {
            m_Waypoints.push_back(fullRoute[i]);
        }
    }

    // Reserve headings
    const unsigned int numSegments = m_Waypoints.size() - 1;
    m_HeadingDegrees.reserve(numSegments);

    // Loop through route segments
    for(unsigned int i = 0; i < numSegments; i++) {
        // Get waypoints at start and end of segment
        const auto &segmentStart = m_Waypoints[i];
        const auto &segmentEnd = m_Waypoints[i + 1];

        // Calculate segment heading
        const float headingDegrees = radiansToDegrees * atan2(segmentStart[1] - segmentEnd[1],
                                                              segmentEnd[0] - segmentStart[0]);

        // Round to nearest whole number and add to headings array
        m_HeadingDegrees.push_back(round(headingDegrees * 0.5f) * 2.0f);
    }

    // Loop through waypoints other than first
    for(unsigned int i = 1; i < m_Waypoints.size(); i++)
    {
        // Get previous and current waypoiny
        const auto &prevWaypoint = m_Waypoints[i - 1];
        auto &waypoint = m_Waypoints[i];

        // Convert the segment heading back to radians
        const float headingRadians = degreesToRadians * m_HeadingDegrees[i - 1];

        // Realign segment to this angle
        waypoint[0] = prevWaypoint[0] + (0.1f * cos(headingRadians));
        waypoint[1] = prevWaypoint[1] - (0.1f * sin(headingRadians));
    }

    // Create a vertex array object to bind everything together
    glGenVertexArrays(1, &m_WaypointsVAO);

    // Generate vertex buffer objects for positions and colours
    glGenBuffers(1, &m_WaypointsPositionVBO);
    glGenBuffers(1, &m_WaypointsColourVBO);

    // Bind vertex array
    glBindVertexArray(m_WaypointsVAO);

    // Bind and upload positions buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_WaypointsPositionVBO);
    glBufferData(GL_ARRAY_BUFFER, m_Waypoints.size() * sizeof(GLfloat) * 2, m_Waypoints.data(), GL_STATIC_DRAW);

    // Set vertex pointer and enable client state in VAO
    glVertexPointer(2, GL_FLOAT, 0, BUFFER_OFFSET(0));
    glEnableClientState(GL_VERTEX_ARRAY);

    {
        // Bind and upload zeros to colour buffer
        std::vector<uint8_t> colours(m_Waypoints.size() * 3, 0);
        glBindBuffer(GL_ARRAY_BUFFER, m_WaypointsColourVBO);
        glBufferData(GL_ARRAY_BUFFER, m_Waypoints.size() * sizeof(uint8_t) * 3, colours.data(), GL_DYNAMIC_DRAW);

        // Set colour pointer and enable client state in VAO
        glColorPointer(3, GL_UNSIGNED_BYTE, 0, BUFFER_OFFSET(0));
        glEnableClientState(GL_COLOR_ARRAY);
    }
    return true;
}
//----------------------------------------------------------------------------
void Route::render(float antX, float antY, float antHeading) const
{
    // Bind route VAO
    glBindVertexArray(m_WaypointsVAO);

    glPushMatrix();
    glTranslatef(0.0f, 0.0f, 0.1f);
    glDrawArrays(GL_POINTS, 0, m_Waypoints.size());

    // If there are any route points, bind
    if(m_RouteNumPoints > 0) {
        glBindVertexArray(m_RouteVAO);

        glDrawArrays(GL_LINE_STRIP, 0, m_RouteNumPoints);
    }

    glBindVertexArray(m_OverlayVAO);

    glTranslatef(antX, antY, 0.1f);
    glRotatef(-antHeading, 0.0f, 0.0f, 1.0f);
    glDrawArrays(GL_LINES, 0, 2);
    glPopMatrix();

}
//----------------------------------------------------------------------------
bool Route::atDestination(float x, float y, float threshold) const
{
    // If route's empty, there is no destination so return false
    if(m_Waypoints.empty()) {
        return false;
    }
    // Otherwise return true if
    else {
        return (distanceSquared(x, y, m_Waypoints.back()[0], m_Waypoints.back()[1]) < sqr(threshold));
    }
}
//----------------------------------------------------------------------------
std::tuple<float, size_t> Route::getDistanceToRoute(float x, float y) const
{
    // Loop through segments
    float minimumDistanceSquared = std::numeric_limits<float>::max();
    size_t nearestWaypoint;
    for(unsigned int s = 0; s < m_Waypoints.size(); s++)
    {
        const float distanceToWaypointSquared = distanceSquared(x, y, m_Waypoints[s][0], m_Waypoints[s][1]);

        // If this is closer than current minimum, update minimum and nearest waypoint
        if(distanceToWaypointSquared < minimumDistanceSquared) {
            minimumDistanceSquared = distanceToWaypointSquared;
            nearestWaypoint = s;
        }
    }

    // Return the minimum distance to the path and the segment in which this occured
    return std::make_tuple(sqrt(minimumDistanceSquared), nearestWaypoint);
}
//----------------------------------------------------------------------------
void Route::setWaypointFamiliarity(size_t pos, double familiarity)
{
    // Convert familiarity to a grayscale colour
    const uint8_t intensity = (uint8_t)std::min(255.0, std::max(0.0, std::round(255.0 * familiarity)));
    const uint8_t colour[3] = {intensity, intensity, intensity};

    // Update this positions colour in colour buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_WaypointsColourVBO);
    glBufferSubData(GL_ARRAY_BUFFER, pos * sizeof(uint8_t) * 3, sizeof(uint8_t) * 3, colour);

}
//----------------------------------------------------------------------------
void Route::addPoint(float x, float y, bool error)
{
    const static uint8_t errorColour[3] = {0xFF, 0, 0};
    const static uint8_t correctColour[3] = {0, 0xFF, 0};

    const float position[2] = {x, y};

    // Update this positions colour in colour buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_RouteColourVBO);
    glBufferSubData(GL_ARRAY_BUFFER, m_RouteNumPoints * sizeof(uint8_t) * 3,
                    sizeof(uint8_t) * 3, error ? errorColour : correctColour);

    // Update this positions colour in colour buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_RoutePositionVBO);
    glBufferSubData(GL_ARRAY_BUFFER, m_RouteNumPoints * sizeof(float) * 2,
                    sizeof(float) * 2, position);

    m_RouteNumPoints++;
}
//----------------------------------------------------------------------------
std::tuple<float, float, float> Route::operator[](size_t waypoint) const
{
    const float x = m_Waypoints[waypoint][0];
    const float y = m_Waypoints[waypoint][1];

    // If this isn't the last waypoint, return the heading of the segment from this waypoint
    if(waypoint < m_HeadingDegrees.size()) {
        return std::make_tuple(x, y, 90.0f + m_HeadingDegrees[waypoint]);
    }
    else {
        return std::make_tuple(x, y, 0.0f);
    }
}