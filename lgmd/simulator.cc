#include "lgmd_CODE/definitions.h"

// Standard C++ includes
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <vector>

// Standard C includes
#include <cassert>
#include <cstdlib>

// LGMD includes
#include "parameters.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
typedef void (*allocateFn)(unsigned int);

unsigned int get_neuron_index(unsigned int resolution, unsigned int x, unsigned int y)
{
    return x + (y * resolution);
}

unsigned int get_distance_squared(unsigned int xi, unsigned int yi,
                                  unsigned int xj, unsigned int yj)
{
    const unsigned int delta_x = xi - xj;
    const unsigned int delta_y = yi - yj;

    return (delta_x * delta_x) + (delta_y * delta_y);
}

void print_sparse_matrix(unsigned int pre_resolution, const SparseProjection &projection)
{
    const unsigned int pre_size = pre_resolution * pre_resolution;
    for(unsigned int i = 0; i < pre_size; i++)
    {
        std::cout << i << ":";

        for(unsigned int j = projection.indInG[i]; j < projection.indInG[i + 1]; j++)
        {
            std::cout << projection.ind[j] << ",";
        }

        std::cout << std::endl;
    }
}

void build_one_to_one_connection(unsigned int resolution,
                                 SparseProjection &projection, allocateFn allocate)
{
    // Allocate one connection per neuron
    const unsigned int pop_size = resolution * resolution;
    allocate(pop_size);

    for(unsigned int i = 0; i < pop_size; i++)
    {
        projection.indInG[i] = i;
        projection.ind[i] = i;
    }

    projection.indInG[pop_size] = pop_size;
}

void build_centre_to_one_connection(unsigned int pre_resolution, unsigned int centre_size,
                                    SparseProjection &projection, allocateFn allocate)
{
    // Allocate centre_size * centre_size connections
    allocate(centre_size * centre_size);

    // Calculate start and end of border on each row
    const unsigned int border_size = (pre_resolution - centre_size) / 2;
    const unsigned int far_border = pre_resolution - border_size;

    // Loop through rows of pixels in centre
    unsigned int s = 0;
    unsigned int i = 0;
    for(unsigned int yi = 0; yi < pre_resolution; yi++)
    {
        for(unsigned int xi = 0; xi < pre_resolution; xi++)
        {
            projection.indInG[i++] = s;

            // If we're in the centre
            if(xi >= border_size && xi < far_border && yi >= border_size && yi < far_border) {
                projection.ind[s++] = 0;
            }
        }
    }

    // Add ending entry to data structure
    projection.indInG[i] = s;

    // Check
    assert(s == (centre_size * centre_size));
    assert(i == (pre_resolution * pre_resolution));
}

void build_i_s_connections(unsigned int resolution, unsigned int centre_size,
                           SparseProjection &projection1, allocateFn allocate1,
                           SparseProjection &projection2, allocateFn allocate2,
                           SparseProjection &projection4, allocateFn allocate4)
{
    const unsigned int pop_size = resolution * resolution;
    
    // Allocate sparse projections
    allocate1(centre_size * centre_size * 4);
    allocate2(centre_size * centre_size * 4);
    allocate4(centre_size * centre_size * 4);

    // Calculate start and end of border on each row
    const unsigned int border_size = (resolution - centre_size) / 2;
    const unsigned int far_border = resolution - border_size;

    // Create temporary vectors to hold rows
    std::vector<std::vector<unsigned int>> projection1Map(pop_size);
    std::vector<std::vector<unsigned int>> projection2Map(pop_size);
    std::vector<std::vector<unsigned int>> projection4Map(pop_size);

    // Loop through POSTsynaptic neurons
    for(unsigned int xj =  border_size; xj < far_border; xj++)
    {
        for(unsigned int yj = border_size; yj < far_border; yj++)
        {
            const unsigned int j = get_neuron_index(resolution, xj, yj);

            // Add adjacent neighbours to projection 1
            projection1Map[get_neuron_index(resolution, xj - 1, yj)].push_back(j);
            projection1Map[get_neuron_index(resolution, xj, yj - 1)].push_back(j);
            projection1Map[get_neuron_index(resolution, xj + 1, yj)].push_back(j);
            projection1Map[get_neuron_index(resolution, xj, yj + 1)].push_back(j);

            // Add diagonal neighbours to projection 2
            projection2Map[get_neuron_index(resolution, xj - 1, yj - 1)].push_back(j);
            projection2Map[get_neuron_index(resolution, xj + 1, yj - 1)].push_back(j);
            projection2Map[get_neuron_index(resolution, xj + 1, yj + 1)].push_back(j);
            projection2Map[get_neuron_index(resolution, xj - 1, yj + 1)].push_back(j);

            // Add one away neighbours to projection 4
            projection4Map[get_neuron_index(resolution, xj - 2, yj)].push_back(j);
            projection4Map[get_neuron_index(resolution, xj, yj - 2)].push_back(j);
            projection4Map[get_neuron_index(resolution, xj + 2, yj)].push_back(j);
            projection4Map[get_neuron_index(resolution, xj, yj + 2)].push_back(j);
        }
    }

    // Convert vector of vectors to GeNN format
    unsigned int s1 = 0;
    unsigned int s2 = 0;
    unsigned int s4 = 0;
    for(unsigned int i = 0; i < pop_size; i++)
    {
        projection1.indInG[i] = s1;
        projection2.indInG[i] = s2;
        projection4.indInG[i] = s4;

        std::copy(projection1Map[i].cbegin(), projection1Map[i].cend(), &projection1.ind[s1]);
        std::copy(projection2Map[i].cbegin(), projection2Map[i].cend(), &projection2.ind[s2]);
        std::copy(projection4Map[i].cbegin(), projection4Map[i].cend(), &projection4.ind[s4]);

        s1 += projection1Map[i].size();
        s2 += projection2Map[i].size();
        s4 += projection4Map[i].size();
    }

    // Add ending entries to data structure
    projection1.indInG[pop_size] = s1;
    projection2.indInG[pop_size] = s2;
    projection4.indInG[pop_size] = s4;

    // Check
    assert(s1 == (centre_size * centre_size * 4));
    assert(s2 == (centre_size * centre_size * 4));
    assert(s4 == (centre_size * centre_size * 4));
}

unsigned read_p_input(unsigned int output_resolution, unsigned int original_resolution,
                      std::ifstream &stream, std::vector<unsigned int> &indices)
{
    // Calculate resolution scale
    const double scale = (double)original_resolution / (double)output_resolution;

    // Read lines into string
    std::string line;
    std::getline(stream, line);

    if(line.empty()) {
        return std::numeric_limits<unsigned int>::max();
    }

    // Create string stream from line
    std::stringstream lineStream(line);

    // Read time from start of line
    std::string nextTimeString;
    std::getline(lineStream, nextTimeString, ';');
    unsigned int nextTime = (unsigned int)std::stoul(nextTimeString);

    // Clear existing times
    indices.clear();

    while(lineStream.good()) {
        // Read input spike index
        std::string inputIndexString;
        std::getline(lineStream, inputIndexString, ',');
        const int input_index = std::stoi(inputIndexString);

        // Convert this into x and y
        const div_t input_coord = std::div(input_index, original_resolution);

        // Scale into output resolution
        const unsigned int output_x = (unsigned int)std::floor((double)input_coord.quot / scale);
        const unsigned int output_y = (unsigned int)std::floor((double)input_coord.rem / scale);
        assert(output_x < output_resolution);
        assert(output_y < output_resolution);

        // Convert back to index and add to vector
        const unsigned int output_index = (output_x * output_resolution) + output_y;
        indices.push_back(output_index);
    }

    return nextTime;
}
}

int main(int argc, char *argv[])
{
    std::ifstream spikeInput(argv[1]);

    allocateMem();
    initialize();

    build_centre_to_one_connection(Parameters::input_size, Parameters::centre_size,
                                   CP_F_LGMD, &allocateP_F_LGMD);
    build_centre_to_one_connection(Parameters::input_size, Parameters::centre_size,
                                   CS_LGMD, &allocateS_LGMD);
    build_one_to_one_connection(Parameters::input_size,
                                CP_E_S, &allocateP_E_S);
    build_i_s_connections(Parameters::input_size, Parameters::centre_size,
                          CP_I_S_1, &allocateP_I_S_1,
                          CP_I_S_2, &allocateP_I_S_2,
                          CP_I_S_4, &allocateP_I_S_4);

    initlgmd();

    // Read first line of input
    std::vector<unsigned int> inputIndices;
    unsigned int nextInputTime = read_p_input(Parameters::input_size, 128, spikeInput, inputIndices);

    // Open spike output file
    std::ofstream s_spikes("s_spikes.csv");
    s_spikes << "Time [ms], Neuron ID [ms]" << std::endl;

    // Loop through timesteps
    for(unsigned int t = 0; t < 10000; t++)
    {
        // If we should supply input this timestep
        if(nextInputTime == t) {
            // Copy into spike source
            spikeCount_P = inputIndices.size();
            std::copy(inputIndices.cbegin(), inputIndices.cend(), &spike_P[0]);

#ifndef CPU_ONLY
            // Copy to GPU
            pushPCurrentSpikesToDevice();
#endif

            // Read NEXT input
            nextInputTime = read_p_input(Parameters::input_size, 128, spikeInput, inputIndices);
        }

        // Simulate
#ifndef CPU_ONLY
        stepTimeGPU();

        pullLGMDCurrentSpikesFromDevice();
        pullSCurrentSpikesFromDevice();
#else
        stepTimeCPU();
#endif

        for(unsigned int i = 0; i < spikeCount_S; i++) {
            s_spikes << t << "," << spike_S[i] << std::endl;
        }


        if(spikeCount_LGMD > 0) {
            std::cout << "LGMD SPIKE!" << std::endl;
        }
    }


  return 0;
}