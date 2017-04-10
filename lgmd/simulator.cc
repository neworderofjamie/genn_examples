#include "lgmd_CODE/definitions.h"

#include <fstream>

#include "parameters.h"

namespace
{
typedef void (*allocateFn)(unsigned int);

unsigned int get_neuron_index(unsigned int population_size, unsigned int x, unsigned int y)
{
    return x + (y * population_size);
}

unsigned int get_distance_squared(unsigned int xi, unsigned int yi,
                                  unsigned int xj, unsigned int yj)
{
    const unsigned int delta_x = xi - xj;
    const unsigned int delta_y = yi - yj;

    return (delta_x * delta_x) + (delta_y * delta_y);
}

void print_sparse_matrix(unsigned int pre_size, const SparseProjection &projection)
{
    for(unsigned int i = 0; i < (pre_size * pre_size); i++)
    {
        std::cout << i << ":";

        for(unsigned int j = projection.indInG[i]; j < projection.indInG[i + 1]; j++)
        {
            std::cout << projection.ind[j] << ",";
        }

        std::cout << std::endl;
    }
}

void build_one_to_one_connection(unsigned int pop_size,
                                 SparseProjection &projection, allocateFn allocate)
{
    // Allocate one connection per neuron
    allocate(pop_size);

    for(unsigned int i = 0; i < pop_size; i++)
    {
        projection.indInG[i] = i;
        projection.ind[i] = i;
    }

    projection.indInG[pop_size] = pop_size;

}

void build_centre_to_one_connection(unsigned int pre_size, unsigned int centre_size,
                                    SparseProjection &projection, allocateFn allocate)
{
    // Allocate centre_size * centre_size connections
    allocate(centre_size * centre_size);

    // Calculate start and end of border on each row
    const unsigned int border_size = (pre_size - centre_size) / 2;
    const unsigned int far_border = pre_size - border_size;

    // Zero matrix row lengths above centre
    std::fill_n(&projection.indInG[0], pre_size * border_size, 0);

    // Loop through rows of pixels in centre
    unsigned int s = 0;
    for(unsigned int yi = border_size; yi < far_border; yi++)
    {
        // Fill left 'margin' of row
        std::fill_n(&projection.indInG[yi * pre_size], border_size, s);

        // Loop through neuron indices in remainder of row
        for(unsigned int i = get_neuron_index(pre_size, border_size, yi);
            i < get_neuron_index(pre_size, far_border, yi); i++)
        {
            projection.indInG[i] = s;
            projection.ind[s++] = 0;
        }

        // Fill right 'margin' of row
        std::fill_n(&projection.indInG[(yi * pre_size) + far_border], border_size, s);
    }

    // Zero matrix row lengths below centre
    // **NOTE** extra entry to complete Yale structure
    std::fill_n(&projection.indInG[far_border * pre_size], (pre_size * border_size) + 1, s);
}

void build_i_s_connections(unsigned int pop_size, unsigned int centre_size,
                           SparseProjection &projection1, allocateFn allocate1,
                           SparseProjection &projection2, allocateFn allocate2,
                           SparseProjection &projection4, allocateFn allocate4)
{
    // Allocate sparse projections
    allocate1(centre_size * centre_size * 4);
    allocate2(centre_size * centre_size * 4);
    allocate4(centre_size * centre_size * 4);

    // Calculate start and end of border on each row
    const unsigned int border_size = (pop_size - centre_size) / 2;
    const unsigned int far_border = pop_size - border_size;

    // Zero matrix row lengths above centre
    std::fill_n(&projection1.indInG[0], pop_size * border_size, 0);
    std::fill_n(&projection2.indInG[0], pop_size * border_size, 0);
    std::fill_n(&projection4.indInG[0], pop_size * border_size, 0);

    // Loop through rows of pixels in centre
    unsigned int s = 0;
    for(unsigned int yi = border_size; yi < far_border; yi++)
    {
        // Fill left 'margin' of row
        std::fill_n(&projection1.indInG[yi * pop_size], border_size, s);
        std::fill_n(&projection2.indInG[yi * pop_size], border_size, s);
        std::fill_n(&projection4.indInG[yi * pop_size], border_size, s);

        // Loop through neuron indices in remainder of row
        for(unsigned int xi = border_size; xi < far_border; xi++)
        {
            const unsigned int i = get_neuron_index(pop_size, xi, yi);

            projection1.indInG[i] = s;
            projection2.indInG[i] = s;
            projection4.indInG[i] = s;

            projection1.ind[s] = get_neuron_index(pop_size, xi - 1, yi);
            projection1.ind[s + 1] = get_neuron_index(pop_size, xi, yi - 1);
            projection1.ind[s + 2] = get_neuron_index(pop_size, xi + 1, yi);
            projection1.ind[s + 3] = get_neuron_index(pop_size, xi, yi + 1);

            projection2.ind[s] = get_neuron_index(pop_size, xi - 1, yi - 1);
            projection2.ind[s + 1] = get_neuron_index(pop_size, xi + 1, yi - 1);
            projection2.ind[s + 2] = get_neuron_index(pop_size, xi + 1, yi + 1);
            projection2.ind[s + 3] = get_neuron_index(pop_size, xi - 1, yi + 1);

            projection4.ind[s] = get_neuron_index(pop_size, xi - 2, yi);
            projection4.ind[s + 1] = get_neuron_index(pop_size, xi, yi - 2);
            projection4.ind[s + 2] = get_neuron_index(pop_size, xi + 2, yi);
            projection4.ind[s + 3] = get_neuron_index(pop_size, xi, yi + 2);

            // Update s
            s += 4;
        }

        // Fill right 'margin' of row
        std::fill_n(&projection1.indInG[(yi * pop_size) + far_border], border_size, s);
        std::fill_n(&projection2.indInG[(yi * pop_size) + far_border], border_size, s);
        std::fill_n(&projection4.indInG[(yi * pop_size) + far_border], border_size, s);
    }

    // Zero matrix row lengths below centre
    // **NOTE** extra entry to complete Yale structure
    std::fill_n(&projection1.indInG[far_border * pop_size], (pop_size * border_size) + 1, s);
    std::fill_n(&projection2.indInG[far_border * pop_size], (pop_size * border_size) + 1, s);
    std::fill_n(&projection4.indInG[far_border * pop_size], (pop_size * border_size) + 1, s);


}
/*void read_p_input(std::ifstream &stream)
{
    // Read time from start of line
    std::string time;
    std::getline(stream, time, ';');


}*/
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


    // Open CSV output files
    FILE *spikes = fopen("spikes.csv", "w");
    fprintf(spikes, "Time(ms), Neuron ID\n");

    // Loop through timesteps
    for(unsigned int t = 0; t < 10000; t++)
    {
        // Simulate
#ifndef CPU_ONLY
        stepTimeGPU();

        //pullECurrentSpikesFromDevice();
        //pullIEStateFromDevice();
#else
        stepTimeCPU();
#endif

        // Write spike times to file
        /*for(unsigned int i = 0; i < glbSpkCntE[0]; i++)
        {
        fprintf(spikes, "%f, %u\n", 1.0 * (double)t, glbSpkE[i]);
        }

        // Calculate mean IE weights
        float totalWeight = std::accumulate(&gIE[0], &gIE[CIE.connN], 0.0f);
        fprintf(weights, "%f, %f\n", 1.0 * (double)t, totalWeight / (double)CIE.connN);*/

    }

    // Close files
    fclose(spikes);

  return 0;
}