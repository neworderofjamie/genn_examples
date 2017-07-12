#include <cmath>
#include <vector>

// GeNN includes
#include "modelSpec.h"

// Common example includes
#include "../common/opencv_lif.h"

//------------------------------------------------------------------------
// Anonymous namespace
//------------------------------------------------------------------------
namespace
{
double persistance_to_tc(double timestep, double persistance)
{
    return -timestep / std::log(persistance);
}
}

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(1.0);
    model.setName("opencv");

    // Convert persistences to taus
    const double scale = 10.0;
    const double tau_p = persistance_to_tc(1.0, 0.4) * scale;
    
    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters for P population
    OpenCVLIF::ParamValues p_lif_params(
        2.0,        // 0 - C
        tau_p,      // 1 - TauM
        0.0,        // 2 - Vrest
        0.0,        // 3 - Vreset
        0.3,        // 4 - Vthresh
        0.0,        // 5 - Ioffset
        1.0,        // 6 - TauRefrac
        32          // 7 - Resolution
    );

    // LIF initial conditions
    OpenCVLIF::VarValues lif_init(
        -60.0,        // 0 - V
        0.0);       // 1 - RefracTime

    //------------------------------------------------------------------------
    // Neuron populations
    //------------------------------------------------------------------------
    // Create IF_curr neuron
    model.addNeuronPopulation<OpenCVLIF>("P", 32 * 32, p_lif_params, lif_init);
    
    model.finalize();
}