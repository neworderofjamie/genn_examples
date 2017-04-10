#include <cmath>
#include <vector>

// GeNN includes
#include "modelSpec.h"

// Common example includes
#include "../common/exp_curr.h"
#include "../common/lif.h"

// LGMD includes
#include "parameters.h"

// Anonymous namespace
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
    model.setDT(Parameters::timestep);
    model.setName("lgmd");

    std::cout << "Convergent strength:" << Parameters::convergent_strength << std::endl;

    // Convert persistences to taus
    const double scale = 10.0;
    const double tau_s = persistance_to_tc(Parameters::timestep, Parameters::persistance_s) * scale;
    const double tau_i = persistance_to_tc(Parameters::timestep, Parameters::persistance_i) * scale;
    const double tau_e = persistance_to_tc(Parameters::timestep, Parameters::persistance_e) * scale;
    const double tau_lgmd = persistance_to_tc(Parameters::timestep, Parameters::persistance_lgmd) * scale;
    const double tau_f = persistance_to_tc(Parameters::timestep, Parameters::persistance_f) * scale;

    std::cout << "Tau s:" << tau_s << ",Tau i:" << tau_i << ", Tau e:" << tau_e << ", Tau lgmd:" << tau_lgmd << ", Tau_f:" << tau_f << std::endl;

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters for P population
    LIF::ParamValues s_lif_params(
        1.0,        // 0 - C
        tau_s,      // 1 - TauM
        0.0,        // 2 - Vrest
        0.0,        // 3 - Vreset
        0.5,        // 4 - Vthresh
        0.0,        // 5 - Ioffset
        1.0);       // 6 - TauRefrac

    // LIF model parameters for LGMD population
    LIF::ParamValues lgmd_lif_params(
        1.0,        // 0 - C
        tau_lgmd,   // 1 - TauM
        0.0,        // 2 - Vrest
        0.0,        // 3 - Vreset
        0.25,       // 4 - Vthresh
        0.0,        // 5 - Ioffset
        1.0);       // 6 - TauRefrac

    // LIF initial conditions
    LIF::VarValues lif_init(
        0.0,        // 0 - V
        0.0);       // 1 - RefracTime

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues p_f_lgmd_static_syn_init(
        -Parameters::convergent_strength * 5.0 * 0.2);      // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues s_lgmd_static_syn_init(
        Parameters::convergent_strength * 2.0 * 4.0);     // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues p_e_s_static_syn_init(
        0.6 * 2.0);     // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues p_i_s_1_static_syn_init(
        Parameters::i_s_weight_1);     // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues p_i_s_2_static_syn_init(
        Parameters::i_s_weight_2);     // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues p_i_s_4_static_syn_init(
        Parameters::i_s_weight_4);     // 0 - Wij (nA)

    // Exponential current parameters
    ExpCurr::ParamValues p_f_lgmd_exp_curr_params(
        tau_f);         // 0 - TauSyn (ms)

    ExpCurr::ParamValues s_lgmd_exp_curr_params(
        4.0);       // 0 - TauSyn (ms)

    ExpCurr::ParamValues p_e_s_exp_curr_params(
        tau_e);       // 0 - TauSyn (ms)

    ExpCurr::ParamValues p_i_s_exp_curr_params(
        tau_i);       // 0 - TauSyn (ms)

    //------------------------------------------------------------------------
    // Neuron populations
    //------------------------------------------------------------------------
    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::SpikeSource>("P", Parameters::input_size * Parameters::input_size,
                                                         {}, {});
    model.addNeuronPopulation<LIF>("S", Parameters::input_size * Parameters::input_size,
                                   s_lif_params, lif_init);

    model.addNeuronPopulation<LIF>("LGMD", 1,
                                   lgmd_lif_params, lif_init);

    //------------------------------------------------------------------------
    // Synapse populations
    //------------------------------------------------------------------------
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "P_F_LGMD", SynapseMatrixType::SPARSE_GLOBALG, 3,
        "P", "LGMD",
        {}, p_f_lgmd_static_syn_init,
        p_f_lgmd_exp_curr_params, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "S_LGMD", SynapseMatrixType::SPARSE_GLOBALG, 1,
        "S", "LGMD",
        {}, s_lgmd_static_syn_init,
        s_lgmd_exp_curr_params, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "P_E_S", SynapseMatrixType::SPARSE_GLOBALG, 2,
        "P", "S",
        {}, p_e_s_static_syn_init,
        p_e_s_exp_curr_params, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "P_I_S_1", SynapseMatrixType::SPARSE_GLOBALG, Parameters::i_s_delay_1,
        "P", "S",
        {}, p_i_s_1_static_syn_init,
        p_i_s_exp_curr_params, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "P_I_S_2", SynapseMatrixType::SPARSE_GLOBALG, Parameters::i_s_delay_2,
        "P", "S",
        {}, p_i_s_2_static_syn_init,
        p_i_s_exp_curr_params, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "P_I_S_4", SynapseMatrixType::SPARSE_GLOBALG, Parameters::i_s_delay_4,
        "P", "S",
        {}, p_i_s_4_static_syn_init,
        p_i_s_exp_curr_params, {});


    /*model.setSpanTypeToPre("EE");
    model.setSpanTypeToPre("EI");
    model.setSpanTypeToPre("II");
    model.setSpanTypeToPre("IE");*/

    // Use zero-copy for spikes and weights as we want to record them every timestep
    //e->setSpikeZeroCopyEnabled(true);
    //ie->setWUVarZeroCopyEnabled("g", true);

    model.finalize();
}