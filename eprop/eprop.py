import numpy as np

from pygenn import genn_model
from pygenn.genn_wrapper.Models import (VarAccess_READ_ONLY,
                                        VarAccessMode_READ_ONLY, 
                                        VarAccess_REDUCE_BATCH_SUM)

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------
adam_optimizer_model = genn_model.create_custom_custom_update_class(
    "adam_optimizer",
    param_names=["beta1", "beta2", "epsilon"],
    var_name_types=[("m", "scalar"), ("v", "scalar")],
    extra_global_params=[("alpha", "scalar"), ("firstMomentScale", "scalar"),
                         ("secondMomentScale", "scalar")],
    var_refs=[("gradient", "scalar", VarAccessMode_READ_ONLY), ("variable", "scalar")],
    update_code="""
    // Update biased first moment estimate
    $(m) = ($(beta1) * $(m)) + ((1.0 - $(beta1)) * $(gradient));
    // Update biased second moment estimate
    $(v) = ($(beta2) * $(v)) + ((1.0 - $(beta2)) * $(gradient) * $(gradient));
    // Add gradient to variable, scaled by learning rate
    $(variable) -= ($(alpha) * $(m) * $(firstMomentScale)) / (sqrt($(v) * $(secondMomentScale)) + $(epsilon));
    """)

gradient_batch_reduce_model = genn_model.create_custom_custom_update_class(
    "gradient_batch_reduce",
    var_name_types=[("reducedGradient", "scalar", VarAccess_REDUCE_BATCH_SUM)],
    var_refs=[("gradient", "scalar")],
    update_code="""
    $(reducedGradient) = $(gradient);
    $(gradient) = 0;
    """)

#----------------------------------------------------------------------------
# Neuron models
#----------------------------------------------------------------------------
recurrent_alif_model = genn_model.create_custom_neuron_class(
    "recurrent_alif",
    param_names=["TauM", "TauAdap", "Vthresh", "TauRefrac", "Beta"],
    var_name_types=[("V", "scalar"), ("A", "scalar"), ("RefracTime", "scalar"), ("E", "scalar")],
    additional_input_vars=[("ISynFeedback", "scalar", 0.0)],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("Rho", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))())],

    sim_code="""
    $(E) = $(ISynFeedback);
    $(V) = ($(Alpha) * $(V)) + $(Isyn);
    $(A) *= $(Rho);
    if ($(RefracTime) > 0.0) {
      $(RefracTime) -= DT;
    }
    """,
    reset_code="""
    $(RefracTime) = $(TauRefrac);
    $(V) -= $(Vthresh);
    $(A) += 1.0;
    """,
    threshold_condition_code="""
    $(RefracTime) <= 0.0 && $(V) >= ($(Vthresh) + ($(Beta) * $(A)))
    """,
    is_auto_refractory_required=False)

recurrent_lif_model = genn_model.create_custom_neuron_class(
    "recurrent_lif",
    param_names=["TauM", "Vthresh", "TauRefrac"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("E", "scalar")],
    additional_input_vars=[("ISynFeedback", "scalar", 0.0)],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
   
    sim_code="""
    $(E) = $(ISynFeedback);
    $(V) = ($(Alpha) * $(V)) + $(Isyn);
    if ($(RefracTime) > 0.0) {
      $(RefracTime) -= DT;
    }
    """,
    reset_code="""
    $(RefracTime) = $(TauRefrac);
    $(V) -= $(Vthresh);
    """,
    threshold_condition_code="""
    $(RefracTime) <= 0.0 && $(V) >= $(Vthresh)
    """,
    is_auto_refractory_required=False)

# **TODO** helper function to generate these models for arbitrary number of output neurons
output_classification_model_16 = genn_model.create_custom_neuron_class(
    "output_classification_16",
    param_names=["TauOut", "TrialTime"],
    var_name_types=[("Y", "scalar"), ("Pi", "scalar"), ("E", "scalar"), ("B", "scalar", VarAccess_READ_ONLY), ("DeltaB", "scalar")],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
    extra_global_params=[("labels", "uint8_t*")],

    sim_code="""
    $(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(B);
    scalar m = $(Y);
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x1));
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x2));
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x4));
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x8));
    const scalar expPi = exp($(Y) - m);
    scalar sumExpPi = expPi;
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x1);
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x2);
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x4);
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x8);
    $(Pi) = expPi / sumExpPi;

    const scalar piStar = ($(id) == $(labels)[$(batch)]) ? 1.0 : 0.0;
    $(E) = $(Pi) - piStar;

    $(DeltaB) += $(E);
    """,
    is_auto_refractory_required=False)

output_classification_model_32 = genn_model.create_custom_neuron_class(
    "output_classification_32",
    param_names=["TauOut", "TrialTime"],
    var_name_types=[("Y", "scalar"), ("Pi", "scalar"), ("E", "scalar"), ("B", "scalar", VarAccess_READ_ONLY), ("DeltaB", "scalar")],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
    extra_global_params=[("labels", "uint8_t*")],

    sim_code="""
    $(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(B);
    scalar m = $(Y);
    m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x1));
    m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x2));
    m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x4));
    m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x8));
    m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x10));
    const scalar expPi = exp($(Y) - m);
    scalar sumExpPi = expPi;
    sumExpPi +=  __shfl_xor_sync(0xFFFFFFFF, sumExpPi, 0x1);
    sumExpPi +=  __shfl_xor_sync(0xFFFFFFFF, sumExpPi, 0x2);
    sumExpPi +=  __shfl_xor_sync(0xFFFFFFFF, sumExpPi, 0x4);
    sumExpPi +=  __shfl_xor_sync(0xFFFFFFFF, sumExpPi, 0x8);
    sumExpPi +=  __shfl_xor_sync(0xFFFFFFFF, sumExpPi, 0x10);
    $(Pi) = expPi / sumExpPi;

    const scalar piStar = ($(id) == $(labels)[$(batch)]) ? 1.0 : 0.0;
    $(E) = $(Pi) - piStar;

    $(DeltaB) += $(E);
    """,
    is_auto_refractory_required=False)

#----------------------------------------------------------------------------
# Weight update models
#----------------------------------------------------------------------------
feedback_model = genn_model.create_custom_weight_update_class(
    "feedback",
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY)],
    synapse_dynamics_code="""
    $(addToInSyn, $(g) * $(E_pre));
    """)

eprop_alif_model = genn_model.create_custom_weight_update_class(
    "eprop_alif",
    param_names=["TauE", "TauA", "CReg", "FTarget", "TauFAvg", "Beta"],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("Rho", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))()),
                    ("FTargetTimestep", genn_model.create_dpf_class(lambda pars, dt: (pars[3] * dt) / 1000.0)()),
                    ("AlphaFAv", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[4]))())],
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY), ("eFiltered", "scalar"), ("epsilonA", "scalar"), ("DeltaG", "scalar")],
    pre_var_name_types=[("ZFilter", "scalar")],
    post_var_name_types=[("Psi", "scalar"), ("FAvg", "scalar")],
    
    sim_code="""
    $(addToInSyn, $(g));
    """,

    pre_spike_code="""
    $(ZFilter) += 1.0;
    """,
    pre_dynamics_code="""
    $(ZFilter) *= $(Alpha);
    """,

    post_spike_code="""
    $(FAvg) += (1.0 - $(AlphaFAv));
    """,
    post_dynamics_code="""
    $(FAvg) *= $(AlphaFAv);
    if ($(RefracTime_post) > 0.0) {
      $(Psi) = 0.0;
    }
    else {
      $(Psi) = (1.0 / $(Vthresh_post)) * 0.3 * fmax(0.0, 1.0 - fabs(($(V_post) - ($(Vthresh_post) + ($(Beta_post) * $(A_post)))) / $(Vthresh_post)));
    }
    """,

    synapse_dynamics_code="""
    // Calculate some common factors in e and epsilon update
    scalar epsilonA = $(epsilonA);
    const scalar psiZFilter = $(Psi) * $(ZFilter);
    const scalar psiBetaEpsilonA = $(Psi) * $(Beta) * epsilonA;
    
    // Calculate e and episilonA
    const scalar e = psiZFilter  - psiBetaEpsilonA;
    $(epsilonA) = psiZFilter + (($(Rho) * epsilonA) - psiBetaEpsilonA);
    
    // Calculate filtered version of eligibility trace
    scalar eFiltered = $(eFiltered);
    eFiltered = (eFiltered * $(Alpha)) + e;
    
    // Apply weight update
    $(DeltaG) += (eFiltered * $(E_post)) + (($(FAvg) - $(FTargetTimestep)) * $(CReg) * e);
    $(eFiltered) = eFiltered;
    """)

eprop_lif_model = genn_model.create_custom_weight_update_class(
    "eprop_lif",
    param_names=["TauE", "CReg", "FTarget", "TauFAvg"],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("FTargetTimestep", genn_model.create_dpf_class(lambda pars, dt: (pars[2] * dt) / 1000.0)()),
                    ("AlphaFAv", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[3]))())],
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY), ("eFiltered", "scalar"), ("DeltaG", "scalar")],
    pre_var_name_types=[("ZFilter", "scalar")],
    post_var_name_types=[("Psi", "scalar"), ("FAvg", "scalar")],
    
    sim_code="""
    $(addToInSyn, $(g));
    """,

    pre_spike_code="""
    $(ZFilter) += 1.0;
    """,
    pre_dynamics_code="""
    $(ZFilter) *= $(Alpha);
    """,

    post_spike_code="""
    $(FAvg) += (1.0 - $(AlphaFAv));
    """,
    post_dynamics_code="""
    $(FAvg) *= $(AlphaFAv);
    if ($(RefracTime_post) > 0.0) {
      $(Psi) = 0.0;
    }
    else {
      $(Psi) = (1.0 / $(Vthresh_post)) * 0.3 * fmax(0.0, 1.0 - fabs(($(V_post) - $(Vthresh_post)) / $(Vthresh_post)));
    }
    """,

    synapse_dynamics_code="""
    const scalar e = $(ZFilter) * $(Psi);
    scalar eFiltered = $(eFiltered);
    eFiltered = (eFiltered * $(Alpha)) + e;
    $(DeltaG) += (eFiltered * $(E_post)) + (($(FAvg) - $(FTargetTimestep)) * $(CReg) * e);
    $(eFiltered) = eFiltered;
    """)

output_learning_model = genn_model.create_custom_weight_update_class(
    "output_learning",
    param_names=["TauE"],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY), ("DeltaG", "scalar")],
    pre_var_name_types=[("ZFilter", "scalar")],

    sim_code="""
    $(addToInSyn, $(g));
    """,

    pre_spike_code="""
    $(ZFilter) += 1.0;
    """,
    pre_dynamics_code="""
    $(ZFilter) *= $(Alpha);
    """,

    synapse_dynamics_code="""
    $(DeltaG) += $(ZFilter) * $(E_post);
    """)
