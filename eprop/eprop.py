from pygenn import genn_model

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------
adam_optimizer_model = genn_model.create_custom_custom_update_class(
    "adam_optimizer",
    param_names=["beta1", "beta2", "epsilon"],
    var_name_types=[("m", "scalar"), ("v", "scalar")],
    extra_global_params=[("alpha", "scalar"), ("firstMomentScale", "scalar"),
                         ("secondMomentScale", "scalar")],
    var_refs=[("gradient", "scalar"), ("variable", "scalar")],
    update_code="""
    // Update biased first moment estimate
    $(m) = ($(beta1) * $(m)) + ((1.0 - $(beta1)) * $(gradient));
    // Update biased second moment estimate
    $(v) = ($(beta2) * $(v)) + ((1.0 - $(beta2)) * $(gradient) * $(gradient));
    // Add gradient to variable, scaled by learning rate
    $(variable) -= ($(alpha) * $(m) * $(firstMomentScale)) / (sqrt($(v) * $(secondMomentScale)) + $(epsilon));
    // Zero gradient
    $(gradient) = 0.0;
    """)

#----------------------------------------------------------------------------
# Neuron models
#----------------------------------------------------------------------------
recurrent_lif_model = genn_model.create_custom_neuron_class(
    "recurrent_lif",
    param_names=["TauM", "Vthresh", "TauRefrac"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("E", "scalar")],
    additional_input_vars=[("ISynFeedback", "scalar", 0.0)],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
   
    sim_code="""
    $(E) = $(IsynFeedback);
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

recurrent_alif_model = genn_model.create_custom_neuron_class(
    "recurrent_alif",
    param_names=["TauM", "TauAdap", "Vthresh", "TauRefrac", "Beta"],
    var_name_types=[("V", "scalar"), ("A", "scalar"), ("RefracTime", "scalar"), ("E", "scalar")],
    additional_input_vars=[("ISynFeedback", "scalar", 0.0)],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("Rho", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))())],
   
    sim_code="""
    $(E) = $(IsynFeedback);
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

#----------------------------------------------------------------------------
# Weight update models
#----------------------------------------------------------------------------
feedback_model = genn_model.create_custom_weight_update_class(
    "feedback",
    var_name_types=[("g", "scalar")],
    synapse_dynamics_code="""
    $(addToInSyn, $(g) * $(E_pre));
    """)

eprop_lif_model = genn_model.create_custom_weight_update_class(
    "eprop_lif",
    param_names=["TauE", "CReg", "FTarget", "TauFAvg"],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("FTargetTimestep", genn_model.create_dpf_class(lambda pars, dt: (pars[2] * dt) / 1000.0)()),
                    ("AlphaFAv", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[3]))())],
    var_name_types=[("g", "scalar"), ("eFiltered", "scalar"), ("DeltaG", "scalar")],
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

eprop_alif_model = genn_model.create_custom_weight_update_class(
    "eprop_alif",
    param_names=["TauE", "TauA", "CReg", "FTarget", "TauFAvg", "Beta"],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("Rho", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))()),
                    ("FTargetTimestep", genn_model.create_dpf_class(lambda pars, dt: (pars[3] * dt) / 1000.0)()),
                    ("AlphaFAv", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[4]))())],
    var_name_types=[("g", "scalar"), ("eFiltered", "scalar"), {"epsilonA", "scalar"}, ("DeltaG", "scalar")],
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

output_learning_model = genn_model.create_custom_weight_update_class(
    "output_learning",
    param_names=["TauE"],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
    var_name_types=[("g", "scalar"), ("DeltaG", "scalar")],
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

#----------------------------------------------------------------------------
# Postsynaptic models
#----------------------------------------------------------------------------
feedback_psm_model = genn_model.create_custom_postsynaptic_class(
    "feedback_psm",
    apply_input_code="""
    $(ISynFeedback) += $(inSyn);
    $(inSyn) = 0;
    """)
