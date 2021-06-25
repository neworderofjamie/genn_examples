import numpy as np
import matplotlib.pyplot as plt
import time
from os import path
from struct import unpack

from itertools import combinations
from pygenn import genn_model, genn_wrapper

import sys

def get_image_data(raw_filename, filename, correct_magic):
    if path.exists(filename):
        print("Loading existing data")
        return np.load(filename)
    else:
        with open(raw_filename, "rb") as f:
            image_data = f.read()
            
            # Unpack header from first 16 bytes of buffer
            magic, num_items, num_rows, num_cols = unpack('>IIII', image_data[:16])
            assert magic == correct_magic
            assert num_rows == 28
            assert num_cols == 28

            # Convert remainder of buffer to numpy bytes
            image_data_np = np.frombuffer(image_data[16:], dtype=np.uint8)

            # Reshape data into individual images
            image_data_np = np.reshape(image_data_np, (num_items, num_rows * num_cols))

            # Convert image data to float and normalise
            image_data_np = image_data_np.astype(np.float)
            image_magnitude = np.sum(image_data_np, axis=1)
            for i in range(num_items):
                image_data_np[i] /= image_magnitude[i]

            # Write to disk
            np.save(filename, image_data_np)

            return image_data_np

def get_label_data(raw_filename, filename, correct_magic):
    if path.exists(filename):
        print("Loading existing data")
        return np.load(filename)
    else:
        with open(raw_filename, "rb") as f:
            label_data = f.read()

            # Unpack header from first 8 bytes of buffer
            magic, num_items = unpack('>II', label_data[:8])
            assert magic == correct_magic

            # Convert remainder of buffer to numpy bytes
            label_data_np = np.frombuffer(label_data[8:], dtype=np.uint8)
            assert label_data_np.shape == (num_items,)

            # Write to disk
            np.save(filename, label_data_np)

            return label_data_np

def get_training_data():
    images = get_image_data("train-images-idx3-ubyte", "training_images.npy", 2051)
    labels = get_label_data("train-labels-idx1-ubyte", "training_labels.npy", 2049)
    assert images.shape[0] == labels.shape[0]

    return images, labels

def get_testing_data():
    images = get_image_data("t10k-images-idx3-ubyte", "testing_images.npy", 2051)
    labels = get_label_data("t10k-labels-idx1-ubyte", "testing_labels.npy", 2049)
    assert images.shape[0] == labels.shape[0]

    return images, labels

def run_mb(para, g= None):
    
    # ----------------------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------------------
    csd= None
    TRAIN= para["TRAIN"]
    SAVE_G= para["SAVE_G"]
    DT = 0.02
    INPUT_SCALE = 400.0
    RECORD_V = not TRAIN
    PN_analysis = para["PN_analysis"]
    KC_analysis = para["KC_analysis"]
    KC_MBON_analysis= False

    RECORDING= para["RECORD_PN_SPIKES"] or para["RECORD_KC_SPIKES"] or para["RECORD_MBON_SPIKES"] or para["RECORD_GGN_SPIKES"]
    
    NUM_PN = para["NUM_PN"]
    NUM_KC = para["NUM_KC"]
    NUM_MBON = para["NUM_MBON"]
    PRESENT_TIME_MS = 20.0

    PRESENT_TIMESTEPS = int(round(PRESENT_TIME_MS / DT))

    PN_KC_COL_LENGTH = para["PN_KC_COL_LENGTH"]

    # PN params - large refractory period so only spikes once per presentation and increased capacitance
    PN_PARAMS = {
        "C": 1.0,
        "TauM": 20.0,
        "Vrest": -60.0,
        "Vreset": -60.0,
        "Vthresh": -50.0,
        "Ioffset": 0.0,
        "TauRefrac": 100.0}
    
    # KC params - standard LIF neurons
    KC_PARAMS = {
        "C": 0.2,
        "TauM": 20.0,
        "Vrest": -60.0,
        "Vreset": -60.0,
        "Vthresh": -50.0,
        "Ioffset": 0.0,
        "TauRefrac": 2.0}

    # MBON params - standard LIF neurons
    MBON_PARAMS = {
        "C": 0.2,
        "TauM": 20.0,
        "Vrest": -60.0,
        "Vreset": -60.0,
        "Vthresh": -50.0,
        "Ioffset": 0.0,
        "TauRefrac": 2.0}
    
    MBON_STIMULUS_CURRENT = 5.0
    PN_KC_WEIGHT = para["PN_KC_WEIGHT"]
    PN_KC_TAU_SYN = 3.0
    KC_MBON_WEIGHT = 0.0
    KC_MBON_TAU_SYN = 3.0
    KC_MBON_PARAMS = {"tau": 15.0,
                      "rho": para["rho"],
                      "eta": para["eta"],
                      "wMin": 0.0,
                      "wMax": para["wMax"]}
    
    
    # ----------------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------------
    # Load MNIST data
    training_images, training_labels = get_training_data()
    testing_images, testing_labels = get_testing_data()

    assert training_images.shape[1] == NUM_PN
    assert testing_images.shape[1] == NUM_PN
    assert np.max(training_labels) == (NUM_MBON - 1)
    assert np.max(testing_labels) == (NUM_MBON - 1)

    # Current source model, allowing current to be injected into neuron from variable
    cs_model = genn_model.create_custom_current_source_class(
        "cs_model",
        var_name_types=[("magnitude", "scalar")],
        injection_code="$(injectCurrent, $(magnitude));")

    # STDP synapse with additive weight dependence
    symmetric_stdp = genn_model.create_custom_weight_update_class(
        "symmetric_stdp",
        param_names=["tau", "rho", "eta", "wMin", "wMax"],
        var_name_types=[("g", "scalar")],
        sim_code=
        """
        const scalar dt = $(t) - $(sT_post);
        const scalar timing = exp(-dt / $(tau)) - $(rho);
        const scalar newWeight = $(g) + ($(eta) * timing);
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
        """,
        learn_post_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        const scalar timing = fmax(exp(-dt / $(tau)) - $(rho), -0.1*$(rho));
        const scalar newWeight = $(g) + ($(eta) * timing);
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
        """,
        is_pre_spike_time_required=True,
        is_post_spike_time_required=True)
    
    # custom IF neuron for gain control
    IF_neuron = genn_model.create_custom_neuron_class(
        "IF_neuron",
        param_names=["theta"],
        var_name_types=[("V", "scalar")],
        sim_code=
        """
        $(V)+= $(Isyn);
        """,
        threshold_condition_code=
        """
        $(V) >= $(theta)
        """,
        reset_code=
        """
        $(V)= 0.0;
        """)

    # Create model
    model = genn_model.GeNNModel("float", "mnist_mb")
    model.dT = DT
    model._model.set_seed(1337)

    # Create neuron populations
    lif_init = {"V": -60.0, "RefracTime": 0.0}
    pn = model.add_neuron_population("pn", NUM_PN, "LIF", PN_PARAMS, lif_init)
    kc = model.add_neuron_population("kc", NUM_KC, "LIF", KC_PARAMS, lif_init)
    mbon = model.add_neuron_population("mbon", NUM_MBON, "LIF", MBON_PARAMS, lif_init)
    ggn= model.add_neuron_population("ggn", 1, IF_neuron, {"theta":para["num_KC"]}, {"V": 0.0})

    # Turn on spike recording
    pn.spike_recording_enabled = para["RECORD_PN_SPIKES"]
    kc.spike_recording_enabled = para["RECORD_KC_SPIKES"]
    mbon.spike_recording_enabled = para["RECORD_MBON_SPIKES"]
    ggn.spike_recording_enabled= para["RECORD_GGN_SPIKES"]

    # Create current sources to deliver input and supervision to network
    pn_input = model.add_current_source("pn_input", cs_model, pn , {}, {"magnitude": 0.0})
    mbon_input = model.add_current_source("mbon_input", cs_model, mbon , {}, {"magnitude": 0.0})

    # Create synapse populations
    pn_kc = model.add_synapse_population("pn_kc", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                         pn, kc,
                                         "StaticPulse", {}, {"g": PN_KC_WEIGHT}, {}, {},
                                         "ExpCurr", {"tau": PN_KC_TAU_SYN}, {},
                                         genn_model.init_connectivity("FixedNumberPreWithReplacement", {"colLength": PN_KC_COL_LENGTH}))

    kc_ggn = model.add_synapse_population("kc_ggn", "DENSE_GLOBALG", genn_wrapper.NO_DELAY, kc, ggn, "StaticPulse", {}, {"g": 1.0}, {}, {}, "DeltaCurr", {}, {})

    ggn_kc = model.add_synapse_population("ggn_kc", "DENSE_GLOBALG", genn_wrapper.NO_DELAY, ggn, kc, "StaticPulse", {}, {"g": -5.0}, {}, {}, "ExpCurr", {"tau": 5.0}, {})

    if TRAIN:
        kc_mbon = model.add_synapse_population("kc_mbon", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                               kc, mbon,
                                               symmetric_stdp, KC_MBON_PARAMS, {"g": np.ones(NUM_KC*NUM_MBON)*KC_MBON_WEIGHT}, {}, {},
                                               "ExpCurr", {"tau": KC_MBON_TAU_SYN}, {})
    else:
        kc_mbon = model.add_synapse_population("kc_mbon", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                               kc, mbon,
                                               "StaticPulse", {}, {"g": g.flatten()}, {}, {},
                                               "ExpCurr", {"tau": KC_MBON_TAU_SYN}, {})

    mbon_mbon = model.add_synapse_population("mbon_mbon", "DENSE_GLOBALG", genn_wrapper.NO_DELAY, mbon, mbon, "StaticPulse", {}, {"g": -1.0}, {}, {}, "ExpCurr", {"tau": 5.0}, {})

    # Build model and load it
    model.build()
    model.load(num_recording_timesteps=PRESENT_TIMESTEPS)

    # Get views to efficiently access state variables
    pn_input_current_view = pn_input.vars["magnitude"].view
    pn_refrac_time_view = pn.vars["RefracTime"].view
    pn_v_view = pn.vars["V"].view
    ggn_v_view= ggn.vars["V"].view
    kc_refrac_time_view = kc.vars["RefracTime"].view
    kc_v_view = kc.vars["V"].view
    mbon_input_current_view = mbon_input.vars["magnitude"].view
    mbon_refrac_time_view = mbon.vars["RefracTime"].view
    mbon_v_view = mbon.vars["V"].view
    kc_mbon_g_view = kc_mbon.vars["g"].view

    pn_kc_insyn_view = pn_kc._assign_ext_ptr_array("inSyn", NUM_KC, "scalar")
    kc_mbon_insyn_view = kc_mbon._assign_ext_ptr_array("inSyn", NUM_MBON, "scalar")
    ggn_kc_insyn_view= ggn_kc._assign_ext_ptr_array("inSyn", NUM_KC, "scalar")
    mbon_mbon_insyn_view= mbon_mbon._assign_ext_ptr_array("inSyn", NUM_MBON, "scalar")

    if TRAIN:
        kc_spike_time_view = kc._assign_ext_ptr_array("sT", NUM_KC, "scalar")
        mbon_spike_time_view = mbon._assign_ext_ptr_array("sT", NUM_MBON, "scalar")

    plot = para["plot"]

    images = training_images# if TRAIN else testing_images
    labels = training_labels# if TRAIN else testing_labels

    # Loop through stimuli
    pn_spikes = ([], [])
    kc_spikes = ([], [])
    mbon_spikes = ([], [])
    ggn_spikes= ([], [])
    mbon_v = []
    ggn_v= []
    STIM_MOD = para["STIM_MOD"]
    if TRAIN:
        STIM_SHIFT= para["STIM_SHIFT_TRAIN"]
        NUM_STIM = para["NUM_STIM_TRAIN"]
    else:
        STIM_SHIFT= para["STIM_SHIFT_TEST"]
        NUM_STIM = para["NUM_STIM_TEST"]

    for rs in range(STIM_SHIFT,NUM_STIM+STIM_SHIFT):
        s= rs % STIM_MOD
        if para["SHUFFLE"]:
            if s == 0:
                idx= np.arange(STIM_MOD)
                np.random.shuffle(idx)
                images[STIM_SHIFT:STIM_SHIFT+STIM_MOD]= images[idx+STIM_SHIFT]
                labels[STIM_SHIFT:STIM_SHIFT+STIM_MOD]= labels[idx+STIM_SHIFT]
        if rs % 500 == 0:
            kc_mbon.pull_var_from_device("g")
            print("{}: gmax: {}/{}, gmean: {}, nHigh: {}".format(rs, np.max(kc_mbon_g_view), para["wMax"], np.mean(kc_mbon_g_view), len(kc_mbon_g_view[kc_mbon_g_view > 0.9*max(kc_mbon_g_view)])))
        # Set training image
        pn_input_current_view[:] = images[s] * INPUT_SCALE
        pn_input.push_var_to_device("magnitude")
    
        # Turn on correct output neuron
        if TRAIN:
            mbon_input_current_view[:] = 0
            mbon_input_current_view[labels[s]] = MBON_STIMULUS_CURRENT
            mbon_input.push_var_to_device("magnitude")
        
        # Loop through stimuli presentation
        for i in range(PRESENT_TIMESTEPS):
            model.step_time()
        
            if RECORD_V:
                mbon.pull_var_from_device("V")
                mbon_v.append(np.copy(mbon_v_view))
                #ggn.pull_var_from_device("V")
                #ggn_v.append(np.copy(ggn_v_view))
    
        # Reset neurons
        pn_refrac_time_view[:] = 0.0
        pn_v_view[:] = -60.0
        kc_refrac_time_view[:] = 0.0
        kc_v_view[:] = -60.0
        mbon_refrac_time_view[:] = 0.0
        mbon_v_view[:] = -60.0
        ggn_v_view[:]= 0.0;
        pn_kc_insyn_view[:] = 0.0
        kc_mbon_insyn_view[:] = 0.0
        ggn_kc_insyn_view[:] = 0.0
        mbon_mbon_insyn_view[:] = 0.0
        pn.push_var_to_device("RefracTime")
        pn.push_var_to_device("V")
        kc.push_var_to_device("RefracTime")
        kc.push_var_to_device("V")
        mbon.push_var_to_device("RefracTime")
        mbon.push_var_to_device("V")
        ggn.push_var_to_device("V")
        model.push_var_to_device("pn_kc", "inSyn")
        model.push_var_to_device("kc_mbon", "inSyn")
        model.push_var_to_device("ggn_kc", "inSyn")
        model.push_var_to_device("mbon_mbon", "inSyn")

        if TRAIN:
            kc_spike_time_view[:] = -np.finfo(np.float32).max
            mbon_spike_time_view[:] = -np.finfo(np.float32).max
            model.push_var_to_device("SpikeTimes", "mbon")
            model.push_var_to_device("SpikeTimes", "kc")

        if RECORDING:
            model.pull_recording_buffers_from_device();
            
            if para["RECORD_PN_SPIKES"]:
                pn_spike_times, pn_spike_ids = pn.spike_recording_data
                pn_spikes[0].append(pn_spike_times)
                pn_spikes[1].append(pn_spike_ids)
            if para["RECORD_KC_SPIKES"]:
                kc_spike_times, kc_spike_ids = kc.spike_recording_data
                kc_spikes[0].append(kc_spike_times)
                kc_spikes[1].append(kc_spike_ids)
            if para["RECORD_MBON_SPIKES"]:
                mbon_spike_times, mbon_spike_ids = mbon.spike_recording_data
                mbon_spikes[0].append(mbon_spike_times)
                mbon_spikes[1].append(mbon_spike_ids)
            if para["RECORD_GGN_SPIKES"]:
                ggn_spike_times, ggn_spike_ids = ggn.spike_recording_data
                ggn_spikes[0].append(ggn_spike_times)
                ggn_spikes[1].append(ggn_spike_ids)
            
    if TRAIN:
        # Save weights
        kc_mbon.pull_var_from_device("g")
        kc_mbon_g_view = np.reshape(kc_mbon_g_view, (NUM_KC, NUM_MBON))
        if SAVE_G:
             basename= para["basename"]
             np.save(basename+"-kc_mbon_g.npy", kc_mbon_g_view)
    else:
        # save classification results
        good= 0.0
        cnt= 0.0
        for t, s, l in zip(mbon_spikes[0], mbon_spikes[1], labels[STIM_SHIFT:STIM_SHIFT+NUM_STIM]):
            if len(s) > 0:
                first_spike = np.argmin(t)
                classification = s[first_spike]
                good+= 1 if classification == l else 0
                cnt+= 1
        with open("test.out","a") as f:
            if cnt > 0:
                f.write(para["basename"]+": Classification accuracy: {}\n".format(good/cnt))
                f.write("{} of {} responses.\n".format(cnt, para["NUM_STIM_TEST"]))
                print(para["basename"]+": Classification accuracy: {}\n".format(good/cnt))
                print("{} of {} responses.\n".format(cnt, para["NUM_STIM_TEST"]))
            else:
                f.write(para["basename"]+": No output spikes! \n")
            f.close()
            
    stimuli_bounds = np.arange(0.0, NUM_STIM * PRESENT_TIME_MS, PRESENT_TIME_MS)
    if plot:
        spike_fig, spike_axes = plt.subplots(4 if RECORD_V else 3, sharex="col")
    
        # Plot spikes
        spike_axes[0].scatter(np.concatenate(pn_spikes[0]), np.concatenate(pn_spikes[1]), s=1)
        spike_axes[1].scatter(np.concatenate(kc_spikes[0]), np.concatenate(kc_spikes[1]), s=1)
        spike_axes[2].scatter(np.concatenate(mbon_spikes[0]), np.concatenate(mbon_spikes[1]), s=1)
        spike_axes[2].scatter(np.concatenate(ggn_spikes[0]), np.concatenate(ggn_spikes[1]), c="r", s=1)
        spike_axes[2].set_xlim([0,600])
        # Plot voltages
        if RECORD_V:
            spike_axes[3].plot(np.arange(0.0, PRESENT_TIME_MS * NUM_STIM, DT), np.vstack(mbon_v))
            #        spike_axes[3].plot(np.arange(0.0, PRESENT_TIME_MS * NUM_STIM, DT), np.vstack(ggn_v),"--")
    
        # Mark stimuli changes on figure
        spike_axes[0].vlines(stimuli_bounds, ymin=0, ymax=NUM_PN, linestyle="--")
        spike_axes[1].vlines(stimuli_bounds, ymin=0, ymax=NUM_KC, linestyle="--")
        spike_axes[2].vlines(stimuli_bounds, ymin=0, ymax=NUM_MBON, linestyle="--")
    
        # Label axes
        spike_axes[0].set_title("PN")
        spike_axes[1].set_title("KC")
        spike_axes[2].set_title("MBON")
    
        if RECORD_V:
            spike_axes[3].set_title("MBON/GGN")
        
        # Show classification output
        for b, t, s, l in zip(stimuli_bounds, mbon_spikes[0], mbon_spikes[1], labels[STIM_SHIFT:STIM_SHIFT+NUM_STIM]):
            if len(s) > 0:
                first_spike = np.argmin(t)
                classification = s[first_spike]
                #classification = np.argmax(np.bincount(s, minlength=NUM_MBON))
                colour = "green" if classification == l else "red"
                spike_axes[2].hlines(classification, b, b + PRESENT_TIME_MS, 
                                     color=colour, alpha=0.5)

        # Show training labels
        for i, x in enumerate(stimuli_bounds):
            spike_axes[0].text(x, 20, labels[STIM_SHIFT+i])
        
        fig, axis = plt.subplots()
        axis.hist(kc_mbon_g_view.flatten(), bins=100)
        fig, axis = plt.subplots()
        axis.hist(np.concatenate(mbon_spikes[1]), bins=10)
        #unique_neurons = [np.unique(k) for k in kc_spikes[1]]
        #for (i, a), (j, b) in combinations(enumerate(unique_neurons), 2):
        #    print("label %u vs %u = %u" % (labels[i], labels[j], len(np.intersect1d(a, b))))
        if para["INTERACTIVE"]:
            plt.show()
        plt.close()

    if para["RECORD_MBON_SPIKES"]:
        MBON_sN= []
        for s in mbon_spikes[0]:
            MBON_sN.append(len(s))
        
    if PN_analysis:
        fig= plt.figure()
        for i in range(np.minimum(NUM_STIM,15*15)):
            x= np.zeros((NUM_PN,))
            x[pn_spikes[1][i]]= PRESENT_TIME_MS-(pn_spikes[0][i] - stimuli_bounds[i])
            plt.subplot(15,15,i+1)
            plt.imshow(np.reshape(x,(28,28)), vmin= 0, vmax= PRESENT_TIME_MS)
            plt.text(0,5,labels[STIM_SHIFT+i],c="w")
        idx= np.argsort(labels[STIM_SHIFT:STIM_SHIFT+NUM_STIM])
        if para["INTERACTIVE"]:
            plt.show()
        plt.close()
        csd= np.zeros((NUM_STIM,NUM_STIM))
        idx= np.argsort(labels[STIM_SHIFT:STIM_SHIFT+NUM_STIM])
        for i in range(NUM_STIM):
            x= np.zeros((NUM_KC,))
            x[pn_spikes[1][i]]= 1
            for j in range(NUM_STIM):
                y= np.zeros((NUM_KC,))
                y[pn_spikes[1][j]]= 1
                csd[i,j]= np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
        plt.figure()
        plt.imshow(csd[idx,:][:,idx], interpolation="none")
        plt.colorbar()
        if para["INTERACTIVE"]:
            plt.show()
        plt.close()
        
    if KC_analysis:
        fig= plt.figure()
        first= True
        the_lbl= 4
        idx= np.array(range(NUM_KC))
        for i in range(np.minimum(NUM_STIM,15*15)):
            x= np.zeros((NUM_KC,))
            x[kc_spikes[1][i]]= PRESENT_TIME_MS-(kc_spikes[0][i] - stimuli_bounds[i])
            if (labels[STIM_SHIFT+i] == the_lbl) and first:
                idx= np.argsort(x)
                first= False
            
            plt.subplot(15,15,i+1)
            plt.imshow(np.reshape(x[idx],(100,200)), vmin= 0, vmax= PRESENT_TIME_MS/3)
            plt.text(0,20,labels[STIM_SHIFT+i],c="w")
        if para["INTERACTIVE"]:
            plt.show()
        plt.close()
        csd= np.zeros((NUM_STIM,NUM_STIM))
        idx= np.argsort(labels[STIM_SHIFT:STIM_SHIFT+NUM_STIM])
        scsd= np.zeros((10,10))
        lbl= labels[STIM_SHIFT:STIM_SHIFT+NUM_STIM]
        for i in range(NUM_STIM):
            x= np.zeros((NUM_KC,))
            x[kc_spikes[1][i]]= 1
            for j in range(NUM_STIM):
                y= np.zeros((NUM_KC,))
                y[kc_spikes[1][j]]= 1
                csd[i,j]= np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
                scsd[int(lbl[i]),int(lbl[j])]+= csd[i,j]
        print(scsd)
        intra= np.trace(scsd)
        inter= np.sum(np.sum(scsd))-np.trace(scsd)
        print("intra: {}".format(intra))
        print("inter: {}".format(inter))
        with open("dist.dat", "a") as f:
            f.write("intra: {}, inter: {}, intra/inter: {} \n".format(intra,inter,intra/inter))
            f.close()
        plt.figure()
        plt.imshow(csd[idx,:][:,idx], vmin= 0, vmax= 0.5, interpolation="none", cmap="jet")
        plt.colorbar()
        plt.savefig(para["basename"]+"_KC_corr.png")
        if para["INTERACTIVE"]:
            plt.show()
        plt.close()
        
    if KC_MBON_analysis:
        kc_mbon.pull_var_from_device("g")
        kc_mbon_g_view = np.reshape(kc_mbon_g_view, (NUM_KC, NUM_MBON))
        np.save("kc_mbon_g.npy", kc_mbon_g_view)
        for i in range(NUM_STIM):
            x= kc_mbon_g_view[:,labels[STIM_SHIFT+i]]
            idx= np.argsort(x)
            x= np.reshape(x[idx],(200,100))
            y= np.zeros((NUM_KC,))
            y[kc_spikes[1][i]]= 2*KC_MBON_PARAMS["wMax"]
            y= np.reshape(y[idx],(200,100))
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(x)
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(y)
            plt.colorbar()
            first_spike = np.argmin(mbon_spikes[0][i])
            j = mbon_spikes[1][i][first_spike]
            print("mbon_spk: {},{}, first: {}".format(mbon_spikes[0][i],mbon_spikes[1][i],j))
            x= kc_mbon_g_view[:,j]
            x= np.reshape(x[idx],(200,100))
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(x)
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(y)
            print("current: {}, confused: {}".format(labels[STIM_SHIFT+i],j))
            if para[INTERACTIVE]:
                plt.show()
            plt.close()
    return (np.copy(kc_mbon_g_view))
