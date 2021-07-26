import numpy as np

def preprocess_events(events, num_input_neurons):
    # Calculate cumulative sum of each neuron's spike count
    end_spikes = np.cumsum(np.bincount(events[:,1].astype(int), 
                                       minlength=num_input_neurons))
    
    # Sort events first by neuron id and then by time and use to order spike times
    spike_times = events[np.lexsort((events[:,0], events[:,1])),0]
   
    # Return end spike indices and spike times (converted to ms)
    return end_spikes, (spike_times / 1000.0)

def get_start_spikes(end_spikes):
    start_spikes = np.empty_like(end_spikes)
    if end_spikes.ndim == 1:
        start_spikes[0] = 0
        start_spikes[1:] = end_spikes[:-1]
    else:
        start_spikes[0,0] = 0
        start_spikes[1:,0] = end_spikes[:-1,-1]
        start_spikes[:,1:] = end_spikes[:,:-1]
    
    return start_spikes

def concatenate_events(events):
    # Check that all stimuli are for same number of neurons
    assert all(len(e[0]) == len(events[0][0]) for e in events)
    
    # Extract seperate lists of each stimuli's end spike indices and spike times
    end_spikes, spike_times = zip(*events)
    
    # Make corresponding array of start spikes
    start_spikes = [get_start_spikes(e) for e in end_spikes]
    
    # Create empty array to hold spike times
    concat_spike_times = np.empty(sum(len(s) for s in spike_times))
    
    # End spikes are simply sum of the end spikes
    concat_end_spikes = np.sum(end_spikes, axis=0)
    
    # Loop through neurons
    start_idx = 0
    for i in range(len(concat_end_spikes)):
        # Loop through the start and end spike; 
        # and the spike times from each stimuli
        for s, e, t in zip(start_spikes, end_spikes, spike_times):
            # Copy block of spike times into place and advance start_idx
            num_spikes = e[i] - s[i]
            concat_spike_times[start_idx:start_idx + num_spikes] = t[s[i]:e[i]]
            start_idx += num_spikes
    
    return concat_end_spikes, concat_spike_times

def batch_events(events, batch_size):
    # Check that there aren't more stimuli than batch size 
    # and that all stimuli are for same number of neurons
    num_neurons = len(events[0][0])
    assert len(events) <= batch_size
    assert all(len(e[0]) == num_neurons for e in events)

    # Extract seperate lists of each stimuli's end spike indices and spike times
    end_spikes, spike_times = zip(*events)
    
    # Calculate cumulative sum of spikes counts across batch
    cum_spikes_per_stimuli = np.concatenate(([0], np.cumsum([len(s) for s in spike_times])))
    
    # Add this cumulative sum onto the end spikes array of each stimuli
    # **NOTE** zip will stop before extra cum_spikes_per_stimuli value
    batch_end_spikes = np.vstack([c + e
                                  for e, c in zip(end_spikes, cum_spikes_per_stimuli)])
   
    # If this isn't a full batch
    if len(events) < batch_size:
        # Create spike padding for remainder of batch
        spike_padding = np.ones((batch_size - len(events), num_neurons), dtype=int) * cum_spikes_per_stimuli[-1]
        
        # Stack onto end spikes
        batch_end_spikes = np.vstack((batch_end_spikes, spike_padding))
    
    # Concatenate together all spike times
    batch_spike_times = np.concatenate(spike_times)
    
    return batch_end_spikes, batch_spike_times

def preprocess_data(data_loader, batch_size, num_input_neurons):
    batch_data = []
    data_iter = iter(data_loader)
    for d in data_iter:
        # Unzip batch data into events and labels
        if batch_size == 1:
            batch_events = [d[0]]
            batch_labels = [d[1]]
        else:
            batch_events, batch_labels = zip(*d)
        
        # Sort events first by neuron id and then by time and use to order spike times
        batch_spike_times = [e[np.lexsort((e[:,1], e[:,1])),0]
                             for e in batch_events]
        
        # Convert events ids to integer
        batch_id_int = [e[:,1].astype(int) for e in batch_events]
        
        # Calculate starting index of spikes in each stimuli across the batch
        # **NOTE** we calculate extra end value for use if padding is required
        cum_spikes_per_stimuli = np.concatenate(([0], np.cumsum([len(e) for e in batch_id_int])))
        
        # Add this cumulative sum onto the cumulative sum of spikes per neuron
        # **NOTE** zip will stop before extra cum_spikes_per_stimuli value
        end_spikes = np.vstack([c + np.cumsum(np.bincount(e, minlength=num_input_neurons)) 
                                for e, c in zip(batch_id_int, cum_spikes_per_stimuli)])
        
        # Build start spikes array
        start_spikes = np.empty((len(batch_events), num_input_neurons), dtype=int)
        start_spikes[:,0] = cum_spikes_per_stimuli[:-1]
        start_spikes[:,1:] = end_spikes[:,:-1]
        
        # If this isn't a full batch
        if len(batch_events) != batch_size:
            spike_padding = np.ones((batch_size - len(batch_events), num_input_neurons), dtype=int) * cum_spikes_per_stimuli[-1]
            end_spikes = np.vstack((end_spikes, spike_padding))
            start_spikes = np.vstack((start_spikes, spike_padding))
        
        # Concatenate together all spike times
        spike_times = np.concatenate(batch_spike_times)
        
        # Add tuple of pre-processed data to list
        batch_data.append((start_spikes, end_spikes, spike_times, batch_labels))
    return batch_data