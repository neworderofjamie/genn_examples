import numpy as np

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