import numpy as np

from collections import namedtuple
from copy import copy

PreprocessedEvents = namedtuple("PreprocessedEvents", ["end_spikes", "spike_times"])

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
    assert all(len(e.spike_times) == len(events[0].spike_times) for e in events)
    
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
    
    return PreprocessedEvents(concat_end_spikes, concat_spike_times)

def batch_events(events, batch_size):
    # Check that there aren't more stimuli than batch size 
    # and that all stimuli are for same number of neurons
    num_neurons = len(events[0].end_spikes)
    assert len(events) <= batch_size
    assert all(len(e.end_spikes) == num_neurons for e in events)

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
    
    return PreprocessedEvents(batch_end_spikes, batch_spike_times)


class DataLoader:
    def __init__(self, dataset, shuffle=False, batch_size=1):
        self.dataset = dataset
        self.length = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Allocate numpy array for labels
        self._labels = np.empty(self.length, dtype=int)
        self._preprocessed_events = []

        # Loop through dataset
        for i in range(self.length):
            events, label = self.dataset[i]
            
            # Store label
            self._labels[i] = label
            
            # Preprocess events and add to list
            self._preprocessed_events.append(self._preprocess(events))
    def __iter__(self):
        class DataIter:
            def __init__(self, data_loader):
                self.data_loader = data_loader
                self.iteration = 0
                self.indices = np.arange(0, self.data_loader.length, dtype=int)
                if self.data_loader.shuffle:
                    np.random.shuffle(self.indices)

            def __iter__(self):
                return self

            def __next__(self):
                dl = self.data_loader
                if self.iteration >= dl.length:
                    raise StopIteration
                elif dl.batch_size == 1:
                    idx = self.indices[self.iteration]
                    self.iteration += 1
                    return dl._preprocessed_events[idx], dl._labels[idx]
                else:
                    # Get start and end of batch (end might be past end of indices)
                    begin = self.iteration
                    end = self.iteration + dl.batch_size
                    
                    # Get indices and thus slice of data
                    inds = self.indices[begin:end]

                    # Add number of indices to iteration count
                    # (will take into account size of dataset)
                    self.iteration += len(inds)
                    
                    # Return list of data
                    return ([dl._preprocessed_events[i] for i in inds],
                            dl._labels[inds])

        return DataIter(self)

    def __len__(self):
        return int(np.ceil(self.length / float(self.batch_size)))
    
    def _preprocess(self, events):
        # Calculate cumulative sum of each neuron's spike count
        num_input_neurons = np.product(self.dataset.sensor_size) 
        end_spikes = np.cumsum(np.bincount(events[:,1].astype(int), 
                                           minlength=num_input_neurons))
        
        # Sort events first by neuron id and then by time and use to order spike times
        spike_times = events[np.lexsort((events[:,0], events[:,1])),0]
       
        # Return end spike indices and spike times (converted to ms)
        return PreprocessedEvents(end_spikes, (spike_times / 1000.0))